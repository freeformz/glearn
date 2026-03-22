package tree

import (
	"math"
	"sort"
)

// splitResult holds the result of finding the best split for a node.
type splitResult struct {
	feature   int     // feature index
	threshold float64 // split threshold
	gain      float64 // impurity reduction
	leftIdx   []int   // indices of samples going left
	rightIdx  []int   // indices of samples going right
}

// giniImpurity computes the Gini impurity for a set of class counts.
// counts maps class label to count, total is the total number of samples.
func giniImpurity(counts map[float64]int, total int) float64 {
	if total == 0 {
		return 0
	}
	t := float64(total)
	impurity := 1.0
	for _, count := range counts {
		p := float64(count) / t
		impurity -= p * p
	}
	return impurity
}

// entropyImpurity computes the entropy for a set of class counts.
func entropyImpurity(counts map[float64]int, total int) float64 {
	if total == 0 {
		return 0
	}
	t := float64(total)
	entropy := 0.0
	for _, count := range counts {
		if count == 0 {
			continue
		}
		p := float64(count) / t
		entropy -= p * math.Log2(p)
	}
	return entropy
}

// mseImpurity computes the mean squared error for a set of target values.
// It equals the variance of the values.
func mseImpurity(values []float64) float64 {
	n := len(values)
	if n == 0 {
		return 0
	}
	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(n)
	mse := 0.0
	for _, v := range values {
		d := v - mean
		mse += d * d
	}
	return mse / float64(n)
}

// classificationImpurityFunc returns the impurity function for the given criterion.
func classificationImpurityFunc(criterion string) func(map[float64]int, int) float64 {
	switch criterion {
	case "entropy":
		return entropyImpurity
	default: // "gini"
		return giniImpurity
	}
}

// indexedFeature is a helper for sorting samples by a single feature value.
type indexedFeature struct {
	idx int
	val float64
}

// findBestClassificationSplit finds the best binary split for classification.
// X is accessed via the raw data slice, y contains class labels.
// indices are the sample indices to consider, nFeatures is the number of features,
// stride is the row stride in the raw data.
func findBestClassificationSplit(
	xData []float64, stride int,
	y []float64,
	indices []int,
	nFeatures int,
	criterion string,
	minSamplesLeaf int,
) *splitResult {
	impurityFunc := classificationImpurityFunc(criterion)
	n := len(indices)

	// Compute parent class counts.
	parentCounts := make(map[float64]int)
	for _, idx := range indices {
		parentCounts[y[idx]]++
	}
	parentImpurity := impurityFunc(parentCounts, n)

	if parentImpurity == 0 {
		// Already pure.
		return nil
	}

	bestGain := -1.0
	var best *splitResult

	// Pre-allocate the sorted slice once and reuse.
	sorted := make([]indexedFeature, n)

	for feat := 0; feat < nFeatures; feat++ {
		// Build sorted order for this feature.
		for i, idx := range indices {
			sorted[i] = indexedFeature{idx: idx, val: xData[idx*stride+feat]}
		}
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i].val < sorted[j].val
		})

		// Scan through sorted values to find the best threshold.
		leftCounts := make(map[float64]int)
		rightCounts := make(map[float64]int)
		for k, v := range parentCounts {
			rightCounts[k] = v
		}

		for i := 0; i < n-1; i++ {
			idx := sorted[i].idx
			cls := y[idx]
			leftCounts[cls]++
			rightCounts[cls]--
			if rightCounts[cls] == 0 {
				delete(rightCounts, cls)
			}

			leftN := i + 1
			rightN := n - leftN

			// Skip if the feature value is the same as the next (no valid threshold between them).
			if sorted[i].val == sorted[i+1].val {
				continue
			}

			// Check minimum samples in each leaf.
			if leftN < minSamplesLeaf || rightN < minSamplesLeaf {
				continue
			}

			leftImpurity := impurityFunc(leftCounts, leftN)
			rightImpurity := impurityFunc(rightCounts, rightN)

			// Weighted impurity reduction.
			gain := parentImpurity -
				(float64(leftN)/float64(n))*leftImpurity -
				(float64(rightN)/float64(n))*rightImpurity

			if gain > bestGain {
				bestGain = gain
				threshold := (sorted[i].val + sorted[i+1].val) / 2.0
				leftIdx := make([]int, leftN)
				rightIdx := make([]int, rightN)
				for j := 0; j <= i; j++ {
					leftIdx[j] = sorted[j].idx
				}
				for j := i + 1; j < n; j++ {
					rightIdx[j-i-1] = sorted[j].idx
				}
				best = &splitResult{
					feature:   feat,
					threshold: threshold,
					gain:      gain,
					leftIdx:   leftIdx,
					rightIdx:  rightIdx,
				}
			}
		}
	}

	return best
}

// findBestRegressionSplit finds the best binary split for regression.
func findBestRegressionSplit(
	xData []float64, stride int,
	y []float64,
	indices []int,
	nFeatures int,
	minSamplesLeaf int,
) *splitResult {
	n := len(indices)

	// Compute parent MSE.
	parentMean := 0.0
	for _, idx := range indices {
		parentMean += y[idx]
	}
	parentMean /= float64(n)

	parentMSE := 0.0
	for _, idx := range indices {
		d := y[idx] - parentMean
		parentMSE += d * d
	}
	parentMSE /= float64(n)

	if parentMSE == 0 {
		return nil
	}

	bestGain := 0.0
	var best *splitResult

	sorted := make([]indexedFeature, n)

	for feat := 0; feat < nFeatures; feat++ {
		for i, idx := range indices {
			sorted[i] = indexedFeature{idx: idx, val: xData[idx*stride+feat]}
		}
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i].val < sorted[j].val
		})

		// Use running sums for efficient MSE computation.
		leftSum := 0.0
		leftSumSq := 0.0
		totalSum := 0.0
		totalSumSq := 0.0
		for _, sf := range sorted {
			v := y[sf.idx]
			totalSum += v
			totalSumSq += v * v
		}

		for i := 0; i < n-1; i++ {
			v := y[sorted[i].idx]
			leftSum += v
			leftSumSq += v * v

			leftN := i + 1
			rightN := n - leftN

			if sorted[i].val == sorted[i+1].val {
				continue
			}

			if leftN < minSamplesLeaf || rightN < minSamplesLeaf {
				continue
			}

			rightSum := totalSum - leftSum
			rightSumSq := totalSumSq - leftSumSq

			// MSE = E[X^2] - (E[X])^2
			leftMean := leftSum / float64(leftN)
			leftMSE := leftSumSq/float64(leftN) - leftMean*leftMean

			rightMean := rightSum / float64(rightN)
			rightMSE := rightSumSq/float64(rightN) - rightMean*rightMean

			gain := parentMSE -
				(float64(leftN)/float64(n))*leftMSE -
				(float64(rightN)/float64(n))*rightMSE

			if gain > bestGain {
				bestGain = gain
				threshold := (sorted[i].val + sorted[i+1].val) / 2.0
				leftIdx := make([]int, leftN)
				rightIdx := make([]int, rightN)
				for j := 0; j <= i; j++ {
					leftIdx[j] = sorted[j].idx
				}
				for j := i + 1; j < n; j++ {
					rightIdx[j-i-1] = sorted[j].idx
				}
				best = &splitResult{
					feature:   feat,
					threshold: threshold,
					gain:      gain,
					leftIdx:   leftIdx,
					rightIdx:  rightIdx,
				}
			}
		}
	}

	return best
}
