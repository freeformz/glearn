package metrics

import "sort"

// ROCAUC computes the area under the Receiver Operating Characteristic curve
// for binary classification. yTrue contains the true binary labels (0 or 1)
// and yScores contains the predicted scores (e.g., probabilities for the
// positive class).
//
// The ROC curve is computed by varying the decision threshold, and the area
// is calculated using the trapezoidal rule.
//
// It panics if the inputs are empty or have different lengths.
func ROCAUC(yTrue, yScores []float64) float64 {
	validateInputs(yTrue, yScores)

	// Count positives and negatives.
	nPos := 0
	nNeg := 0
	for _, y := range yTrue {
		if y == 1.0 {
			nPos++
		} else {
			nNeg++
		}
	}

	// If all samples are of one class, AUC is undefined. Return 0.
	if nPos == 0 || nNeg == 0 {
		return 0.0
	}

	// Create index array sorted by descending score.
	n := len(yTrue)
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		si, sj := yScores[indices[i]], yScores[indices[j]]
		if si != sj {
			return si > sj // Descending by score.
		}
		// Tie-breaking: put actual negatives first so we get a pessimistic curve
		// for tied scores (matches sklearn behavior).
		return yTrue[indices[i]] < yTrue[indices[j]]
	})

	// Walk through sorted predictions to build the ROC curve and compute AUC
	// using the trapezoidal rule incrementally.
	auc := 0.0
	prevFPR := 0.0
	prevTPR := 0.0
	tp := 0.0
	fp := 0.0

	fPos := float64(nPos)
	fNeg := float64(nNeg)

	for i := range n {
		idx := indices[i]
		if yTrue[idx] == 1.0 {
			tp++
		} else {
			fp++
		}

		// Only emit a point when the score changes or we're at the last element.
		if i == n-1 || yScores[indices[i]] != yScores[indices[i+1]] {
			fpr := fp / fNeg
			tpr := tp / fPos

			// Trapezoidal rule: area of trapezoid.
			auc += (fpr - prevFPR) * (tpr + prevTPR) / 2.0

			prevFPR = fpr
			prevTPR = tpr
		}
	}

	return auc
}
