package modelselection

import (
	"fmt"
	"math/rand/v2"
	"sort"

	"github.com/freeformz/glearn"
)

// Fold holds the training and testing indices for a single cross-validation fold.
type Fold struct {
	TrainIndices []int
	TestIndices  []int
}

// KFold provides K-Fold cross-validation splitting.
//
// NSplits is the number of folds (must be >= 2).
// Shuffle controls whether to shuffle indices before splitting.
// Seed controls the random shuffling for reproducibility.
type KFold struct {
	NSplits int
	Shuffle bool
	Seed    int64
}

// Split generates cross-validation folds for nSamples data points.
// Each sample appears in exactly one test fold.
func (kf KFold) Split(nSamples int) ([]Fold, error) {
	if kf.NSplits < 2 {
		return nil, fmt.Errorf("glearn/modelselection: %w: NSplits must be >= 2, got %d",
			glearn.ErrInvalidParameter, kf.NSplits)
	}
	if nSamples < kf.NSplits {
		return nil, fmt.Errorf("glearn/modelselection: %w: nSamples (%d) must be >= NSplits (%d)",
			glearn.ErrInvalidParameter, nSamples, kf.NSplits)
	}

	indices := make([]int, nSamples)
	for i := range nSamples {
		indices[i] = i
	}
	if kf.Shuffle {
		rng := rand.New(rand.NewPCG(uint64(kf.Seed), 0))
		rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}

	folds := make([]Fold, kf.NSplits)
	foldSize := nSamples / kf.NSplits
	remainder := nSamples % kf.NSplits

	start := 0
	for i := range kf.NSplits {
		end := start + foldSize
		if i < remainder {
			end++
		}

		testIndices := make([]int, end-start)
		copy(testIndices, indices[start:end])

		trainIndices := make([]int, 0, nSamples-(end-start))
		trainIndices = append(trainIndices, indices[:start]...)
		trainIndices = append(trainIndices, indices[end:]...)

		folds[i] = Fold{
			TrainIndices: trainIndices,
			TestIndices:  testIndices,
		}
		start = end
	}

	return folds, nil
}

// StratifiedKFold provides Stratified K-Fold cross-validation splitting.
//
// It ensures each fold has approximately the same class distribution as the
// overall dataset. NSplits is the number of folds (must be >= 2).
type StratifiedKFold struct {
	NSplits int
	Shuffle bool
	Seed    int64
}

// Split generates stratified cross-validation folds based on the class labels y.
// Each fold has approximately the same proportion of each class.
func (skf StratifiedKFold) Split(y []float64) ([]Fold, error) {
	if skf.NSplits < 2 {
		return nil, fmt.Errorf("glearn/modelselection: %w: NSplits must be >= 2, got %d",
			glearn.ErrInvalidParameter, skf.NSplits)
	}
	nSamples := len(y)
	if nSamples < skf.NSplits {
		return nil, fmt.Errorf("glearn/modelselection: %w: nSamples (%d) must be >= NSplits (%d)",
			glearn.ErrInvalidParameter, nSamples, skf.NSplits)
	}

	// Group indices by class.
	classIndices := make(map[float64][]int)
	for i, label := range y {
		classIndices[label] = append(classIndices[label], i)
	}

	// Sort class keys for determinism.
	classes := make([]float64, 0, len(classIndices))
	for c := range classIndices {
		classes = append(classes, c)
	}
	sort.Float64s(classes)

	// Validate that each class has at least NSplits samples.
	for _, c := range classes {
		if len(classIndices[c]) < skf.NSplits {
			return nil, fmt.Errorf("glearn/modelselection: %w: class %g has %d samples, need at least %d (NSplits)",
				glearn.ErrInvalidParameter, c, len(classIndices[c]), skf.NSplits)
		}
	}

	// Optionally shuffle within each class.
	if skf.Shuffle {
		rng := rand.New(rand.NewPCG(uint64(skf.Seed), 0))
		for _, c := range classes {
			idxs := classIndices[c]
			rng.Shuffle(len(idxs), func(i, j int) {
				idxs[i], idxs[j] = idxs[j], idxs[i]
			})
		}
	}

	// Assign each class's indices to folds proportionally.
	// testSets[fold] collects all test indices for that fold.
	testSets := make([][]int, skf.NSplits)
	for i := range skf.NSplits {
		testSets[i] = make([]int, 0)
	}

	for _, c := range classes {
		idxs := classIndices[c]
		n := len(idxs)
		foldSize := n / skf.NSplits
		remainder := n % skf.NSplits

		start := 0
		for fold := range skf.NSplits {
			end := start + foldSize
			if fold < remainder {
				end++
			}
			testSets[fold] = append(testSets[fold], idxs[start:end]...)
			start = end
		}
	}

	// Build the full set of all indices for computing train sets.
	allIndices := make(map[int]struct{}, nSamples)
	for i := range nSamples {
		allIndices[i] = struct{}{}
	}

	folds := make([]Fold, skf.NSplits)
	for i := range skf.NSplits {
		testSet := make(map[int]struct{}, len(testSets[i]))
		for _, idx := range testSets[i] {
			testSet[idx] = struct{}{}
		}

		trainIndices := make([]int, 0, nSamples-len(testSets[i]))
		for j := range nSamples {
			if _, inTest := testSet[j]; !inTest {
				trainIndices = append(trainIndices, j)
			}
		}

		// Sort test indices for deterministic output.
		sort.Ints(testSets[i])

		folds[i] = Fold{
			TrainIndices: trainIndices,
			TestIndices:  testSets[i],
		}
	}

	return folds, nil
}
