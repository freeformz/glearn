package modelselection

import (
	"fmt"
	"math/rand/v2"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

// TrainTestSplit splits X and y into training and testing sets.
//
// testSize is the fraction of samples to include in the test set (0.0, 1.0).
// seed controls the random shuffling for reproducibility.
func TrainTestSplit(X *mat.Dense, y []float64, testSize float64, seed int64) (
	XTrain, XTest *mat.Dense, yTrain, yTest []float64, err error,
) {
	if X == nil {
		return nil, nil, nil, nil, fmt.Errorf("glearn/modelselection: %w: X is nil", glearn.ErrEmptyInput)
	}

	nSamples, nFeatures := X.Dims()
	if nSamples == 0 || nFeatures == 0 {
		return nil, nil, nil, nil, fmt.Errorf("glearn/modelselection: %w: X has dimensions %dx%d",
			glearn.ErrEmptyInput, nSamples, nFeatures)
	}
	if len(y) != nSamples {
		return nil, nil, nil, nil, fmt.Errorf("glearn/modelselection: %w: X has %d samples but y has %d elements",
			glearn.ErrDimensionMismatch, nSamples, len(y))
	}
	if testSize <= 0 || testSize >= 1 {
		return nil, nil, nil, nil, fmt.Errorf("glearn/modelselection: %w: testSize must be in (0, 1), got %g",
			glearn.ErrInvalidParameter, testSize)
	}

	// Shuffle indices.
	indices := make([]int, nSamples)
	for i := range nSamples {
		indices[i] = i
	}
	rng := rand.New(rand.NewPCG(uint64(seed), 0))
	rng.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	// Split.
	nTest := int(float64(nSamples) * testSize)
	if nTest == 0 {
		nTest = 1
	}
	nTrain := nSamples - nTest

	trainIndices := indices[:nTrain]
	testIndices := indices[nTrain:]

	XTrain = extractRows(X, trainIndices, nFeatures)
	XTest = extractRows(X, testIndices, nFeatures)
	yTrain = extractElements(y, trainIndices)
	yTest = extractElements(y, testIndices)

	return XTrain, XTest, yTrain, yTest, nil
}

// extractRows builds a new Dense matrix from the specified row indices of src.
func extractRows(src *mat.Dense, indices []int, nFeatures int) *mat.Dense {
	n := len(indices)
	data := make([]float64, n*nFeatures)
	raw := src.RawMatrix()
	for i, idx := range indices {
		copy(data[i*nFeatures:(i+1)*nFeatures], raw.Data[idx*raw.Stride:idx*raw.Stride+nFeatures])
	}
	return mat.NewDense(n, nFeatures, data)
}

// extractElements builds a new slice from the specified indices of src.
func extractElements(src []float64, indices []int) []float64 {
	result := make([]float64, len(indices))
	for i, idx := range indices {
		result[i] = src[idx]
	}
	return result
}
