// Package validate provides input validation helpers for glearn algorithms.
package validate

import (
	"fmt"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

// Dimensions checks that X has at least one sample and one feature.
func Dimensions(X *mat.Dense) (rows, cols int, err error) {
	rows, cols = X.Dims()
	if rows == 0 || cols == 0 {
		return 0, 0, fmt.Errorf("%w: X has dimensions %dx%d", glearn.ErrEmptyInput, rows, cols)
	}
	return rows, cols, nil
}

// FitInputs validates X and y for supervised learning.
// Returns the number of samples and features.
func FitInputs(X *mat.Dense, y []float64) (nSamples, nFeatures int, err error) {
	nSamples, nFeatures, err = Dimensions(X)
	if err != nil {
		return 0, 0, err
	}
	if len(y) != nSamples {
		return 0, 0, fmt.Errorf("%w: X has %d samples but y has %d elements",
			glearn.ErrDimensionMismatch, nSamples, len(y))
	}
	return nSamples, nFeatures, nil
}

// PredictInputs validates X for prediction against expected feature count.
func PredictInputs(X *mat.Dense, expectedFeatures int) (nSamples int, err error) {
	nSamples, nFeatures, err := Dimensions(X)
	if err != nil {
		return 0, err
	}
	if nFeatures != expectedFeatures {
		return 0, fmt.Errorf("%w: model expects %d features but X has %d",
			glearn.ErrDimensionMismatch, expectedFeatures, nFeatures)
	}
	return nSamples, nil
}
