package datasets

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

const (
	irisNSamples  = 150
	irisNFeatures = 4
)

// LoadIris returns the Iris dataset: 150 samples, 4 features, 3 classes (0, 1, 2).
//
// Features are sepal length, sepal width, petal length, and petal width in cm.
// Classes are Iris-setosa (0), Iris-versicolor (1), and Iris-virginica (2).
func LoadIris() (*mat.Dense, []float64, error) {
	if len(irisFeatures) != irisNSamples*irisNFeatures {
		return nil, nil, fmt.Errorf("datasets: iris feature data has %d elements, expected %d", len(irisFeatures), irisNSamples*irisNFeatures)
	}
	if len(irisTargets) != irisNSamples {
		return nil, nil, fmt.Errorf("datasets: iris target data has %d elements, expected %d", len(irisTargets), irisNSamples)
	}

	// Copy data so callers cannot mutate the embedded slices.
	features := make([]float64, len(irisFeatures))
	copy(features, irisFeatures)
	targets := make([]float64, len(irisTargets))
	copy(targets, irisTargets)

	X := mat.NewDense(irisNSamples, irisNFeatures, features)
	return X, targets, nil
}
