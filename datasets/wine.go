package datasets

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

const (
	wineNSamples  = 178
	wineNFeatures = 13
)

// LoadWine returns the Wine dataset: 178 samples, 13 features, 3 classes (0, 1, 2).
//
// Features are alcohol, malic acid, ash, alcalinity of ash, magnesium,
// total phenols, flavanoids, nonflavanoid phenols, proanthocyanins,
// color intensity, hue, OD280/OD315 of diluted wines, and proline.
// Classes correspond to three Italian wine cultivars.
func LoadWine() (*mat.Dense, []float64, error) {
	if len(wineFeatures) != wineNSamples*wineNFeatures {
		return nil, nil, fmt.Errorf("datasets: wine feature data has %d elements, expected %d", len(wineFeatures), wineNSamples*wineNFeatures)
	}
	if len(wineTargets) != wineNSamples {
		return nil, nil, fmt.Errorf("datasets: wine target data has %d elements, expected %d", len(wineTargets), wineNSamples)
	}

	features := make([]float64, len(wineFeatures))
	copy(features, wineFeatures)
	targets := make([]float64, len(wineTargets))
	copy(targets, wineTargets)

	X := mat.NewDense(wineNSamples, wineNFeatures, features)
	return X, targets, nil
}
