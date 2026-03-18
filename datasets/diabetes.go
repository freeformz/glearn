package datasets

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

const (
	diabetesNSamples  = 442
	diabetesNFeatures = 10
)

// LoadDiabetes returns the Diabetes dataset: 442 samples, 10 features, continuous target.
//
// The 10 features are age, sex, body mass index, average blood pressure,
// and six blood serum measurements (s1-s6). Each feature has been mean-centered
// and scaled by the number of samples times its standard deviation (as in
// scikit-learn). The target is a quantitative measure of disease progression
// one year after baseline.
func LoadDiabetes() (*mat.Dense, []float64, error) {
	if len(diabetesFeatures) != diabetesNSamples*diabetesNFeatures {
		return nil, nil, fmt.Errorf("datasets: diabetes feature data has %d elements, expected %d", len(diabetesFeatures), diabetesNSamples*diabetesNFeatures)
	}
	if len(diabetesTargets) != diabetesNSamples {
		return nil, nil, fmt.Errorf("datasets: diabetes target data has %d elements, expected %d", len(diabetesTargets), diabetesNSamples)
	}

	features := make([]float64, len(diabetesFeatures))
	copy(features, diabetesFeatures)
	targets := make([]float64, len(diabetesTargets))
	copy(targets, diabetesTargets)

	X := mat.NewDense(diabetesNSamples, diabetesNFeatures, features)
	return X, targets, nil
}
