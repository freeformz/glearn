package neighbors

import (
	"context"
	"fmt"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/internal/validate"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface checks.
var (
	_ glearn.Estimator = KNeighborsRegressorConfig{}
	_ glearn.Predictor = (*KNeighborsRegressor)(nil)
)

// KNeighborsRegressorConfig holds hyperparameters for the K-Nearest Neighbors
// regressor. It has Fit() but no Predict().
type KNeighborsRegressorConfig struct {
	// K is the number of neighbors to consider. Default is 5.
	K int
	// Weights specifies the weight function: "uniform" or "distance". Default is "uniform".
	Weights Weights
}

// NewKNeighborsRegressor creates a KNeighborsRegressorConfig with the given options.
// By default, K is 5 and weights are uniform.
func NewKNeighborsRegressor(opts ...KNeighborsRegressorOption) KNeighborsRegressorConfig {
	cfg := KNeighborsRegressorConfig{
		K:       5,
		Weights: WeightsUniform,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit stores the training data for lazy prediction and returns a fitted
// KNeighborsRegressor. The input data is copied.
func (cfg KNeighborsRegressorConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.FitInputs(X, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/neighbors: KNN regressor fit: %w", err)
	}

	// Check for context cancellation.
	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("glearn/neighbors: KNN regressor fit cancelled: %w", ctx.Err())
	default:
	}

	k := cfg.K
	if k <= 0 {
		return nil, fmt.Errorf("glearn/neighbors: KNN regressor fit: %w: K must be positive, got %d",
			glearn.ErrInvalidParameter, k)
	}
	if k > nSamples {
		return nil, fmt.Errorf("glearn/neighbors: KNN regressor fit: %w: K=%d exceeds number of samples=%d",
			glearn.ErrInvalidParameter, k, nSamples)
	}

	weights := cfg.Weights
	if weights == "" {
		weights = WeightsUniform
	}
	if weights != WeightsUniform && weights != WeightsDistance {
		return nil, fmt.Errorf("glearn/neighbors: KNN regressor fit: %w: unknown weights %q, expected \"uniform\" or \"distance\"",
			glearn.ErrInvalidParameter, weights)
	}

	// Copy training data.
	xCopy := mat.NewDense(nSamples, nFeatures, nil)
	xCopy.Copy(X)
	yCopy := make([]float64, nSamples)
	copy(yCopy, y)

	return &KNeighborsRegressor{
		X:         xCopy,
		Y:         yCopy,
		K:         k,
		Weights:   weights,
		NFeatures: nFeatures,
	}, nil
}

// KNeighborsRegressor is a fitted K-Nearest Neighbors regressor.
// It has Predict() but no Fit().
//
// KNeighborsRegressor is immutable after construction and safe for concurrent
// Predict calls.
type KNeighborsRegressor struct {
	// X is the training feature matrix.
	X *mat.Dense
	// Y is the training target values.
	Y []float64
	// K is the number of neighbors.
	K int
	// Weights is the weight function used in prediction.
	Weights Weights
	// NFeatures is the number of features seen during fitting.
	NFeatures int
}

// Predict returns the predicted target value for each row in X.
// For uniform weights, this is the mean of the K nearest neighbors' targets.
// For distance weights, this is the weighted mean using 1/distance as weights.
func (knn *KNeighborsRegressor) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, err := validate.PredictInputs(X, knn.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/neighbors: KNN regressor predict: %w", err)
	}

	nearest := findKNearest(knn.X, X, knn.K)
	preds := make([]float64, nSamples)

	for i := range nSamples {
		preds[i] = knn.weightedMean(nearest[i])
	}

	return preds, nil
}

// weightedMean computes the (weighted) mean of the neighbors' targets.
func (knn *KNeighborsRegressor) weightedMean(neighbors []neighbor) float64 {
	switch knn.Weights {
	case WeightsDistance:
		var weightedSum, totalWeight float64
		for _, n := range neighbors {
			target := knn.Y[n.index]
			if n.distance == 0 {
				// Exact match: give very high weight.
				weightedSum += target * 1e18
				totalWeight += 1e18
			} else {
				w := 1.0 / n.distance
				weightedSum += target * w
				totalWeight += w
			}
		}
		if totalWeight == 0 {
			return 0
		}
		return weightedSum / totalWeight
	default: // uniform
		var sum float64
		for _, n := range neighbors {
			sum += knn.Y[n.index]
		}
		return sum / float64(len(neighbors))
	}
}

// Score returns the R-squared score of the regressor on the given data.
func (knn *KNeighborsRegressor) Score(X *mat.Dense, y []float64) (float64, error) {
	preds, err := knn.Predict(X)
	if err != nil {
		return 0, err
	}
	return r2Score(y, preds), nil
}

// r2Score computes the R-squared (coefficient of determination) score.
func r2Score(yTrue, yPred []float64) float64 {
	n := len(yTrue)
	mean := 0.0
	for _, v := range yTrue {
		mean += v
	}
	mean /= float64(n)

	ssRes := 0.0
	ssTot := 0.0
	for i := range yTrue {
		diff := yTrue[i] - yPred[i]
		ssRes += diff * diff
		diffMean := yTrue[i] - mean
		ssTot += diffMean * diffMean
	}

	if ssTot == 0 {
		if ssRes == 0 {
			return 1.0
		}
		return 0.0
	}
	return 1.0 - ssRes/ssTot
}
