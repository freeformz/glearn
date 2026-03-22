package neighbors

import (
	"context"
	"fmt"
	"sort"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/internal/validate"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface checks.
var (
	_ glearn.Estimator = KNeighborsClassifierConfig{}
	_ glearn.Predictor = (*KNeighborsClassifier)(nil)
)

// KNeighborsClassifierConfig holds hyperparameters for the K-Nearest Neighbors
// classifier. It has Fit() but no Predict().
type KNeighborsClassifierConfig struct {
	// K is the number of neighbors to consider. Default is 5.
	K int
	// Weights specifies the weight function: "uniform" or "distance". Default is "uniform".
	Weights Weights
}

// NewKNeighborsClassifier creates a KNeighborsClassifierConfig with the given options.
// By default, K is 5 and weights are uniform.
func NewKNeighborsClassifier(opts ...KNeighborsClassifierOption) KNeighborsClassifierConfig {
	cfg := KNeighborsClassifierConfig{
		K:       5,
		Weights: WeightsUniform,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit stores the training data for lazy prediction and returns a fitted
// KNeighborsClassifier. The input data is copied.
func (cfg KNeighborsClassifierConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.FitInputs(X, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/neighbors: KNN classifier fit: %w", err)
	}

	// Check for context cancellation.
	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("glearn/neighbors: KNN classifier fit cancelled: %w", ctx.Err())
	default:
	}

	k := cfg.K
	if k <= 0 {
		return nil, fmt.Errorf("glearn/neighbors: KNN classifier fit: %w: K must be positive, got %d",
			glearn.ErrInvalidParameter, k)
	}
	if k > nSamples {
		return nil, fmt.Errorf("glearn/neighbors: KNN classifier fit: %w: K=%d exceeds number of samples=%d",
			glearn.ErrInvalidParameter, k, nSamples)
	}

	weights := cfg.Weights
	if weights == "" {
		weights = WeightsUniform
	}
	if weights != WeightsUniform && weights != WeightsDistance {
		return nil, fmt.Errorf("glearn/neighbors: KNN classifier fit: %w: unknown weights %q, expected \"uniform\" or \"distance\"",
			glearn.ErrInvalidParameter, weights)
	}

	// Copy training data.
	xCopy := mat.NewDense(nSamples, nFeatures, nil)
	xCopy.Copy(X)
	yCopy := make([]float64, nSamples)
	copy(yCopy, y)

	// Extract unique classes.
	classSet := make(map[float64]struct{})
	for _, v := range yCopy {
		classSet[v] = struct{}{}
	}
	classes := make([]float64, 0, len(classSet))
	for c := range classSet {
		classes = append(classes, c)
	}
	sort.Float64s(classes)

	return &KNeighborsClassifier{
		X:         xCopy,
		Y:         yCopy,
		K:         k,
		Weights:   weights,
		NFeatures: nFeatures,
		Classes:   classes,
	}, nil
}

// KNeighborsClassifier is a fitted K-Nearest Neighbors classifier.
// It has Predict() but no Fit().
//
// KNeighborsClassifier is immutable after construction and safe for concurrent
// Predict calls.
type KNeighborsClassifier struct {
	// X is the training feature matrix.
	X *mat.Dense
	// Y is the training labels.
	Y []float64
	// K is the number of neighbors.
	K int
	// Weights is the weight function used in prediction.
	Weights Weights
	// NFeatures is the number of features seen during fitting.
	NFeatures int
	// Classes are the unique class labels from the training data, sorted.
	Classes []float64
}

// Predict returns the predicted class label for each row in X.
func (knn *KNeighborsClassifier) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, err := validate.PredictInputs(X, knn.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/neighbors: KNN classifier predict: %w", err)
	}

	nearest := findKNearest(knn.X, X, knn.K)
	preds := make([]float64, nSamples)

	for i := range nSamples {
		preds[i] = knn.majorityVote(nearest[i])
	}

	return preds, nil
}

// majorityVote returns the class with the highest (weighted) vote among neighbors.
func (knn *KNeighborsClassifier) majorityVote(neighbors []neighbor) float64 {
	votes := make(map[float64]float64)

	for _, n := range neighbors {
		label := knn.Y[n.index]
		switch knn.Weights {
		case WeightsDistance:
			if n.distance == 0 {
				// Exact match: give infinite weight (return immediately if all zero-distance
				// neighbors agree, otherwise this will dominate).
				votes[label] += 1e18
			} else {
				votes[label] += 1.0 / n.distance
			}
		default: // uniform
			votes[label] += 1.0
		}
	}

	// Find the class with the most votes.
	bestLabel := 0.0
	bestVote := -1.0
	for label, vote := range votes {
		if vote > bestVote || (vote == bestVote && label < bestLabel) {
			bestVote = vote
			bestLabel = label
		}
	}
	return bestLabel
}

// Score returns the accuracy of the classifier on the given data.
func (knn *KNeighborsClassifier) Score(X *mat.Dense, y []float64) (float64, error) {
	preds, err := knn.Predict(X)
	if err != nil {
		return 0, err
	}
	correct := 0
	for i := range y {
		if preds[i] == y[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(y)), nil
}

// GetClasses returns the unique class labels from the training data.
func (knn *KNeighborsClassifier) GetClasses() []float64 {
	out := make([]float64, len(knn.Classes))
	copy(out, knn.Classes)
	return out
}
