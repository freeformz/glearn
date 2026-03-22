package tree

import (
	"context"
	"fmt"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/internal/validate"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface checks.
var (
	_ glearn.Estimator            = DecisionTreeRegressorConfig{}
	_ glearn.Predictor            = (*DecisionTreeRegressor)(nil)
	_ glearn.Scorer               = (*DecisionTreeRegressor)(nil)
	_ glearn.HasFeatureImportances = (*DecisionTreeRegressor)(nil)
)

// DecisionTreeRegressorConfig holds hyperparameters for a CART regression tree.
// It has Fit() but no Predict().
type DecisionTreeRegressorConfig struct {
	// MaxDepth is the maximum depth of the tree. -1 means unlimited.
	MaxDepth int

	// MinSamplesSplit is the minimum number of samples required to split a node.
	MinSamplesSplit int

	// MinSamplesLeaf is the minimum number of samples required at a leaf node.
	MinSamplesLeaf int

	// Criterion is the impurity measure: "mse" (default).
	Criterion string

	// Seed is the random seed for reproducibility.
	Seed int64
}

// NewDecisionTreeRegressor creates a DecisionTreeRegressorConfig with
// sensible defaults and applies the given options.
func NewDecisionTreeRegressor(opts ...RegressorOption) DecisionTreeRegressorConfig {
	cfg := DecisionTreeRegressorConfig{
		MaxDepth:        -1,
		MinSamplesSplit: 2,
		MinSamplesLeaf:  1,
		Criterion:       "mse",
		Seed:            0,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit builds a decision tree regressor from the training data.
// X is the feature matrix and y contains target values.
// Returns a fitted DecisionTreeRegressor.
func (cfg DecisionTreeRegressorConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.FitInputs(X, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/tree: regressor fit failed: %w", err)
	}

	// Validate parameters.
	if cfg.Criterion != "mse" {
		return nil, fmt.Errorf("glearn/tree: regressor fit failed: %w: unsupported criterion %q (use \"mse\")",
			glearn.ErrInvalidParameter, cfg.Criterion)
	}
	if cfg.MinSamplesSplit < 2 {
		return nil, fmt.Errorf("glearn/tree: regressor fit failed: %w: MinSamplesSplit must be >= 2, got %d",
			glearn.ErrInvalidParameter, cfg.MinSamplesSplit)
	}
	if cfg.MinSamplesLeaf < 1 {
		return nil, fmt.Errorf("glearn/tree: regressor fit failed: %w: MinSamplesLeaf must be >= 1, got %d",
			glearn.ErrInvalidParameter, cfg.MinSamplesLeaf)
	}

	// Check for context cancellation.
	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("glearn/tree: regressor fit cancelled: %w", ctx.Err())
	default:
	}

	// Initialize sample indices.
	indices := make([]int, nSamples)
	for i := range indices {
		indices[i] = i
	}

	raw := X.RawMatrix()

	// Feature importances accumulator.
	importances := make([]float64, nFeatures)

	// Recursively build the tree.
	tree := cfg.buildRegressorNode(ctx, raw.Data, raw.Stride, y, indices, nFeatures, importances, 0)

	// Normalize feature importances to sum to 1.
	totalImp := 0.0
	for _, v := range importances {
		totalImp += v
	}
	if totalImp > 0 {
		for i := range importances {
			importances[i] /= totalImp
		}
	}

	return &DecisionTreeRegressor{
		Tree:               tree,
		NFeatures:          nFeatures,
		FeatureImportances: importances,
	}, nil
}

// buildRegressorNode recursively builds a regression tree node.
func (cfg DecisionTreeRegressorConfig) buildRegressorNode(
	ctx context.Context,
	xData []float64, stride int,
	y []float64,
	indices []int,
	nFeatures int,
	importances []float64,
	depth int,
) *TreeNode {
	n := len(indices)

	// Compute the mean target value for this node.
	mean := 0.0
	for _, idx := range indices {
		mean += y[idx]
	}
	mean /= float64(n)

	// Make a leaf if stopping criteria met.
	if cfg.shouldStop(n, depth) {
		return &TreeNode{
			Feature:  -1,
			Value:    mean,
			NSamples: n,
		}
	}

	// Check context cancellation.
	select {
	case <-ctx.Done():
		return &TreeNode{
			Feature:  -1,
			Value:    mean,
			NSamples: n,
		}
	default:
	}

	// Find best split.
	split := findBestRegressionSplit(xData, stride, y, indices, nFeatures, cfg.MinSamplesLeaf)
	if split == nil {
		return &TreeNode{
			Feature:  -1,
			Value:    mean,
			NSamples: n,
		}
	}

	// Accumulate feature importance (weighted impurity decrease).
	importances[split.feature] += split.gain * float64(n)

	// Recurse.
	left := cfg.buildRegressorNode(ctx, xData, stride, y, split.leftIdx, nFeatures, importances, depth+1)
	right := cfg.buildRegressorNode(ctx, xData, stride, y, split.rightIdx, nFeatures, importances, depth+1)

	return &TreeNode{
		Feature:   split.feature,
		Threshold: split.threshold,
		Left:      left,
		Right:     right,
		Value:     mean,
		NSamples:  n,
	}
}

// shouldStop returns true if tree building should stop at this node.
func (cfg DecisionTreeRegressorConfig) shouldStop(nSamples, depth int) bool {
	if cfg.MaxDepth >= 0 && depth >= cfg.MaxDepth {
		return true
	}
	if nSamples < cfg.MinSamplesSplit {
		return true
	}
	return false
}

// DecisionTreeRegressor is a fitted CART regression tree.
// It has Predict() and Score() but no Fit().
type DecisionTreeRegressor struct {
	// Tree is the root of the learned decision tree.
	Tree *TreeNode

	// NFeatures is the number of features seen during fitting.
	NFeatures int

	// FeatureImportances are the mean decrease in impurity for each feature,
	// normalized to sum to 1.
	FeatureImportances []float64
}

// Predict returns the predicted target value for each sample in X.
func (dt *DecisionTreeRegressor) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, err := validate.PredictInputs(X, dt.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/tree: regressor predict failed: %w", err)
	}

	raw := X.RawMatrix()
	preds := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		sample := raw.Data[i*raw.Stride : i*raw.Stride+dt.NFeatures]
		preds[i] = dt.Tree.predict(sample)
	}
	return preds, nil
}

// Score returns the R-squared score of the model on the given data.
func (dt *DecisionTreeRegressor) Score(X *mat.Dense, y []float64) (float64, error) {
	preds, err := dt.Predict(X)
	if err != nil {
		return 0, err
	}
	return r2Score(y, preds), nil
}

// GetFeatureImportances returns a copy of the feature importance scores.
func (dt *DecisionTreeRegressor) GetFeatureImportances() []float64 {
	out := make([]float64, len(dt.FeatureImportances))
	copy(out, dt.FeatureImportances)
	return out
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
