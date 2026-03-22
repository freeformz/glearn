package tree

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
	_ glearn.Estimator            = DecisionTreeClassifierConfig{}
	_ glearn.Predictor            = (*DecisionTreeClassifier)(nil)
	_ glearn.Classifier           = (*DecisionTreeClassifier)(nil)
	_ glearn.Scorer               = (*DecisionTreeClassifier)(nil)
	_ glearn.HasFeatureImportances = (*DecisionTreeClassifier)(nil)
	_ glearn.HasClasses           = (*DecisionTreeClassifier)(nil)
)

// DecisionTreeClassifierConfig holds hyperparameters for a CART classification tree.
// It has Fit() but no Predict().
type DecisionTreeClassifierConfig struct {
	// MaxDepth is the maximum depth of the tree. -1 means unlimited.
	MaxDepth int

	// MinSamplesSplit is the minimum number of samples required to split a node.
	MinSamplesSplit int

	// MinSamplesLeaf is the minimum number of samples required at a leaf node.
	MinSamplesLeaf int

	// Criterion is the impurity measure: "gini" (default) or "entropy".
	Criterion string

	// Seed is the random seed for reproducibility.
	Seed int64
}

// NewDecisionTreeClassifier creates a DecisionTreeClassifierConfig with
// sensible defaults and applies the given options.
func NewDecisionTreeClassifier(opts ...ClassifierOption) DecisionTreeClassifierConfig {
	cfg := DecisionTreeClassifierConfig{
		MaxDepth:        -1,
		MinSamplesSplit: 2,
		MinSamplesLeaf:  1,
		Criterion:       "gini",
		Seed:            0,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit builds a decision tree classifier from the training data.
// X is the feature matrix and y contains class labels.
// Returns a fitted DecisionTreeClassifier.
func (cfg DecisionTreeClassifierConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.FitInputs(X, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/tree: classifier fit failed: %w", err)
	}

	// Validate parameters.
	if cfg.Criterion != "gini" && cfg.Criterion != "entropy" {
		return nil, fmt.Errorf("glearn/tree: classifier fit failed: %w: unsupported criterion %q (use \"gini\" or \"entropy\")",
			glearn.ErrInvalidParameter, cfg.Criterion)
	}
	if cfg.MinSamplesSplit < 2 {
		return nil, fmt.Errorf("glearn/tree: classifier fit failed: %w: MinSamplesSplit must be >= 2, got %d",
			glearn.ErrInvalidParameter, cfg.MinSamplesSplit)
	}
	if cfg.MinSamplesLeaf < 1 {
		return nil, fmt.Errorf("glearn/tree: classifier fit failed: %w: MinSamplesLeaf must be >= 1, got %d",
			glearn.ErrInvalidParameter, cfg.MinSamplesLeaf)
	}

	// Check for context cancellation.
	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("glearn/tree: classifier fit cancelled: %w", ctx.Err())
	default:
	}

	// Discover unique classes, sorted.
	classSet := make(map[float64]struct{})
	for _, v := range y {
		classSet[v] = struct{}{}
	}
	classes := make([]float64, 0, len(classSet))
	for c := range classSet {
		classes = append(classes, c)
	}
	sort.Float64s(classes)

	// Build a class-to-index map.
	classIdx := make(map[float64]int, len(classes))
	for i, c := range classes {
		classIdx[c] = i
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
	tree := cfg.buildClassifierNode(ctx, raw.Data, raw.Stride, y, indices, nFeatures, classes, classIdx, importances, 0)

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

	return &DecisionTreeClassifier{
		Tree:               tree,
		NFeatures:          nFeatures,
		Classes:            classes,
		FeatureImportances: importances,
	}, nil
}

// buildClassifierNode recursively builds a classification tree node.
func (cfg DecisionTreeClassifierConfig) buildClassifierNode(
	ctx context.Context,
	xData []float64, stride int,
	y []float64,
	indices []int,
	nFeatures int,
	classes []float64,
	classIdx map[float64]int,
	importances []float64,
	depth int,
) *TreeNode {
	n := len(indices)

	// Compute class counts and probabilities.
	counts := make(map[float64]int)
	for _, idx := range indices {
		counts[y[idx]]++
	}

	proba := make([]float64, len(classes))
	for i, c := range classes {
		proba[i] = float64(counts[c]) / float64(n)
	}

	// Find the majority class.
	majorityClass := classes[0]
	majorityCount := 0
	for _, c := range classes {
		if counts[c] > majorityCount {
			majorityCount = counts[c]
			majorityClass = c
		}
	}

	// Make a leaf if stopping criteria met.
	if cfg.shouldStop(n, depth) {
		return &TreeNode{
			Feature:  -1,
			Value:    majorityClass,
			Proba:    proba,
			NSamples: n,
		}
	}

	// Check context cancellation.
	select {
	case <-ctx.Done():
		return &TreeNode{
			Feature:  -1,
			Value:    majorityClass,
			Proba:    proba,
			NSamples: n,
		}
	default:
	}

	// Find best split.
	split := findBestClassificationSplit(xData, stride, y, indices, nFeatures, cfg.Criterion, cfg.MinSamplesLeaf)
	if split == nil {
		return &TreeNode{
			Feature:  -1,
			Value:    majorityClass,
			Proba:    proba,
			NSamples: n,
		}
	}

	// Accumulate feature importance (weighted impurity decrease).
	importances[split.feature] += split.gain * float64(n)

	// Recurse.
	left := cfg.buildClassifierNode(ctx, xData, stride, y, split.leftIdx, nFeatures, classes, classIdx, importances, depth+1)
	right := cfg.buildClassifierNode(ctx, xData, stride, y, split.rightIdx, nFeatures, classes, classIdx, importances, depth+1)

	return &TreeNode{
		Feature:   split.feature,
		Threshold: split.threshold,
		Left:      left,
		Right:     right,
		Value:     majorityClass,
		Proba:     proba,
		NSamples:  n,
	}
}

// shouldStop returns true if tree building should stop at this node.
func (cfg DecisionTreeClassifierConfig) shouldStop(nSamples, depth int) bool {
	if cfg.MaxDepth >= 0 && depth >= cfg.MaxDepth {
		return true
	}
	if nSamples < cfg.MinSamplesSplit {
		return true
	}
	return false
}

// DecisionTreeClassifier is a fitted CART classification tree.
// It has Predict(), PredictProbabilities(), and Score() but no Fit().
type DecisionTreeClassifier struct {
	// Tree is the root of the learned decision tree.
	Tree *TreeNode

	// NFeatures is the number of features seen during fitting.
	NFeatures int

	// Classes are the unique class labels, sorted.
	Classes []float64

	// FeatureImportances are the mean decrease in impurity for each feature,
	// normalized to sum to 1.
	FeatureImportances []float64
}

// Predict returns the predicted class for each sample in X.
func (dt *DecisionTreeClassifier) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, err := validate.PredictInputs(X, dt.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/tree: classifier predict failed: %w", err)
	}

	raw := X.RawMatrix()
	preds := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		sample := raw.Data[i*raw.Stride : i*raw.Stride+dt.NFeatures]
		preds[i] = dt.Tree.predict(sample)
	}
	return preds, nil
}

// PredictProbabilities returns the class probability distribution for each sample in X.
// The result matrix has shape (nSamples, nClasses) where column order matches dt.Classes.
func (dt *DecisionTreeClassifier) PredictProbabilities(X *mat.Dense) (*mat.Dense, error) {
	nSamples, err := validate.PredictInputs(X, dt.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/tree: classifier predict_proba failed: %w", err)
	}

	nClasses := len(dt.Classes)
	raw := X.RawMatrix()
	result := mat.NewDense(nSamples, nClasses, nil)

	for i := 0; i < nSamples; i++ {
		sample := raw.Data[i*raw.Stride : i*raw.Stride+dt.NFeatures]
		proba := dt.Tree.predictProba(sample)
		for j := 0; j < nClasses; j++ {
			result.Set(i, j, proba[j])
		}
	}
	return result, nil
}

// Score returns the accuracy of the classifier on the given data.
func (dt *DecisionTreeClassifier) Score(X *mat.Dense, y []float64) (float64, error) {
	preds, err := dt.Predict(X)
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

// GetFeatureImportances returns a copy of the feature importance scores.
func (dt *DecisionTreeClassifier) GetFeatureImportances() []float64 {
	out := make([]float64, len(dt.FeatureImportances))
	copy(out, dt.FeatureImportances)
	return out
}

// GetClasses returns a copy of the class labels the classifier was trained on.
func (dt *DecisionTreeClassifier) GetClasses() []float64 {
	out := make([]float64, len(dt.Classes))
	copy(out, dt.Classes)
	return out
}
