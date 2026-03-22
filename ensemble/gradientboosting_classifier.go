package ensemble

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/internal/validate"
	"github.com/freeformz/glearn/tree"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface checks.
var (
	_ glearn.Estimator  = GradientBoostingClassifierConfig{}
	_ glearn.Predictor  = (*GradientBoostingClassifier)(nil)
	_ glearn.Classifier = (*GradientBoostingClassifier)(nil)
	_ glearn.Scorer     = (*GradientBoostingClassifier)(nil)
	_ glearn.HasClasses = (*GradientBoostingClassifier)(nil)
)

// GradientBoostingClassifierConfig holds hyperparameters for gradient boosted
// classification trees with log-loss (binary classification).
// It has Fit() but no Predict().
type GradientBoostingClassifierConfig struct {
	opts gbOptions
}

// NewGradientBoostingClassifier creates a GradientBoostingClassifierConfig with
// sensible defaults and applies the given options.
func NewGradientBoostingClassifier(opts ...GBOption) GradientBoostingClassifierConfig {
	o := defaultGBOptions()
	for _, opt := range opts {
		opt(&o)
	}
	return GradientBoostingClassifierConfig{opts: o}
}

// Fit trains a gradient boosting classifier for binary classification.
// Labels in y must be 0 or 1. Trees are fit to the negative gradient of
// the binomial deviance (log-loss).
func (cfg GradientBoostingClassifierConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.FitInputs(X, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/ensemble: gradient boosting classifier fit failed: %w", err)
	}

	if cfg.opts.nTrees < 1 {
		return nil, fmt.Errorf("glearn/ensemble: gradient boosting classifier fit failed: %w: NTrees must be >= 1, got %d",
			glearn.ErrInvalidParameter, cfg.opts.nTrees)
	}
	if cfg.opts.learningRate <= 0 {
		return nil, fmt.Errorf("glearn/ensemble: gradient boosting classifier fit failed: %w: LearningRate must be > 0, got %f",
			glearn.ErrInvalidParameter, cfg.opts.learningRate)
	}
	if cfg.opts.subsample <= 0 || cfg.opts.subsample > 1 {
		return nil, fmt.Errorf("glearn/ensemble: gradient boosting classifier fit failed: %w: Subsample must be in (0, 1], got %f",
			glearn.ErrInvalidParameter, cfg.opts.subsample)
	}

	// Discover classes.
	classSet := make(map[float64]struct{})
	for _, v := range y {
		classSet[v] = struct{}{}
	}
	classes := make([]float64, 0, len(classSet))
	for c := range classSet {
		classes = append(classes, c)
	}
	sort.Float64s(classes)

	if len(classes) != 2 {
		return nil, fmt.Errorf("glearn/ensemble: gradient boosting classifier fit failed: %w: binary classification requires exactly 2 classes, got %d",
			glearn.ErrInvalidParameter, len(classes))
	}

	// Map labels to 0/1. classes[0] -> 0, classes[1] -> 1.
	yBinary := make([]float64, nSamples)
	for i, v := range y {
		if v == classes[1] {
			yBinary[i] = 1
		}
	}

	nTrees := cfg.opts.nTrees
	lr := cfg.opts.learningRate

	// Initialize with log-odds: log(p / (1-p)) where p = mean(y_binary).
	posCount := 0.0
	for _, v := range yBinary {
		posCount += v
	}
	p := posCount / float64(nSamples)
	// Clamp p to avoid log(0) or log(inf).
	if p <= 0 {
		p = 1e-10
	}
	if p >= 1 {
		p = 1 - 1e-10
	}
	initValue := math.Log(p / (1 - p))

	// Current raw predictions (log-odds space) for each sample.
	rawPreds := make([]float64, nSamples)
	for i := range nSamples {
		rawPreds[i] = initValue
	}

	rng := rand.New(rand.NewPCG(uint64(cfg.opts.seed), 0))
	trees := make([]*tree.DecisionTreeRegressor, 0, nTrees)

	for m := range nTrees {
		// Check context cancellation.
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("glearn/ensemble: gradient boosting classifier fit cancelled at iteration %d: %w", m, ctx.Err())
		default:
		}

		// Compute negative gradient of log-loss: y - sigmoid(F(x)).
		residuals := make([]float64, nSamples)
		for i := range nSamples {
			prob := sigmoid(rawPreds[i])
			residuals[i] = yBinary[i] - prob
		}

		// Subsample if configured.
		var fitX *mat.Dense
		var fitY []float64
		if cfg.opts.subsample < 1.0 {
			subN := int(float64(nSamples) * cfg.opts.subsample)
			if subN < 1 {
				subN = 1
			}
			indices := rng.Perm(nSamples)[:subN]
			raw := X.RawMatrix()
			fitX = mat.NewDense(subN, nFeatures, nil)
			fitY = make([]float64, subN)
			for j, idx := range indices {
				fitY[j] = residuals[idx]
				for k := range nFeatures {
					fitX.Set(j, k, raw.Data[idx*raw.Stride+k])
				}
			}
		} else {
			fitX = X
			fitY = residuals
		}

		// Fit a regression tree to the pseudo-residuals.
		treeCfg := tree.NewDecisionTreeRegressor(
			tree.WithRegressorMaxDepth(cfg.opts.maxDepth),
			tree.WithRegressorMinSamplesLeaf(cfg.opts.minSamplesLeaf),
		)

		predictor, err := treeCfg.Fit(ctx, fitX, fitY)
		if err != nil {
			return nil, fmt.Errorf("glearn/ensemble: gradient boosting classifier fit failed at iteration %d: %w", m, err)
		}
		fittedTree := predictor.(*tree.DecisionTreeRegressor)
		trees = append(trees, fittedTree)

		// Update raw predictions: F_m(x) = F_{m-1}(x) + lr * h_m(x).
		treePreds, err := fittedTree.Predict(X)
		if err != nil {
			return nil, fmt.Errorf("glearn/ensemble: gradient boosting classifier fit failed at iteration %d: %w", m, err)
		}
		for i := range nSamples {
			rawPreds[i] += lr * treePreds[i]
		}
	}

	return &GradientBoostingClassifier{
		Trees:        trees,
		InitValue:    initValue,
		LearningRate: lr,
		NFeatures:    nFeatures,
		Classes:      classes,
	}, nil
}

// GradientBoostingClassifier is a fitted gradient boosted binary classifier.
// It has Predict() but no Fit().
type GradientBoostingClassifier struct {
	// Trees are the fitted regression trees for each boosting iteration.
	Trees []*tree.DecisionTreeRegressor

	// InitValue is the initial log-odds prediction.
	InitValue float64

	// LearningRate is the shrinkage applied to each tree's contribution.
	LearningRate float64

	// NFeatures is the number of features seen during fitting.
	NFeatures int

	// Classes are the two class labels, sorted. classes[0] is the negative
	// class and classes[1] is the positive class.
	Classes []float64
}

// rawPredict returns the raw log-odds predictions for each sample.
func (gb *GradientBoostingClassifier) rawPredict(X *mat.Dense) ([]float64, error) {
	nSamples, err := validate.PredictInputs(X, gb.NFeatures)
	if err != nil {
		return nil, err
	}

	preds := make([]float64, nSamples)
	for i := range nSamples {
		preds[i] = gb.InitValue
	}

	for _, t := range gb.Trees {
		treePreds, err := t.Predict(X)
		if err != nil {
			return nil, err
		}
		for i := range nSamples {
			preds[i] += gb.LearningRate * treePreds[i]
		}
	}

	return preds, nil
}

// Predict returns the predicted class label for each sample in X.
func (gb *GradientBoostingClassifier) Predict(X *mat.Dense) ([]float64, error) {
	rawPreds, err := gb.rawPredict(X)
	if err != nil {
		return nil, fmt.Errorf("glearn/ensemble: gradient boosting classifier predict failed: %w", err)
	}

	preds := make([]float64, len(rawPreds))
	for i, raw := range rawPreds {
		prob := sigmoid(raw)
		if prob >= 0.5 {
			preds[i] = gb.Classes[1]
		} else {
			preds[i] = gb.Classes[0]
		}
	}

	return preds, nil
}

// PredictProbabilities returns class probability estimates for each sample.
// The result matrix has shape (nSamples, 2) with columns [P(class_0), P(class_1)].
func (gb *GradientBoostingClassifier) PredictProbabilities(X *mat.Dense) (*mat.Dense, error) {
	rawPreds, err := gb.rawPredict(X)
	if err != nil {
		return nil, fmt.Errorf("glearn/ensemble: gradient boosting classifier predict_proba failed: %w", err)
	}

	nSamples := len(rawPreds)
	result := mat.NewDense(nSamples, 2, nil)
	for i, raw := range rawPreds {
		p1 := sigmoid(raw)
		result.Set(i, 0, 1-p1)
		result.Set(i, 1, p1)
	}

	return result, nil
}

// Score returns the accuracy of the classifier on the given data.
func (gb *GradientBoostingClassifier) Score(X *mat.Dense, y []float64) (float64, error) {
	preds, err := gb.Predict(X)
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

// GetClasses returns a copy of the class labels.
func (gb *GradientBoostingClassifier) GetClasses() []float64 {
	out := make([]float64, len(gb.Classes))
	copy(out, gb.Classes)
	return out
}

// sigmoid computes the logistic function: 1 / (1 + exp(-x)).
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
