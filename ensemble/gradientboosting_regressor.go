package ensemble

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/internal/validate"
	"github.com/freeformz/glearn/tree"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface checks.
var (
	_ glearn.Estimator = GradientBoostingRegressorConfig{}
	_ glearn.Predictor = (*GradientBoostingRegressor)(nil)
	_ glearn.Scorer    = (*GradientBoostingRegressor)(nil)
)

// GradientBoostingRegressorConfig holds hyperparameters for gradient boosted
// regression trees with MSE loss. It has Fit() but no Predict().
type GradientBoostingRegressorConfig struct {
	opts gbOptions
}

// NewGradientBoostingRegressor creates a GradientBoostingRegressorConfig with
// sensible defaults and applies the given options.
func NewGradientBoostingRegressor(opts ...GBOption) GradientBoostingRegressorConfig {
	o := defaultGBOptions()
	for _, opt := range opts {
		opt(&o)
	}
	return GradientBoostingRegressorConfig{opts: o}
}

// Fit trains a gradient boosting regressor by sequentially fitting trees to
// the negative gradient (residuals) of the MSE loss.
func (cfg GradientBoostingRegressorConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.FitInputs(X, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/ensemble: gradient boosting regressor fit failed: %w", err)
	}

	if cfg.opts.nTrees < 1 {
		return nil, fmt.Errorf("glearn/ensemble: gradient boosting regressor fit failed: %w: NTrees must be >= 1, got %d",
			glearn.ErrInvalidParameter, cfg.opts.nTrees)
	}
	if cfg.opts.learningRate <= 0 {
		return nil, fmt.Errorf("glearn/ensemble: gradient boosting regressor fit failed: %w: LearningRate must be > 0, got %f",
			glearn.ErrInvalidParameter, cfg.opts.learningRate)
	}
	if cfg.opts.subsample <= 0 || cfg.opts.subsample > 1 {
		return nil, fmt.Errorf("glearn/ensemble: gradient boosting regressor fit failed: %w: Subsample must be in (0, 1], got %f",
			glearn.ErrInvalidParameter, cfg.opts.subsample)
	}

	nTrees := cfg.opts.nTrees
	lr := cfg.opts.learningRate

	// Initialize predictions with the mean of y.
	initValue := 0.0
	for _, v := range y {
		initValue += v
	}
	initValue /= float64(nSamples)

	// Current predictions for each sample (F_m(x_i)).
	currentPreds := make([]float64, nSamples)
	for i := range nSamples {
		currentPreds[i] = initValue
	}

	rng := rand.New(rand.NewPCG(uint64(cfg.opts.seed), 0))
	trees := make([]*tree.DecisionTreeRegressor, 0, nTrees)

	for m := range nTrees {
		// Check context cancellation.
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("glearn/ensemble: gradient boosting regressor fit cancelled at iteration %d: %w", m, ctx.Err())
		default:
		}

		// Compute negative gradient (residuals for MSE loss): y - F(x).
		residuals := make([]float64, nSamples)
		for i := range nSamples {
			residuals[i] = y[i] - currentPreds[i]
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

		// Fit a regression tree to the residuals.
		treeCfg := tree.NewDecisionTreeRegressor(
			tree.WithRegressorMaxDepth(cfg.opts.maxDepth),
			tree.WithRegressorMinSamplesLeaf(cfg.opts.minSamplesLeaf),
		)

		predictor, err := treeCfg.Fit(ctx, fitX, fitY)
		if err != nil {
			return nil, fmt.Errorf("glearn/ensemble: gradient boosting regressor fit failed at iteration %d: %w", m, err)
		}
		fittedTree := predictor.(*tree.DecisionTreeRegressor)
		trees = append(trees, fittedTree)

		// Update predictions: F_m(x) = F_{m-1}(x) + lr * h_m(x).
		treePreds, err := fittedTree.Predict(X)
		if err != nil {
			return nil, fmt.Errorf("glearn/ensemble: gradient boosting regressor fit failed at iteration %d: %w", m, err)
		}
		for i := range nSamples {
			currentPreds[i] += lr * treePreds[i]
		}
	}

	return &GradientBoostingRegressor{
		Trees:        trees,
		InitValue:    initValue,
		LearningRate: lr,
		NFeatures:    nFeatures,
	}, nil
}

// GradientBoostingRegressor is a fitted gradient boosted regression model.
// It has Predict() but no Fit().
type GradientBoostingRegressor struct {
	// Trees are the fitted regression trees for each boosting iteration.
	Trees []*tree.DecisionTreeRegressor

	// InitValue is the initial constant prediction (mean of training targets).
	InitValue float64

	// LearningRate is the shrinkage applied to each tree's contribution.
	LearningRate float64

	// NFeatures is the number of features seen during fitting.
	NFeatures int
}

// Predict returns the predicted target value for each sample in X.
func (gb *GradientBoostingRegressor) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, err := validate.PredictInputs(X, gb.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/ensemble: gradient boosting regressor predict failed: %w", err)
	}

	preds := make([]float64, nSamples)
	for i := range nSamples {
		preds[i] = gb.InitValue
	}

	for _, t := range gb.Trees {
		treePreds, err := t.Predict(X)
		if err != nil {
			return nil, fmt.Errorf("glearn/ensemble: gradient boosting regressor predict failed: %w", err)
		}
		for i := range nSamples {
			preds[i] += gb.LearningRate * treePreds[i]
		}
	}

	return preds, nil
}

// Score returns the R-squared score of the model on the given data.
func (gb *GradientBoostingRegressor) Score(X *mat.Dense, y []float64) (float64, error) {
	preds, err := gb.Predict(X)
	if err != nil {
		return 0, err
	}

	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= float64(len(y))

	ssRes := 0.0
	ssTot := 0.0
	for i := range y {
		diff := y[i] - preds[i]
		ssRes += diff * diff
		diffMean := y[i] - mean
		ssTot += diffMean * diffMean
	}

	if ssTot == 0 {
		if ssRes == 0 {
			return 1.0, nil
		}
		return 0.0, nil
	}
	return 1.0 - ssRes/ssTot, nil
}
