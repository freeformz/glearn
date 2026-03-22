package ensemble

import (
	"context"
	"fmt"
	"math/rand/v2"
	"runtime"
	"sort"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/internal/validate"
	"github.com/freeformz/glearn/tree"
	"golang.org/x/sync/errgroup"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface checks.
var (
	_ glearn.Estimator            = RandomForestRegressorConfig{}
	_ glearn.Predictor            = (*RandomForestRegressor)(nil)
	_ glearn.Scorer               = (*RandomForestRegressor)(nil)
	_ glearn.HasFeatureImportances = (*RandomForestRegressor)(nil)
)

// RandomForestRegressorConfig holds hyperparameters for a random forest
// regressor. It has Fit() but no Predict().
type RandomForestRegressorConfig struct {
	opts rfOptions
}

// NewRandomForestRegressor creates a RandomForestRegressorConfig with
// sensible defaults and applies the given options.
func NewRandomForestRegressor(opts ...RFOption) RandomForestRegressorConfig {
	o := defaultRFOptions()
	for _, opt := range opts {
		opt(&o)
	}
	return RandomForestRegressorConfig{opts: o}
}

// Fit trains a random forest regressor using bootstrap aggregation (bagging).
// Each tree is trained on a bootstrap sample with a random feature subset.
func (cfg RandomForestRegressorConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.FitInputs(X, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/ensemble: random forest regressor fit failed: %w", err)
	}

	if cfg.opts.nTrees < 1 {
		return nil, fmt.Errorf("glearn/ensemble: random forest regressor fit failed: %w: NTrees must be >= 1, got %d",
			glearn.ErrInvalidParameter, cfg.opts.nTrees)
	}

	// Determine number of features to consider at each split: nFeatures/3.
	maxFeatures := nFeatures / 3
	if maxFeatures < 1 {
		maxFeatures = 1
	}

	// Determine parallelism.
	nJobs := cfg.opts.nJobs
	if nJobs <= 0 {
		nJobs = runtime.NumCPU()
	}

	nTrees := cfg.opts.nTrees

	// Create per-tree RNGs from the seed.
	baseRNG := rand.New(rand.NewPCG(uint64(cfg.opts.seed), 0))
	treeSeeds := make([]uint64, nTrees)
	for i := range nTrees {
		treeSeeds[i] = baseRNG.Uint64()
	}

	// Train trees in parallel.
	trees := make([]*tree.DecisionTreeRegressor, nTrees)
	featureSubsets := make([][]int, nTrees)

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(nJobs)

	for i := range nTrees {
		g.Go(func() error {
			rng := rand.New(rand.NewPCG(treeSeeds[i], 0))

			// Bootstrap sample: sample nSamples with replacement.
			bootIdx := make([]int, nSamples)
			for j := range nSamples {
				bootIdx[j] = rng.IntN(nSamples)
			}

			// Random feature subset.
			featureIdx := rng.Perm(nFeatures)[:maxFeatures]
			sort.Ints(featureIdx)
			featureSubsets[i] = featureIdx

			// Build bootstrap X and y with selected features.
			bootX := mat.NewDense(nSamples, maxFeatures, nil)
			bootY := make([]float64, nSamples)
			raw := X.RawMatrix()
			for j := range nSamples {
				sampleIdx := bootIdx[j]
				bootY[j] = y[sampleIdx]
				for k, fIdx := range featureIdx {
					bootX.Set(j, k, raw.Data[sampleIdx*raw.Stride+fIdx])
				}
			}

			// Fit a decision tree on the bootstrap sample.
			treeCfg := tree.NewDecisionTreeRegressor(
				tree.WithRegressorMaxDepth(cfg.opts.maxDepth),
				tree.WithRegressorMinSamplesSplit(cfg.opts.minSamplesSplit),
				tree.WithRegressorMinSamplesLeaf(cfg.opts.minSamplesLeaf),
				tree.WithRegressorSeed(int64(treeSeeds[i])),
			)

			predictor, err := treeCfg.Fit(gctx, bootX, bootY)
			if err != nil {
				return fmt.Errorf("tree %d: %w", i, err)
			}
			trees[i] = predictor.(*tree.DecisionTreeRegressor)
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, fmt.Errorf("glearn/ensemble: random forest regressor fit failed: %w", err)
	}

	// Compute average feature importances mapped back to original feature space.
	importances := make([]float64, nFeatures)
	for i := range nTrees {
		treeImp := trees[i].GetFeatureImportances()
		for j, fIdx := range featureSubsets[i] {
			importances[fIdx] += treeImp[j]
		}
	}
	// Normalize to sum to 1.
	totalImp := 0.0
	for _, v := range importances {
		totalImp += v
	}
	if totalImp > 0 {
		for i := range importances {
			importances[i] /= totalImp
		}
	}

	return &RandomForestRegressor{
		Trees:              trees,
		FeatureSubsets:     featureSubsets,
		NFeatures:          nFeatures,
		FeatureImportances: importances,
	}, nil
}

// RandomForestRegressor is a fitted random forest for regression.
// It has Predict() but no Fit().
type RandomForestRegressor struct {
	// Trees are the fitted decision tree regressors.
	Trees []*tree.DecisionTreeRegressor

	// FeatureSubsets[i] is the list of original feature indices used by Trees[i].
	FeatureSubsets [][]int

	// NFeatures is the number of features in the original training data.
	NFeatures int

	// FeatureImportances are the averaged impurity-based importances,
	// normalized to sum to 1.
	FeatureImportances []float64
}

// Predict returns the predicted value for each sample in X by averaging
// predictions from all trees.
func (rf *RandomForestRegressor) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, err := validate.PredictInputs(X, rf.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/ensemble: random forest regressor predict failed: %w", err)
	}

	raw := X.RawMatrix()
	preds := make([]float64, nSamples)
	nTrees := float64(len(rf.Trees))

	for i := range nSamples {
		sample := raw.Data[i*raw.Stride : i*raw.Stride+rf.NFeatures]
		sum := 0.0

		for t := range rf.Trees {
			subset := rf.FeatureSubsets[t]
			subSample := make([]float64, len(subset))
			for k, fIdx := range subset {
				subSample[k] = sample[fIdx]
			}
			subX := mat.NewDense(1, len(subset), subSample)

			treePreds, err := rf.Trees[t].Predict(subX)
			if err != nil {
				return nil, fmt.Errorf("glearn/ensemble: random forest regressor predict failed at tree %d: %w", t, err)
			}
			sum += treePreds[0]
		}

		preds[i] = sum / nTrees
	}

	return preds, nil
}

// Score returns the R-squared score of the model on the given data.
func (rf *RandomForestRegressor) Score(X *mat.Dense, y []float64) (float64, error) {
	preds, err := rf.Predict(X)
	if err != nil {
		return 0, err
	}

	// Compute R-squared.
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

// GetFeatureImportances returns a copy of the feature importance scores.
func (rf *RandomForestRegressor) GetFeatureImportances() []float64 {
	out := make([]float64, len(rf.FeatureImportances))
	copy(out, rf.FeatureImportances)
	return out
}
