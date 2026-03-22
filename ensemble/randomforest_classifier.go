package ensemble

import (
	"context"
	"fmt"
	"math"
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
	_ glearn.Estimator            = RandomForestClassifierConfig{}
	_ glearn.Predictor            = (*RandomForestClassifier)(nil)
	_ glearn.Classifier           = (*RandomForestClassifier)(nil)
	_ glearn.Scorer               = (*RandomForestClassifier)(nil)
	_ glearn.HasFeatureImportances = (*RandomForestClassifier)(nil)
	_ glearn.HasClasses           = (*RandomForestClassifier)(nil)
)

// RandomForestClassifierConfig holds hyperparameters for a random forest
// classifier. It has Fit() but no Predict().
type RandomForestClassifierConfig struct {
	opts rfOptions
}

// NewRandomForestClassifier creates a RandomForestClassifierConfig with
// sensible defaults and applies the given options.
func NewRandomForestClassifier(opts ...RFOption) RandomForestClassifierConfig {
	o := defaultRFOptions()
	for _, opt := range opts {
		opt(&o)
	}
	return RandomForestClassifierConfig{opts: o}
}

// Fit trains a random forest classifier using bootstrap aggregation (bagging).
// Each tree is trained on a bootstrap sample with a random feature subset.
func (cfg RandomForestClassifierConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.FitInputs(X, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/ensemble: random forest classifier fit failed: %w", err)
	}

	if cfg.opts.nTrees < 1 {
		return nil, fmt.Errorf("glearn/ensemble: random forest classifier fit failed: %w: NTrees must be >= 1, got %d",
			glearn.ErrInvalidParameter, cfg.opts.nTrees)
	}

	// Determine number of features to consider at each split.
	maxFeatures := int(math.Sqrt(float64(nFeatures)))
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
	trees := make([]*tree.DecisionTreeClassifier, nTrees)
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
			treeCfg := tree.NewDecisionTreeClassifier(
				tree.WithClassifierMaxDepth(cfg.opts.maxDepth),
				tree.WithClassifierMinSamplesSplit(cfg.opts.minSamplesSplit),
				tree.WithClassifierMinSamplesLeaf(cfg.opts.minSamplesLeaf),
				tree.WithClassifierSeed(int64(treeSeeds[i])),
			)

			predictor, err := treeCfg.Fit(gctx, bootX, bootY)
			if err != nil {
				return fmt.Errorf("tree %d: %w", i, err)
			}
			trees[i] = predictor.(*tree.DecisionTreeClassifier)
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, fmt.Errorf("glearn/ensemble: random forest classifier fit failed: %w", err)
	}

	// Discover classes from y.
	classSet := make(map[float64]struct{})
	for _, v := range y {
		classSet[v] = struct{}{}
	}
	classes := make([]float64, 0, len(classSet))
	for c := range classSet {
		classes = append(classes, c)
	}
	sort.Float64s(classes)

	// Compute average feature importances across trees, mapped back to
	// the original feature space.
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

	return &RandomForestClassifier{
		Trees:              trees,
		FeatureSubsets:     featureSubsets,
		NFeatures:          nFeatures,
		Classes:            classes,
		FeatureImportances: importances,
	}, nil
}

// RandomForestClassifier is a fitted random forest for classification.
// It has Predict() but no Fit().
type RandomForestClassifier struct {
	// Trees are the fitted decision tree classifiers.
	Trees []*tree.DecisionTreeClassifier

	// FeatureSubsets[i] is the list of original feature indices used by Trees[i].
	FeatureSubsets [][]int

	// NFeatures is the number of features in the original training data.
	NFeatures int

	// Classes are the unique class labels, sorted.
	Classes []float64

	// FeatureImportances are the averaged impurity-based importances,
	// normalized to sum to 1.
	FeatureImportances []float64
}

// Predict returns the predicted class for each sample in X by majority vote.
func (rf *RandomForestClassifier) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, err := validate.PredictInputs(X, rf.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/ensemble: random forest classifier predict failed: %w", err)
	}

	raw := X.RawMatrix()
	preds := make([]float64, nSamples)

	// For each sample, collect votes from all trees.
	for i := range nSamples {
		votes := make(map[float64]int)
		sample := raw.Data[i*raw.Stride : i*raw.Stride+rf.NFeatures]

		for t := range rf.Trees {
			// Extract the feature subset for this tree.
			subset := rf.FeatureSubsets[t]
			subSample := make([]float64, len(subset))
			for k, fIdx := range subset {
				subSample[k] = sample[fIdx]
			}
			subX := mat.NewDense(1, len(subset), subSample)

			treePreds, err := rf.Trees[t].Predict(subX)
			if err != nil {
				return nil, fmt.Errorf("glearn/ensemble: random forest classifier predict failed at tree %d: %w", t, err)
			}
			votes[treePreds[0]]++
		}

		// Majority vote.
		bestClass := rf.Classes[0]
		bestCount := 0
		for cls, count := range votes {
			if count > bestCount {
				bestCount = count
				bestClass = cls
			}
		}
		preds[i] = bestClass
	}

	return preds, nil
}

// PredictProbabilities returns class probability estimates for each sample.
// Probabilities are computed by averaging the predicted probabilities from
// each tree.
func (rf *RandomForestClassifier) PredictProbabilities(X *mat.Dense) (*mat.Dense, error) {
	nSamples, err := validate.PredictInputs(X, rf.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/ensemble: random forest classifier predict_proba failed: %w", err)
	}

	nClasses := len(rf.Classes)
	// Map from class label to index in rf.Classes.
	classIdx := make(map[float64]int, nClasses)
	for i, c := range rf.Classes {
		classIdx[c] = i
	}

	raw := X.RawMatrix()
	result := mat.NewDense(nSamples, nClasses, nil)

	// Use a mutex to avoid creating too many temp matrices; iterate sequentially.
	for i := range nSamples {
		sample := raw.Data[i*raw.Stride : i*raw.Stride+rf.NFeatures]
		avgProba := make([]float64, nClasses)

		for t := range rf.Trees {
			subset := rf.FeatureSubsets[t]
			subSample := make([]float64, len(subset))
			for k, fIdx := range subset {
				subSample[k] = sample[fIdx]
			}
			subX := mat.NewDense(1, len(subset), subSample)

			treeProba, err := rf.Trees[t].PredictProbabilities(subX)
			if err != nil {
				return nil, fmt.Errorf("glearn/ensemble: random forest classifier predict_proba failed at tree %d: %w", t, err)
			}

			// Map tree classes to RF classes.
			treeClasses := rf.Trees[t].GetClasses()
			for j, tc := range treeClasses {
				if idx, ok := classIdx[tc]; ok {
					avgProba[idx] += treeProba.At(0, j)
				}
			}
		}

		// Average over trees.
		nTrees := float64(len(rf.Trees))
		for j := range nClasses {
			result.Set(i, j, avgProba[j]/nTrees)
		}
	}

	return result, nil
}

// Score returns the accuracy of the classifier on the given data.
func (rf *RandomForestClassifier) Score(X *mat.Dense, y []float64) (float64, error) {
	preds, err := rf.Predict(X)
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
func (rf *RandomForestClassifier) GetFeatureImportances() []float64 {
	out := make([]float64, len(rf.FeatureImportances))
	copy(out, rf.FeatureImportances)
	return out
}

// GetClasses returns a copy of the class labels.
func (rf *RandomForestClassifier) GetClasses() []float64 {
	out := make([]float64, len(rf.Classes))
	copy(out, rf.Classes)
	return out
}
