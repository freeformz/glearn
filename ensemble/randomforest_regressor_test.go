package ensemble_test

import (
	"errors"
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/ensemble"
	"gonum.org/v1/gonum/mat"
)

// makeRegressionData creates a simple regression dataset: y = 2*x0 + 3*x1 + noise.
func makeRegressionData(n int) (*mat.Dense, []float64) {
	nFeatures := 5
	data := make([]float64, n*nFeatures)
	labels := make([]float64, n)

	for i := range n {
		x0 := float64(i) * 0.1
		x1 := float64(n-i) * 0.05
		x2 := float64(i%10) * 0.3
		x3 := float64(i%7) * 0.2
		x4 := float64(i%5) * 0.15
		data[i*nFeatures+0] = x0
		data[i*nFeatures+1] = x1
		data[i*nFeatures+2] = x2
		data[i*nFeatures+3] = x3
		data[i*nFeatures+4] = x4
		labels[i] = 2*x0 + 3*x1
	}

	X := mat.NewDense(n, nFeatures, data)
	return X, labels
}

func TestRandomForestRegressor_FitPredict(t *testing.T) {
	X, y := makeRegressionData(100)

	cfg := ensemble.NewRandomForestRegressor(
		ensemble.WithNTrees(50),
		ensemble.WithMaxDepth(10),
		ensemble.WithSeed(42),
		ensemble.WithNJobs(2),
	)

	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	scorer := predictor.(glearn.Scorer)
	r2, err := scorer.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}

	// R-squared should be decent on training data. With nFeatures/3 feature
	// subsampling (1 feature per tree for 5-feature data), R^2 is moderate.
	if r2 < 0.5 {
		t.Errorf("expected R^2 >= 0.5, got %f", r2)
	}
}

func TestRandomForestRegressor_MoreTreesImproves(t *testing.T) {
	X, y := makeRegressionData(100)

	cfg5 := ensemble.NewRandomForestRegressor(
		ensemble.WithNTrees(2),
		ensemble.WithMaxDepth(3),
		ensemble.WithSeed(42),
	)
	cfg50 := ensemble.NewRandomForestRegressor(
		ensemble.WithNTrees(50),
		ensemble.WithMaxDepth(3),
		ensemble.WithSeed(42),
	)

	pred5, err := cfg5.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit with 2 trees failed: %v", err)
	}
	pred50, err := cfg50.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit with 50 trees failed: %v", err)
	}

	scorer5 := pred5.(glearn.Scorer)
	scorer50 := pred50.(glearn.Scorer)

	r2_5, _ := scorer5.Score(X, y)
	r2_50, _ := scorer50.Score(X, y)

	// More trees should not significantly hurt performance.
	if r2_50 < r2_5-0.1 {
		t.Errorf("50 trees (R2=%.3f) significantly worse than 2 trees (R2=%.3f)", r2_50, r2_5)
	}
}

func TestRandomForestRegressor_FeatureImportances(t *testing.T) {
	X, y := makeRegressionData(100)

	cfg := ensemble.NewRandomForestRegressor(
		ensemble.WithNTrees(30),
		ensemble.WithSeed(42),
	)

	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	fi := predictor.(glearn.HasFeatureImportances)
	importances := fi.GetFeatureImportances()

	if len(importances) != 5 {
		t.Fatalf("expected 5 feature importances, got %d", len(importances))
	}

	sum := 0.0
	for _, v := range importances {
		sum += v
		if v < 0 {
			t.Errorf("feature importance should be non-negative, got %f", v)
		}
	}

	if math.Abs(sum-1.0) > 0.05 {
		t.Errorf("feature importances should sum to ~1.0, got %f", sum)
	}
}

func TestRandomForestRegressor_DimensionMismatch(t *testing.T) {
	X, y := makeRegressionData(50)

	cfg := ensemble.NewRandomForestRegressor(
		ensemble.WithNTrees(5),
		ensemble.WithSeed(42),
	)

	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	wrongX := mat.NewDense(10, 3, nil)
	_, err = predictor.Predict(wrongX)
	if err == nil {
		t.Fatal("expected error for dimension mismatch, got nil")
	}
	if !errors.Is(err, glearn.ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got: %v", err)
	}
}

func TestRandomForestRegressor_PredictionIsMean(t *testing.T) {
	// With a single deterministic target, all trees should predict the same mean.
	n := 20
	nFeatures := 2
	data := make([]float64, n*nFeatures)
	labels := make([]float64, n)
	for i := range n {
		data[i*nFeatures+0] = float64(i)
		data[i*nFeatures+1] = float64(i * 2)
		labels[i] = 5.0 // constant target
	}
	X := mat.NewDense(n, nFeatures, data)

	cfg := ensemble.NewRandomForestRegressor(
		ensemble.WithNTrees(10),
		ensemble.WithSeed(42),
	)

	predictor, err := cfg.Fit(t.Context(), X, labels)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	preds, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	for i, p := range preds {
		if math.Abs(p-5.0) > 0.1 {
			t.Errorf("pred[%d] = %f, expected ~5.0", i, p)
		}
	}
}
