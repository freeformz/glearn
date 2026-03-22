package tree

import (
	"errors"
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface verification.
var _ glearn.Estimator = DecisionTreeRegressorConfig{}

func TestRegressorSimpleLinear(t *testing.T) {
	// y = 2*x, a simple function the tree should learn perfectly.
	X := mat.NewDense(5, 1, []float64{1, 2, 3, 4, 5})
	y := []float64{2, 4, 6, 8, 10}

	cfg := NewDecisionTreeRegressor()
	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	preds, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	// Each training sample should be predicted exactly (tree can memorize).
	for i, want := range y {
		if preds[i] != want {
			t.Errorf("sample %d: got %v, want %v", i, preds[i], want)
		}
	}
}

func TestRegressorPerfectTrainingScore(t *testing.T) {
	X := mat.NewDense(8, 2, []float64{
		0, 0,
		1, 0,
		2, 0,
		3, 0,
		0, 1,
		1, 1,
		2, 1,
		3, 1,
	})
	y := []float64{0, 1, 4, 9, 0.5, 1.5, 4.5, 9.5}

	cfg := NewDecisionTreeRegressor() // unlimited depth
	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	dt := predictor.(*DecisionTreeRegressor)
	score, err := dt.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}

	if score != 1.0 {
		t.Errorf("expected perfect training R2 of 1.0, got %f", score)
	}
}

func TestRegressorMaxDepth(t *testing.T) {
	X := mat.NewDense(8, 1, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	y := []float64{1, 4, 9, 16, 25, 36, 49, 64}

	// MaxDepth=1 means only one split; cannot perfectly fit a nonlinear function.
	cfg1 := NewDecisionTreeRegressor(WithRegressorMaxDepth(1))
	predictor1, err := cfg1.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	dt1 := predictor1.(*DecisionTreeRegressor)
	score1, err := dt1.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}

	// Unlimited depth should achieve perfect fit.
	cfg2 := NewDecisionTreeRegressor()
	predictor2, err := cfg2.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	dt2 := predictor2.(*DecisionTreeRegressor)
	score2, err := dt2.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}

	if score1 >= score2 {
		t.Errorf("expected MaxDepth=1 (R2=%f) to have lower score than unlimited (R2=%f)", score1, score2)
	}
	if score2 != 1.0 {
		t.Errorf("expected perfect fit with unlimited depth, got R2=%f", score2)
	}
}

func TestRegressorFeatureImportances(t *testing.T) {
	// Feature 0 determines target; feature 1 is noise.
	X := mat.NewDense(6, 2, []float64{
		1, 100,
		2, 200,
		3, 300,
		4, 400,
		5, 500,
		6, 600,
	})
	y := []float64{10, 20, 30, 40, 50, 60}

	cfg := NewDecisionTreeRegressor()
	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	dt := predictor.(*DecisionTreeRegressor)
	imp := dt.GetFeatureImportances()

	// Feature importances should be non-negative.
	for i, v := range imp {
		if v < 0 {
			t.Errorf("feature importance %d is negative: %f", i, v)
		}
	}

	// Feature importances should sum to approximately 1.
	sum := 0.0
	for _, v := range imp {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("feature importances sum to %f, want ~1.0", sum)
	}
}

func TestRegressorDimensionMismatch(t *testing.T) {
	X := mat.NewDense(5, 2, []float64{
		0, 0,
		1, 1,
		2, 2,
		3, 3,
		4, 4,
	})
	y := []float64{0, 1, 2, 3, 4}

	cfg := NewDecisionTreeRegressor()
	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Predict with wrong number of features.
	Xbad := mat.NewDense(2, 3, []float64{
		0, 0, 0,
		1, 1, 1,
	})
	_, err = predictor.Predict(Xbad)
	if err == nil {
		t.Fatal("expected error for dimension mismatch, got nil")
	}
	if !errors.Is(err, glearn.ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got: %v", err)
	}
}

func TestRegressorFitDimensionMismatch(t *testing.T) {
	X := mat.NewDense(5, 2, []float64{
		0, 0,
		1, 1,
		2, 2,
		3, 3,
		4, 4,
	})
	// y has wrong length.
	y := []float64{0, 1, 2}

	cfg := NewDecisionTreeRegressor()
	_, err := cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for dimension mismatch, got nil")
	}
	if !errors.Is(err, glearn.ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got: %v", err)
	}
}

func TestRegressorInvalidCriterion(t *testing.T) {
	X := mat.NewDense(4, 1, []float64{1, 2, 3, 4})
	y := []float64{1, 2, 3, 4}

	cfg := NewDecisionTreeRegressor(WithRegressorCriterion("bad"))
	_, err := cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for invalid criterion, got nil")
	}
	if !errors.Is(err, glearn.ErrInvalidParameter) {
		t.Errorf("expected ErrInvalidParameter, got: %v", err)
	}
}

func TestRegressorMinSamplesLeaf(t *testing.T) {
	X := mat.NewDense(10, 1, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	y := []float64{0, 1, 4, 9, 16, 25, 36, 49, 64, 81}

	cfg := NewDecisionTreeRegressor(WithRegressorMinSamplesLeaf(3))
	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	dt := predictor.(*DecisionTreeRegressor)
	checkMinLeafSamplesRegressor(t, dt.Tree, 3)
}

func checkMinLeafSamplesRegressor(t *testing.T, node *TreeNode, minSamples int) {
	t.Helper()
	if node.IsLeaf() {
		if node.NSamples < minSamples {
			t.Errorf("leaf has %d samples, minimum is %d", node.NSamples, minSamples)
		}
		return
	}
	checkMinLeafSamplesRegressor(t, node.Left, minSamples)
	checkMinLeafSamplesRegressor(t, node.Right, minSamples)
}

func TestRegressorSingleSample(t *testing.T) {
	// Single sample: the tree should be a leaf.
	X := mat.NewDense(1, 2, []float64{1, 2})
	y := []float64{42}

	cfg := NewDecisionTreeRegressor()
	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	dt := predictor.(*DecisionTreeRegressor)
	if !dt.Tree.IsLeaf() {
		t.Error("expected a single leaf node for single sample")
	}

	preds, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	if preds[0] != 42 {
		t.Errorf("expected prediction 42, got %f", preds[0])
	}
}

func TestRegressorConstantTarget(t *testing.T) {
	// When all targets are the same, the tree should be a single leaf.
	X := mat.NewDense(5, 2, []float64{
		0, 0,
		1, 1,
		2, 2,
		3, 3,
		4, 4,
	})
	y := []float64{42, 42, 42, 42, 42}

	cfg := NewDecisionTreeRegressor()
	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	dt := predictor.(*DecisionTreeRegressor)
	if !dt.Tree.IsLeaf() {
		t.Error("expected a single leaf node for constant target")
	}
	if dt.Tree.Value != 42 {
		t.Errorf("expected leaf value 42, got %f", dt.Tree.Value)
	}
}
