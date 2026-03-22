package tree

import (
	"errors"
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface verification.
var _ glearn.Estimator = DecisionTreeClassifierConfig{}

func TestClassifierXOR(t *testing.T) {
	// XOR pattern: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
	X := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})
	y := []float64{0, 1, 1, 0}

	cfg := NewDecisionTreeClassifier()
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	preds, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	for i, want := range y {
		if preds[i] != want {
			t.Errorf("sample %d: got %v, want %v", i, preds[i], want)
		}
	}
}

func TestClassifierPerfectTrainingAccuracy(t *testing.T) {
	// A dataset where a deep enough tree should achieve 100% training accuracy.
	// Simple linearly separable data.
	X := mat.NewDense(8, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
		2, 0,
		2, 1,
		3, 0,
		3, 1,
	})
	y := []float64{0, 0, 0, 1, 1, 1, 1, 1}

	cfg := NewDecisionTreeClassifier() // unlimited depth
	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	dt := predictor.(*DecisionTreeClassifier)
	score, err := dt.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}

	if score != 1.0 {
		t.Errorf("expected perfect training accuracy 1.0, got %f", score)
	}
}

func TestClassifierMaxDepth(t *testing.T) {
	// XOR cannot be solved with depth 1 (a single split).
	X := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})
	y := []float64{0, 1, 1, 0}

	cfg := NewDecisionTreeClassifier(WithClassifierMaxDepth(1))
	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	dt := predictor.(*DecisionTreeClassifier)
	score, err := dt.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}

	// With MaxDepth=1, accuracy should be less than 1 for XOR.
	if score >= 1.0 {
		t.Errorf("expected imperfect accuracy with MaxDepth=1, got %f", score)
	}

	// With unlimited depth, it should solve XOR perfectly.
	cfg2 := NewDecisionTreeClassifier()
	predictor2, err := cfg2.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	dt2 := predictor2.(*DecisionTreeClassifier)
	score2, err := dt2.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}

	if score2 != 1.0 {
		t.Errorf("expected perfect accuracy with unlimited depth, got %f", score2)
	}
}

func TestClassifierFeatureImportances(t *testing.T) {
	// Feature 0 is the relevant feature, feature 1 is noise.
	X := mat.NewDense(6, 2, []float64{
		0, 10,
		1, 20,
		2, 30,
		3, 40,
		4, 50,
		5, 60,
	})
	y := []float64{0, 0, 0, 1, 1, 1}

	cfg := NewDecisionTreeClassifier()
	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	dt := predictor.(*DecisionTreeClassifier)
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

func TestClassifierDimensionMismatch(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})
	y := []float64{0, 1, 1, 0}

	cfg := NewDecisionTreeClassifier()
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

func TestClassifierFitDimensionMismatch(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})
	// y has wrong length.
	y := []float64{0, 1, 1}

	cfg := NewDecisionTreeClassifier()
	_, err := cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for dimension mismatch, got nil")
	}
	if !errors.Is(err, glearn.ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got: %v", err)
	}
}

func TestClassifierEntropyCriterion(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	})
	y := []float64{0, 1, 1, 0}

	cfg := NewDecisionTreeClassifier(WithClassifierCriterion("entropy"))
	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit with entropy failed: %v", err)
	}

	dt := predictor.(*DecisionTreeClassifier)
	score, err := dt.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}
	if score != 1.0 {
		t.Errorf("expected perfect accuracy with entropy on XOR, got %f", score)
	}
}

func TestClassifierInvalidCriterion(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{0, 0, 0, 1, 1, 0, 1, 1})
	y := []float64{0, 1, 1, 0}

	cfg := NewDecisionTreeClassifier(WithClassifierCriterion("bad"))
	_, err := cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for invalid criterion, got nil")
	}
	if !errors.Is(err, glearn.ErrInvalidParameter) {
		t.Errorf("expected ErrInvalidParameter, got: %v", err)
	}
}

func TestClassifierPredictProbabilities(t *testing.T) {
	// Simple data: class 0 for x<0.5, class 1 for x>=0.5.
	X := mat.NewDense(4, 1, []float64{0, 0.3, 0.7, 1.0})
	y := []float64{0, 0, 1, 1}

	cfg := NewDecisionTreeClassifier()
	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	dt := predictor.(*DecisionTreeClassifier)
	proba, err := dt.PredictProbabilities(X)
	if err != nil {
		t.Fatalf("PredictProbabilities failed: %v", err)
	}

	rows, cols := proba.Dims()
	if rows != 4 || cols != 2 {
		t.Fatalf("expected shape (4, 2), got (%d, %d)", rows, cols)
	}

	// For a perfectly fitted tree, each leaf should have probability 1 for its class.
	for i := 0; i < rows; i++ {
		rowSum := 0.0
		for j := 0; j < cols; j++ {
			p := proba.At(i, j)
			if p < 0 || p > 1 {
				t.Errorf("probability out of range [0,1]: %f at (%d, %d)", p, i, j)
			}
			rowSum += p
		}
		if math.Abs(rowSum-1.0) > 1e-10 {
			t.Errorf("row %d probabilities sum to %f, want 1.0", i, rowSum)
		}
	}
}

func TestClassifierGetClasses(t *testing.T) {
	X := mat.NewDense(6, 1, []float64{0, 1, 2, 3, 4, 5})
	y := []float64{2, 0, 1, 2, 0, 1}

	cfg := NewDecisionTreeClassifier()
	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	dt := predictor.(*DecisionTreeClassifier)
	classes := dt.GetClasses()

	want := []float64{0, 1, 2}
	if len(classes) != len(want) {
		t.Fatalf("expected %d classes, got %d", len(want), len(classes))
	}
	for i, c := range classes {
		if c != want[i] {
			t.Errorf("class %d: got %v, want %v", i, c, want[i])
		}
	}
}

func TestClassifierMinSamplesLeaf(t *testing.T) {
	X := mat.NewDense(10, 1, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
	y := []float64{0, 0, 0, 0, 0, 1, 1, 1, 1, 1}

	// With MinSamplesLeaf=4, the tree should be more constrained.
	cfg := NewDecisionTreeClassifier(WithClassifierMinSamplesLeaf(4))
	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	dt := predictor.(*DecisionTreeClassifier)

	// The tree should have leaves with at least 4 samples.
	checkMinLeafSamples(t, dt.Tree, 4)
}

func checkMinLeafSamples(t *testing.T, node *TreeNode, minSamples int) {
	t.Helper()
	if node.IsLeaf() {
		if node.NSamples < minSamples {
			t.Errorf("leaf has %d samples, minimum is %d", node.NSamples, minSamples)
		}
		return
	}
	checkMinLeafSamples(t, node.Left, minSamples)
	checkMinLeafSamples(t, node.Right, minSamples)
}

func TestClassifierSingleSample(t *testing.T) {
	// Single sample: the tree should be a leaf.
	X := mat.NewDense(1, 2, []float64{1, 2})
	y := []float64{0}

	cfg := NewDecisionTreeClassifier()
	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	dt := predictor.(*DecisionTreeClassifier)
	if !dt.Tree.IsLeaf() {
		t.Error("expected a single leaf node for single sample")
	}

	preds, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	if preds[0] != 0 {
		t.Errorf("expected prediction 0, got %f", preds[0])
	}
}
