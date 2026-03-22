package linear

import (
	"errors"
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

func TestLogisticRegressionCompileTimeChecks(t *testing.T) {
	var _ glearn.Estimator = LogisticRegressionConfig{}
	var _ glearn.Predictor = (*LogisticRegression)(nil)
	var _ glearn.Classifier = (*LogisticRegression)(nil)
	var _ glearn.Scorer = (*LogisticRegression)(nil)
}

func TestLogisticRegressionBinary(t *testing.T) {
	// Linearly separable binary classification.
	// Class 0: x1 < 0, Class 1: x1 > 0.
	X := mat.NewDense(10, 2, []float64{
		-3, 0,
		-2, 1,
		-1, -1,
		-2, -1,
		-1, 0,
		1, 0,
		2, 1,
		3, -1,
		2, -1,
		1, 1,
	})
	y := []float64{0, 0, 0, 0, 0, 1, 1, 1, 1, 1}

	cfg := NewLogisticRegression(WithLogisticMaxIter(200))
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	lr := model.(*LogisticRegression)

	// Should get perfect or near-perfect accuracy on training data.
	preds, err := lr.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	correct := 0
	for i, p := range preds {
		if p == y[i] {
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(y))
	if accuracy < 0.9 {
		t.Errorf("expected accuracy >= 0.9, got %g", accuracy)
	}

	// Score should match accuracy.
	score, err := lr.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}
	if math.Abs(score-accuracy) > 1e-10 {
		t.Errorf("Score (%g) does not match computed accuracy (%g)", score, accuracy)
	}

	// Check that we have two classes.
	if len(lr.Classes) != 2 {
		t.Errorf("expected 2 classes, got %d", len(lr.Classes))
	}
	if lr.Classes[0] != 0 || lr.Classes[1] != 1 {
		t.Errorf("expected classes [0, 1], got %v", lr.Classes)
	}
}

func TestLogisticRegressionProbabilities(t *testing.T) {
	X := mat.NewDense(6, 1, []float64{
		-3,
		-2,
		-1,
		1,
		2,
		3,
	})
	y := []float64{0, 0, 0, 1, 1, 1}

	cfg := NewLogisticRegression(WithLogisticMaxIter(200))
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	lr := model.(*LogisticRegression)
	probs, err := lr.PredictProbabilities(X)
	if err != nil {
		t.Fatalf("PredictProbabilities failed: %v", err)
	}

	nSamples, nClasses := probs.Dims()
	if nSamples != 6 || nClasses != 2 {
		t.Fatalf("expected probs shape (6, 2), got (%d, %d)", nSamples, nClasses)
	}

	// Probabilities should sum to 1 for each sample.
	for i := range nSamples {
		sum := probs.At(i, 0) + probs.At(i, 1)
		if math.Abs(sum-1.0) > 1e-6 {
			t.Errorf("sample %d: probabilities sum to %g, expected 1.0", i, sum)
		}
	}

	// Negative x values should have higher class-0 probability.
	for i := range 3 {
		if probs.At(i, 0) < probs.At(i, 1) {
			t.Errorf("sample %d (x=%g): expected class 0 prob > class 1 prob, got %g vs %g",
				i, X.At(i, 0), probs.At(i, 0), probs.At(i, 1))
		}
	}
}

func TestLogisticRegressionMulticlass(t *testing.T) {
	// 3-class problem: well-separated clusters.
	X := mat.NewDense(12, 2, []float64{
		0, 10, // class 0
		1, 10,
		0, 11,
		1, 11,
		10, 0, // class 1
		10, 1,
		11, 0,
		11, 1,
		10, 10, // class 2
		10, 11,
		11, 10,
		11, 11,
	})
	y := []float64{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}

	cfg := NewLogisticRegression(WithLogisticMaxIter(500), WithLogisticC(10.0))
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	lr := model.(*LogisticRegression)

	if len(lr.Classes) != 3 {
		t.Errorf("expected 3 classes, got %d", len(lr.Classes))
	}

	// Should get high accuracy on well-separated data.
	score, err := lr.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}
	if score < 0.9 {
		t.Errorf("expected accuracy >= 0.9 on well-separated data, got %g", score)
	}

	// Check PredictProbabilities shape.
	probs, err := lr.PredictProbabilities(X)
	if err != nil {
		t.Fatalf("PredictProbabilities failed: %v", err)
	}
	nSamples, nClasses := probs.Dims()
	if nSamples != 12 || nClasses != 3 {
		t.Fatalf("expected probs shape (12, 3), got (%d, %d)", nSamples, nClasses)
	}

	// Probabilities should sum to ~1 for each sample.
	for i := range nSamples {
		sum := 0.0
		for c := range nClasses {
			sum += probs.At(i, c)
		}
		if math.Abs(sum-1.0) > 1e-6 {
			t.Errorf("sample %d: probabilities sum to %g, expected 1.0", i, sum)
		}
	}
}

func TestLogisticRegressionNoIntercept(t *testing.T) {
	X := mat.NewDense(6, 1, []float64{-3, -2, -1, 1, 2, 3})
	y := []float64{0, 0, 0, 1, 1, 1}

	cfg := NewLogisticRegression(WithLogisticFitIntercept(false), WithLogisticMaxIter(200))
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	lr := model.(*LogisticRegression)
	// With no intercept and symmetric data, intercept should be zero.
	if math.Abs(lr.Intercept[0]) > 1e-10 {
		t.Errorf("expected intercept ~0.0, got %g", lr.Intercept[0])
	}

	// Should still classify correctly.
	score, err := lr.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}
	if score < 0.9 {
		t.Errorf("expected accuracy >= 0.9, got %g", score)
	}
}

func TestLogisticRegressionDimensionMismatch(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{-1, 0, -2, 0, 1, 0, 2, 0})
	y := []float64{0, 0, 1, 1}

	cfg := NewLogisticRegression()
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	XBad := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})
	_, err = model.Predict(XBad)
	if err == nil {
		t.Fatal("expected error for dimension mismatch")
	}
	if !errors.Is(err, glearn.ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got: %v", err)
	}
}

func TestLogisticRegressionInvalidC(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{-1, 0, -2, 0, 1, 0, 2, 0})
	y := []float64{0, 0, 1, 1}

	cfg := NewLogisticRegression(WithLogisticC(0))
	_, err := cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for C=0")
	}
	if !errors.Is(err, glearn.ErrInvalidParameter) {
		t.Errorf("expected ErrInvalidParameter, got: %v", err)
	}
}

func TestLogisticRegressionSingleClass(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})
	y := []float64{1, 1, 1} // only one class

	cfg := NewLogisticRegression()
	_, err := cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for single class")
	}
	if !errors.Is(err, glearn.ErrInvalidParameter) {
		t.Errorf("expected ErrInvalidParameter, got: %v", err)
	}
}

func TestLogisticRegressionGetClasses(t *testing.T) {
	lr := &LogisticRegression{
		Classes: []float64{0, 1, 2},
	}
	classes := lr.GetClasses()
	if len(classes) != 3 {
		t.Fatalf("expected 3 classes, got %d", len(classes))
	}
	// Verify it's a copy.
	classes[0] = 999
	if lr.Classes[0] == 999 {
		t.Error("GetClasses should return a copy")
	}
}

func TestLogisticRegressionGetCoefficients(t *testing.T) {
	lr := &LogisticRegression{
		Coefficients: [][]float64{{1, 2}, {3, 4}},
	}
	coef := lr.GetCoefficients()
	if len(coef) != 4 {
		t.Fatalf("expected 4 coefficients, got %d", len(coef))
	}
	expected := []float64{1, 2, 3, 4}
	for i, v := range expected {
		if coef[i] != v {
			t.Errorf("coefficient[%d]: expected %g, got %g", i, v, coef[i])
		}
	}
}
