package naivebayes

import (
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

func TestGaussianNBLinearlySeparable(t *testing.T) {
	// Two classes: class 0 centered at (0, 0), class 1 centered at (10, 10).
	X := mat.NewDense(10, 2, []float64{
		0.1, 0.2,
		-0.1, -0.2,
		0.2, -0.1,
		-0.2, 0.1,
		0.0, 0.0,
		10.1, 10.2,
		9.9, 9.8,
		10.2, 9.9,
		9.8, 10.1,
		10.0, 10.0,
	})
	y := []float64{0, 0, 0, 0, 0, 1, 1, 1, 1, 1}

	cfg := NewGaussianNB()
	result, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Predict on training data — should get all correct.
	preds, err := result.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	for i, pred := range preds {
		if pred != y[i] {
			t.Errorf("sample %d: expected class %v, got %v", i, y[i], pred)
		}
	}

	// Predict on new points.
	newX := mat.NewDense(2, 2, []float64{
		0.5, 0.5,  // should be class 0
		9.5, 9.5,  // should be class 1
	})
	newPreds, err := result.Predict(newX)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	if newPreds[0] != 0 {
		t.Errorf("expected class 0 for (0.5, 0.5), got %v", newPreds[0])
	}
	if newPreds[1] != 1 {
		t.Errorf("expected class 1 for (9.5, 9.5), got %v", newPreds[1])
	}
}

func TestGaussianNBProbabilitiesSumToOne(t *testing.T) {
	X := mat.NewDense(6, 2, []float64{
		1, 2,
		2, 3,
		3, 4,
		10, 11,
		11, 12,
		12, 13,
	})
	y := []float64{0, 0, 0, 1, 1, 1}

	cfg := NewGaussianNB()
	result, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	gnb := result.(*GaussianNB)

	testX := mat.NewDense(3, 2, []float64{
		2, 3,    // clearly class 0
		11, 12,  // clearly class 1
		6, 7,    // ambiguous
	})

	probs, err := gnb.PredictProbabilities(testX)
	if err != nil {
		t.Fatalf("PredictProbabilities failed: %v", err)
	}

	nSamples, nClasses := probs.Dims()
	if nSamples != 3 || nClasses != 2 {
		t.Fatalf("expected probs shape (3, 2), got (%d, %d)", nSamples, nClasses)
	}

	for i := range nSamples {
		sum := 0.0
		for j := range nClasses {
			p := probs.At(i, j)
			if p < 0 || p > 1 {
				t.Errorf("sample %d, class %d: probability %f out of [0, 1]", i, j, p)
			}
			sum += p
		}
		if math.Abs(sum-1.0) > 1e-10 {
			t.Errorf("sample %d: probabilities sum to %f, expected 1.0", i, sum)
		}
	}
}

func TestGaussianNBDimensionMismatch(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	y := []float64{0, 0, 1, 1}

	cfg := NewGaussianNB()
	result, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Predict with wrong number of features.
	badX := mat.NewDense(1, 3, []float64{1, 2, 3})
	_, err = result.Predict(badX)
	if err == nil {
		t.Fatal("expected error for dimension mismatch in Predict")
	}

	// PredictProbabilities with wrong number of features.
	gnb := result.(*GaussianNB)
	_, err = gnb.PredictProbabilities(badX)
	if err == nil {
		t.Fatal("expected error for dimension mismatch in PredictProbabilities")
	}
}

func TestGaussianNBFitInputMismatch(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{1, 2, 3, 4, 5, 6, 7, 8})
	y := []float64{0, 0, 1} // wrong length

	cfg := NewGaussianNB()
	_, err := cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for X/y length mismatch")
	}
}

func TestGaussianNBMultiClass(t *testing.T) {
	// Three classes.
	X := mat.NewDense(9, 2, []float64{
		0, 0,
		0.1, -0.1,
		-0.1, 0.1,
		5, 5,
		5.1, 4.9,
		4.9, 5.1,
		10, 0,
		10.1, -0.1,
		9.9, 0.1,
	})
	y := []float64{0, 0, 0, 1, 1, 1, 2, 2, 2}

	cfg := NewGaussianNB()
	result, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	gnb := result.(*GaussianNB)
	if len(gnb.Classes) != 3 {
		t.Fatalf("expected 3 classes, got %d", len(gnb.Classes))
	}

	// Predict on training data.
	preds, err := result.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	for i, pred := range preds {
		if pred != y[i] {
			t.Errorf("sample %d: expected class %v, got %v", i, y[i], pred)
		}
	}

	// Check probabilities sum to 1.
	probs, err := gnb.PredictProbabilities(X)
	if err != nil {
		t.Fatalf("PredictProbabilities failed: %v", err)
	}
	nSamples, nClasses := probs.Dims()
	for i := range nSamples {
		sum := 0.0
		for j := range nClasses {
			sum += probs.At(i, j)
		}
		if math.Abs(sum-1.0) > 1e-10 {
			t.Errorf("sample %d: probabilities sum to %f", i, sum)
		}
	}
}

func TestGaussianNBVarSmoothing(t *testing.T) {
	// Test that VarSmoothing prevents division by zero with constant features.
	X := mat.NewDense(4, 2, []float64{
		1, 5, // class 0: feature 0 is constant
		1, 6,
		2, 5, // class 1: feature 0 is constant
		2, 6,
	})
	y := []float64{0, 0, 1, 1}

	cfg := NewGaussianNB(WithVarSmoothing(1e-9))
	result, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Should not produce NaN/Inf in predictions.
	preds, err := result.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	for i, p := range preds {
		if math.IsNaN(p) || math.IsInf(p, 0) {
			t.Errorf("sample %d: got NaN or Inf prediction", i)
		}
	}
}

func TestGaussianNBGetClasses(t *testing.T) {
	X := mat.NewDense(4, 1, []float64{1, 2, 3, 4})
	y := []float64{0, 0, 1, 1}

	cfg := NewGaussianNB()
	result, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	gnb := result.(*GaussianNB)
	classes := gnb.GetClasses()
	if len(classes) != 2 {
		t.Fatalf("expected 2 classes, got %d", len(classes))
	}
	if classes[0] != 0 || classes[1] != 1 {
		t.Errorf("expected classes [0, 1], got %v", classes)
	}
}

// Compile-time checks.
var _ glearn.Estimator = GaussianNBConfig{}
var _ glearn.Predictor = (*GaussianNB)(nil)
var _ glearn.Classifier = (*GaussianNB)(nil)
