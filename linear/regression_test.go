package linear

import (
	"errors"
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

func TestLinearRegressionCompileTimeChecks(t *testing.T) {
	// These are also checked at package level, but explicitly verify in a test.
	var _ glearn.Estimator = LinearRegressionConfig{}
	var _ glearn.Predictor = (*LinearRegression)(nil)
	var _ glearn.Scorer = (*LinearRegression)(nil)
}

func TestLinearRegressionFitPredict(t *testing.T) {
	// y = 2*x1 + 3*x2 + 1
	X := mat.NewDense(5, 2, []float64{
		1, 1,
		2, 1,
		3, 2,
		4, 2,
		5, 3,
	})
	y := []float64{
		2*1 + 3*1 + 1,
		2*2 + 3*1 + 1,
		2*3 + 3*2 + 1,
		2*4 + 3*2 + 1,
		2*5 + 3*3 + 1,
	}

	cfg := NewLinearRegression()
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	lr := model.(*LinearRegression)

	// Check coefficients.
	if math.Abs(lr.Coefficients[0]-2.0) > 1e-6 {
		t.Errorf("expected coefficient[0] ~2.0, got %g", lr.Coefficients[0])
	}
	if math.Abs(lr.Coefficients[1]-3.0) > 1e-6 {
		t.Errorf("expected coefficient[1] ~3.0, got %g", lr.Coefficients[1])
	}
	if math.Abs(lr.Intercept-1.0) > 1e-6 {
		t.Errorf("expected intercept ~1.0, got %g", lr.Intercept)
	}

	// Predict on training data.
	preds, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	for i, p := range preds {
		if math.Abs(p-y[i]) > 1e-6 {
			t.Errorf("prediction[%d]: expected %g, got %g", i, y[i], p)
		}
	}

	// Score should be ~1.0.
	score, err := lr.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}
	if math.Abs(score-1.0) > 1e-6 {
		t.Errorf("expected R² ~1.0, got %g", score)
	}
}

func TestLinearRegressionNoIntercept(t *testing.T) {
	// y = 2*x1 + 3*x2 (no intercept)
	X := mat.NewDense(5, 2, []float64{
		1, 1,
		2, 1,
		3, 2,
		4, 2,
		5, 3,
	})
	y := []float64{
		2*1 + 3*1,
		2*2 + 3*1,
		2*3 + 3*2,
		2*4 + 3*2,
		2*5 + 3*3,
	}

	cfg := NewLinearRegression(WithFitIntercept(false))
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	lr := model.(*LinearRegression)
	if math.Abs(lr.Coefficients[0]-2.0) > 1e-6 {
		t.Errorf("expected coefficient[0] ~2.0, got %g", lr.Coefficients[0])
	}
	if math.Abs(lr.Coefficients[1]-3.0) > 1e-6 {
		t.Errorf("expected coefficient[1] ~3.0, got %g", lr.Coefficients[1])
	}
	if math.Abs(lr.Intercept) > 1e-6 {
		t.Errorf("expected intercept ~0.0, got %g", lr.Intercept)
	}
}

func TestLinearRegressionDimensionMismatch(t *testing.T) {
	X := mat.NewDense(5, 2, []float64{
		1, 5,
		2, 3,
		3, 1,
		4, 4,
		5, 2,
	})
	y := []float64{1, 2, 3, 4, 5}

	cfg := NewLinearRegression()
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Predict with wrong number of features.
	XBad := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	_, err = model.Predict(XBad)
	if err == nil {
		t.Fatal("expected error for dimension mismatch")
	}
	if !errors.Is(err, glearn.ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got: %v", err)
	}
}

func TestLinearRegressionFitDimensionMismatch(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
	})
	y := []float64{1, 2} // wrong length

	cfg := NewLinearRegression()
	_, err := cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for dimension mismatch")
	}
	if !errors.Is(err, glearn.ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got: %v", err)
	}
}

func TestLinearRegressionGetCoefficients(t *testing.T) {
	lr := &LinearRegression{
		Coefficients: []float64{1.0, 2.0, 3.0},
		NFeatures:    3,
	}
	coef := lr.GetCoefficients()
	if len(coef) != 3 {
		t.Fatalf("expected 3 coefficients, got %d", len(coef))
	}
	// Verify it's a copy.
	coef[0] = 999
	if lr.Coefficients[0] == 999 {
		t.Error("GetCoefficients should return a copy")
	}
}
