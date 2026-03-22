package linear

import (
	"errors"
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

func TestRidgeCompileTimeChecks(t *testing.T) {
	var _ glearn.Estimator = RidgeConfig{}
	var _ glearn.Predictor = (*Ridge)(nil)
	var _ glearn.Scorer = (*Ridge)(nil)
}

func TestRidgeFitPredict(t *testing.T) {
	// y = 2*x1 + 3*x2 + 1
	X := mat.NewDense(5, 2, []float64{
		1, 1,
		2, 1,
		3, 2,
		4, 2,
		5, 3,
	})
	y := []float64{6, 8, 13, 15, 20}

	cfg := NewRidge(WithRidgeAlpha(0.001)) // small alpha -> close to OLS
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	ridge := model.(*Ridge)

	// With very small alpha, should be close to OLS solution.
	if math.Abs(ridge.Coefficients[0]-2.0) > 0.1 {
		t.Errorf("expected coefficient[0] ~2.0, got %g", ridge.Coefficients[0])
	}
	if math.Abs(ridge.Coefficients[1]-3.0) > 0.1 {
		t.Errorf("expected coefficient[1] ~3.0, got %g", ridge.Coefficients[1])
	}

	// Predict.
	preds, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	for i, p := range preds {
		if math.Abs(p-y[i]) > 0.5 {
			t.Errorf("prediction[%d]: expected ~%g, got %g", i, y[i], p)
		}
	}

	// Score should be close to 1.0.
	score, err := ridge.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}
	if score < 0.99 {
		t.Errorf("expected R² > 0.99, got %g", score)
	}
}

func TestRidgeRegularization(t *testing.T) {
	// Coefficients with large alpha should be smaller than OLS.
	X := mat.NewDense(10, 2, []float64{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
		10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
	})
	y := []float64{5, 7, 9, 11, 13, 15, 17, 19, 21, 23}

	olsCfg := NewLinearRegression()
	olsModel, err := olsCfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("OLS Fit failed: %v", err)
	}
	olsLR := olsModel.(*LinearRegression)

	ridgeCfg := NewRidge(WithRidgeAlpha(10.0))
	ridgeModel, err := ridgeCfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Ridge Fit failed: %v", err)
	}
	ridgeLR := ridgeModel.(*Ridge)

	// Ridge coefficients should have smaller magnitude.
	olsNorm := math.Abs(olsLR.Coefficients[0]) + math.Abs(olsLR.Coefficients[1])
	ridgeNorm := math.Abs(ridgeLR.Coefficients[0]) + math.Abs(ridgeLR.Coefficients[1])
	if ridgeNorm >= olsNorm {
		t.Errorf("expected ridge coefficient norm (%g) < OLS coefficient norm (%g)", ridgeNorm, olsNorm)
	}
}

func TestRidgeNoIntercept(t *testing.T) {
	// y = 2*x1 + 3*x2
	X := mat.NewDense(5, 2, []float64{
		1, 1,
		2, 1,
		3, 2,
		4, 2,
		5, 3,
	})
	y := []float64{5, 7, 12, 14, 19}

	cfg := NewRidge(WithRidgeAlpha(0.001), WithRidgeFitIntercept(false))
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	ridge := model.(*Ridge)
	if math.Abs(ridge.Intercept) > 1e-10 {
		t.Errorf("expected intercept ~0.0, got %g", ridge.Intercept)
	}
}

func TestRidgeDimensionMismatch(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})
	y := []float64{1, 2, 3}

	cfg := NewRidge()
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

func TestRidgeInvalidAlpha(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})
	y := []float64{1, 2, 3}

	cfg := NewRidge(WithRidgeAlpha(-1.0))
	_, err := cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for negative alpha")
	}
	if !errors.Is(err, glearn.ErrInvalidParameter) {
		t.Errorf("expected ErrInvalidParameter, got: %v", err)
	}
}
