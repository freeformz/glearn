package linear

import (
	"errors"
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

func TestLassoCompileTimeChecks(t *testing.T) {
	var _ glearn.Estimator = LassoConfig{}
	var _ glearn.Predictor = (*Lasso)(nil)
	var _ glearn.Scorer = (*Lasso)(nil)
}

func TestLassoFitPredict(t *testing.T) {
	// y = 2*x1 + 3*x2 + 1
	X := mat.NewDense(6, 2, []float64{
		1, 1,
		2, 1,
		3, 2,
		4, 2,
		5, 3,
		6, 3,
	})
	y := []float64{6, 8, 13, 15, 20, 22}

	cfg := NewLasso(WithLassoAlpha(0.001)) // small alpha -> close to OLS
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	lasso := model.(*Lasso)

	// With very small alpha, should be close to OLS solution.
	if math.Abs(lasso.Coefficients[0]-2.0) > 0.1 {
		t.Errorf("expected coefficient[0] ~2.0, got %g", lasso.Coefficients[0])
	}
	if math.Abs(lasso.Coefficients[1]-3.0) > 0.1 {
		t.Errorf("expected coefficient[1] ~3.0, got %g", lasso.Coefficients[1])
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
	score, err := lasso.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}
	if score < 0.99 {
		t.Errorf("expected R² > 0.99, got %g", score)
	}
}

func TestLassoSparsity(t *testing.T) {
	// With strong L1 regularization, some coefficients should become zero.
	// y depends only on x1, but we include irrelevant features.
	X := mat.NewDense(20, 3, nil)
	y := make([]float64, 20)
	for i := 0; i < 20; i++ {
		x1 := float64(i) + 1
		x2 := float64(i%5) * 0.01 // irrelevant, near zero
		x3 := float64(i%3) * 0.01 // irrelevant, near zero
		X.Set(i, 0, x1)
		X.Set(i, 1, x2)
		X.Set(i, 2, x3)
		y[i] = 3*x1 + 5 // only depends on x1
	}

	cfg := NewLasso(WithLassoAlpha(0.5))
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	lasso := model.(*Lasso)

	// x1 coefficient should be significant.
	if math.Abs(lasso.Coefficients[0]) < 1.0 {
		t.Errorf("expected coefficient[0] to be significant, got %g", lasso.Coefficients[0])
	}

	// Irrelevant features should have near-zero or zero coefficients.
	if math.Abs(lasso.Coefficients[1]) > 1.0 {
		t.Errorf("expected coefficient[1] near zero, got %g", lasso.Coefficients[1])
	}
	if math.Abs(lasso.Coefficients[2]) > 1.0 {
		t.Errorf("expected coefficient[2] near zero, got %g", lasso.Coefficients[2])
	}
}

func TestLassoRegularization(t *testing.T) {
	// Ridge vs Lasso: lasso coefficients should be smaller than OLS.
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

	lassoCfg := NewLasso(WithLassoAlpha(1.0))
	lassoModel, err := lassoCfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Lasso Fit failed: %v", err)
	}
	lassoLR := lassoModel.(*Lasso)

	olsNorm := math.Abs(olsLR.Coefficients[0]) + math.Abs(olsLR.Coefficients[1])
	lassoNorm := math.Abs(lassoLR.Coefficients[0]) + math.Abs(lassoLR.Coefficients[1])
	if lassoNorm >= olsNorm {
		t.Errorf("expected lasso coefficient norm (%g) < OLS coefficient norm (%g)", lassoNorm, olsNorm)
	}
}

func TestLassoNoIntercept(t *testing.T) {
	// y = 2*x1 + 3*x2
	X := mat.NewDense(6, 2, []float64{
		1, 1,
		2, 1,
		3, 2,
		4, 2,
		5, 3,
		6, 3,
	})
	y := []float64{5, 7, 12, 14, 19, 21}

	cfg := NewLasso(WithLassoAlpha(0.001), WithLassoFitIntercept(false))
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	lasso := model.(*Lasso)
	if math.Abs(lasso.Intercept) > 1e-10 {
		t.Errorf("expected intercept ~0.0, got %g", lasso.Intercept)
	}
}

func TestLassoDimensionMismatch(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})
	y := []float64{1, 2, 3}

	cfg := NewLasso()
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

func TestLassoInvalidAlpha(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})
	y := []float64{1, 2, 3}

	cfg := NewLasso(WithLassoAlpha(-1.0))
	_, err := cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for negative alpha")
	}
	if !errors.Is(err, glearn.ErrInvalidParameter) {
		t.Errorf("expected ErrInvalidParameter, got: %v", err)
	}
}
