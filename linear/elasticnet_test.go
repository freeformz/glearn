package linear

import (
	"errors"
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

func TestElasticNetCompileTimeChecks(t *testing.T) {
	var _ glearn.Estimator = ElasticNetConfig{}
	var _ glearn.Predictor = (*ElasticNet)(nil)
	var _ glearn.Scorer = (*ElasticNet)(nil)
}

func TestElasticNetFitPredict(t *testing.T) {
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

	cfg := NewElasticNet(WithElasticNetAlpha(0.001))
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	en := model.(*ElasticNet)

	if math.Abs(en.Coefficients[0]-2.0) > 0.1 {
		t.Errorf("expected coefficient[0] ~2.0, got %g", en.Coefficients[0])
	}
	if math.Abs(en.Coefficients[1]-3.0) > 0.1 {
		t.Errorf("expected coefficient[1] ~3.0, got %g", en.Coefficients[1])
	}

	preds, err := model.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	for i, p := range preds {
		if math.Abs(p-y[i]) > 0.5 {
			t.Errorf("prediction[%d]: expected ~%g, got %g", i, y[i], p)
		}
	}

	score, err := en.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}
	if score < 0.99 {
		t.Errorf("expected R² > 0.99, got %g", score)
	}
}

func TestElasticNetRegularization(t *testing.T) {
	// ElasticNet coefficients should be smaller than OLS.
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

	enCfg := NewElasticNet(WithElasticNetAlpha(1.0), WithElasticNetL1Ratio(0.5))
	enModel, err := enCfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("ElasticNet Fit failed: %v", err)
	}
	enLR := enModel.(*ElasticNet)

	olsNorm := math.Abs(olsLR.Coefficients[0]) + math.Abs(olsLR.Coefficients[1])
	enNorm := math.Abs(enLR.Coefficients[0]) + math.Abs(enLR.Coefficients[1])
	if enNorm >= olsNorm {
		t.Errorf("expected elastic net coefficient norm (%g) < OLS coefficient norm (%g)", enNorm, olsNorm)
	}
}

func TestElasticNetL1RatioPureL2(t *testing.T) {
	// L1Ratio=0 should apply only L2 regularization (no sparsity).
	// Coefficients should be non-zero and smaller than OLS.
	X := mat.NewDense(6, 2, []float64{
		1, 1,
		2, 1,
		3, 2,
		4, 2,
		5, 3,
		6, 3,
	})
	y := []float64{6, 8, 13, 15, 20, 22}

	enCfg := NewElasticNet(WithElasticNetAlpha(1.0), WithElasticNetL1Ratio(0))
	enModel, err := enCfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("ElasticNet Fit failed: %v", err)
	}

	olsCfg := NewLinearRegression()
	olsModel, err := olsCfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("OLS Fit failed: %v", err)
	}

	en := enModel.(*ElasticNet)
	ols := olsModel.(*LinearRegression)

	// With pure L2, all coefficients should be nonzero (no sparsity).
	for j := range 2 {
		if en.Coefficients[j] == 0 {
			t.Errorf("expected coefficient[%d] to be nonzero with L1Ratio=0", j)
		}
	}

	// Elastic net coefficients should be shrunk relative to OLS.
	enNorm := math.Abs(en.Coefficients[0]) + math.Abs(en.Coefficients[1])
	olsNorm := math.Abs(ols.Coefficients[0]) + math.Abs(ols.Coefficients[1])
	if enNorm >= olsNorm {
		t.Errorf("expected elastic net coefficient norm (%g) < OLS coefficient norm (%g)", enNorm, olsNorm)
	}
}

func TestElasticNetL1RatioPureLasso(t *testing.T) {
	// L1Ratio=1 should behave like lasso.
	X := mat.NewDense(6, 2, []float64{
		1, 1,
		2, 1,
		3, 2,
		4, 2,
		5, 3,
		6, 3,
	})
	y := []float64{6, 8, 13, 15, 20, 22}

	enCfg := NewElasticNet(WithElasticNetAlpha(0.1), WithElasticNetL1Ratio(1.0))
	enModel, err := enCfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("ElasticNet Fit failed: %v", err)
	}

	lassoCfg := NewLasso(WithLassoAlpha(0.1))
	lassoModel, err := lassoCfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Lasso Fit failed: %v", err)
	}

	en := enModel.(*ElasticNet)
	lasso := lassoModel.(*Lasso)

	for j := range 2 {
		if math.Abs(en.Coefficients[j]-lasso.Coefficients[j]) > 0.1 {
			t.Errorf("coefficient[%d]: ElasticNet(L1Ratio=1) %g != Lasso %g",
				j, en.Coefficients[j], lasso.Coefficients[j])
		}
	}
}

func TestElasticNetNoIntercept(t *testing.T) {
	X := mat.NewDense(6, 2, []float64{
		1, 1,
		2, 1,
		3, 2,
		4, 2,
		5, 3,
		6, 3,
	})
	y := []float64{5, 7, 12, 14, 19, 21}

	cfg := NewElasticNet(WithElasticNetAlpha(0.001), WithElasticNetFitIntercept(false))
	model, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	en := model.(*ElasticNet)
	if math.Abs(en.Intercept) > 1e-10 {
		t.Errorf("expected intercept ~0.0, got %g", en.Intercept)
	}
}

func TestElasticNetDimensionMismatch(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})
	y := []float64{1, 2, 3}

	cfg := NewElasticNet()
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

func TestElasticNetInvalidParams(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})
	y := []float64{1, 2, 3}

	// Negative alpha.
	cfg := NewElasticNet(WithElasticNetAlpha(-1.0))
	_, err := cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for negative alpha")
	}
	if !errors.Is(err, glearn.ErrInvalidParameter) {
		t.Errorf("expected ErrInvalidParameter, got: %v", err)
	}

	// Invalid L1Ratio.
	cfg = NewElasticNet(WithElasticNetL1Ratio(1.5))
	_, err = cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for invalid L1Ratio")
	}
	if !errors.Is(err, glearn.ErrInvalidParameter) {
		t.Errorf("expected ErrInvalidParameter, got: %v", err)
	}
}
