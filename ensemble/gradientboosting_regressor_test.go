package ensemble_test

import (
	"errors"
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/ensemble"
	"gonum.org/v1/gonum/mat"
)

func TestGradientBoostingRegressor_FitPredict(t *testing.T) {
	X, y := makeRegressionData(100)

	cfg := ensemble.NewGradientBoostingRegressor(
		ensemble.WithGBNTrees(50),
		ensemble.WithGBLearningRate(0.1),
		ensemble.WithGBMaxDepth(3),
		ensemble.WithGBSeed(42),
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

	if r2 < 0.9 {
		t.Errorf("expected R^2 >= 0.9, got %f", r2)
	}
}

func TestGradientBoostingRegressor_MoreIterationsReducesError(t *testing.T) {
	X, y := makeRegressionData(100)

	cfg10 := ensemble.NewGradientBoostingRegressor(
		ensemble.WithGBNTrees(5),
		ensemble.WithGBLearningRate(0.1),
		ensemble.WithGBMaxDepth(3),
		ensemble.WithGBSeed(42),
	)
	cfg100 := ensemble.NewGradientBoostingRegressor(
		ensemble.WithGBNTrees(100),
		ensemble.WithGBLearningRate(0.1),
		ensemble.WithGBMaxDepth(3),
		ensemble.WithGBSeed(42),
	)

	pred10, err := cfg10.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit with 5 trees failed: %v", err)
	}
	pred100, err := cfg100.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit with 100 trees failed: %v", err)
	}

	// Compute MSE for both.
	preds10, _ := pred10.Predict(X)
	preds100, _ := pred100.Predict(X)

	mse10 := computeMSE(y, preds10)
	mse100 := computeMSE(y, preds100)

	if mse100 > mse10 {
		t.Errorf("more iterations should reduce training error: MSE(5 trees)=%f, MSE(100 trees)=%f", mse10, mse100)
	}
}

func TestGradientBoostingRegressor_DimensionMismatch(t *testing.T) {
	X, y := makeRegressionData(50)

	cfg := ensemble.NewGradientBoostingRegressor(
		ensemble.WithGBNTrees(10),
		ensemble.WithGBSeed(42),
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

func TestGradientBoostingRegressor_Subsample(t *testing.T) {
	X, y := makeRegressionData(100)

	cfg := ensemble.NewGradientBoostingRegressor(
		ensemble.WithGBNTrees(50),
		ensemble.WithGBLearningRate(0.1),
		ensemble.WithGBMaxDepth(3),
		ensemble.WithGBSubsample(0.8),
		ensemble.WithGBSeed(42),
	)

	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit with subsample failed: %v", err)
	}

	scorer := predictor.(glearn.Scorer)
	r2, err := scorer.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}

	if r2 < 0.7 {
		t.Errorf("expected R^2 >= 0.7 with subsample, got %f", r2)
	}
}

func TestGradientBoostingRegressor_ConstantTarget(t *testing.T) {
	n := 30
	nFeatures := 2
	data := make([]float64, n*nFeatures)
	labels := make([]float64, n)
	for i := range n {
		data[i*nFeatures+0] = float64(i)
		data[i*nFeatures+1] = float64(i * 2)
		labels[i] = 7.0
	}
	X := mat.NewDense(n, nFeatures, data)

	cfg := ensemble.NewGradientBoostingRegressor(
		ensemble.WithGBNTrees(10),
		ensemble.WithGBSeed(42),
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
		if math.Abs(p-7.0) > 0.1 {
			t.Errorf("pred[%d] = %f, expected ~7.0", i, p)
		}
	}
}

func TestGradientBoostingRegressor_InvalidParameters(t *testing.T) {
	X, y := makeRegressionData(20)

	// Zero learning rate.
	cfg := ensemble.NewGradientBoostingRegressor(
		ensemble.WithGBLearningRate(0),
	)
	_, err := cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Error("expected error for zero learning rate")
	}
	if !errors.Is(err, glearn.ErrInvalidParameter) {
		t.Errorf("expected ErrInvalidParameter, got: %v", err)
	}
}

func computeMSE(yTrue, yPred []float64) float64 {
	sum := 0.0
	for i := range yTrue {
		diff := yTrue[i] - yPred[i]
		sum += diff * diff
	}
	return sum / float64(len(yTrue))
}
