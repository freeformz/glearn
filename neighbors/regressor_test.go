package neighbors

import (
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

func TestKNeighborsRegressorSimpleLinear(t *testing.T) {
	// Simple linear data: y = 2*x
	X := mat.NewDense(5, 1, []float64{
		1, 2, 3, 4, 5,
	})
	y := []float64{2, 4, 6, 8, 10}

	cfg := NewKNeighborsRegressor(WithRegressorK(2))
	pred, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Predict at x=3.1 -> nearest neighbors are x=3 (y=6) and x=4 (y=8) -> mean = 7
	testX := mat.NewDense(1, 1, []float64{3.1})
	preds, err := pred.Predict(testX)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	if math.Abs(preds[0]-7.0) > 1e-10 {
		t.Errorf("expected prediction ~7.0, got %g", preds[0])
	}
}

func TestKNeighborsRegressorK1PerfectTraining(t *testing.T) {
	// K=1 should perfectly reproduce training data.
	X := mat.NewDense(5, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
		9, 10,
	})
	y := []float64{10, 20, 30, 40, 50}

	cfg := NewKNeighborsRegressor(WithRegressorK(1))
	pred, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	preds, err := pred.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	for i := range y {
		if math.Abs(preds[i]-y[i]) > 1e-10 {
			t.Errorf("K=1 training prediction[%d]: expected %g, got %g", i, y[i], preds[i])
		}
	}
}

func TestKNeighborsRegressorDimensionMismatch(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	y := []float64{10, 20, 30, 40}

	cfg := NewKNeighborsRegressor(WithRegressorK(2))
	pred, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Wrong number of features.
	badX := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})

	_, err = pred.Predict(badX)
	if err == nil {
		t.Fatal("expected dimension mismatch error, got nil")
	}
}

func TestKNeighborsRegressorDistanceWeights(t *testing.T) {
	// Two neighbors with different distances.
	X := mat.NewDense(2, 1, []float64{
		0, // y = 10
		1, // y = 20
	})
	y := []float64{10, 20}

	// Query at x=0.9: distances are 0.9 and 0.1.
	// Uniform: mean(10, 20) = 15.
	// Distance: (10/0.9 + 20/0.1) / (1/0.9 + 1/0.1) = (11.11 + 200) / (1.11 + 10) = 211.11 / 11.11 ~= 19.0
	testX := mat.NewDense(1, 1, []float64{0.9})

	// Uniform weights.
	cfgUniform := NewKNeighborsRegressor(WithRegressorK(2), WithRegressorWeights(WeightsUniform))
	predUniform, err := cfgUniform.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	predsUniform, err := predUniform.Predict(testX)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	if math.Abs(predsUniform[0]-15.0) > 1e-10 {
		t.Errorf("uniform weights: expected 15.0, got %g", predsUniform[0])
	}

	// Distance weights.
	cfgDist := NewKNeighborsRegressor(WithRegressorK(2), WithRegressorWeights(WeightsDistance))
	predDist, err := cfgDist.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	predsDist, err := predDist.Predict(testX)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	// Closer to 20 than 10.
	if predsDist[0] <= 15.0 {
		t.Errorf("distance weights: expected > 15.0, got %g", predsDist[0])
	}
	if predsDist[0] >= 20.0 {
		t.Errorf("distance weights: expected < 20.0, got %g", predsDist[0])
	}
}

func TestKNeighborsRegressorScore(t *testing.T) {
	// K=1 on training data should give R2 = 1.0
	X := mat.NewDense(5, 1, []float64{
		1, 2, 3, 4, 5,
	})
	y := []float64{2, 4, 6, 8, 10}

	cfg := NewKNeighborsRegressor(WithRegressorK(1))
	pred, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	knn := pred.(*KNeighborsRegressor)
	score, err := knn.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}
	if math.Abs(score-1.0) > 1e-10 {
		t.Errorf("expected R2=1.0 for K=1 on training data, got %g", score)
	}
}

func TestKNeighborsRegressorInvalidK(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	y := []float64{10, 20, 30, 40}

	// K = 0
	cfg := NewKNeighborsRegressor(WithRegressorK(0))
	_, err := cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for K=0, got nil")
	}

	// K > nSamples
	cfg = NewKNeighborsRegressor(WithRegressorK(10))
	_, err = cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for K > nSamples, got nil")
	}
}

func TestKNeighborsRegressorEmptyInput(t *testing.T) {
	t.Skip("gonum panics on zero-dimension matrices; empty input cannot reach Fit")
}

// TestKNeighborsRegressorCompileTimeChecks verifies interface satisfaction.
func TestKNeighborsRegressorCompileTimeChecks(t *testing.T) {
	var _ glearn.Estimator = KNeighborsRegressorConfig{}
	var _ glearn.Predictor = (*KNeighborsRegressor)(nil)
}
