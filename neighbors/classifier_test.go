package neighbors

import (
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

func TestKNeighborsClassifierSimple2D(t *testing.T) {
	// Two clusters with clear separation.
	// Class 0: points near (0, 0)
	// Class 1: points near (10, 10)
	X := mat.NewDense(8, 2, []float64{
		0, 0,
		0.5, 0.5,
		1, 0,
		0, 1,
		10, 10,
		10.5, 10.5,
		11, 10,
		10, 11,
	})
	y := []float64{0, 0, 0, 0, 1, 1, 1, 1}

	cfg := NewKNeighborsClassifier(WithClassifierK(3))
	pred, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Predict on test points clearly in each cluster.
	testX := mat.NewDense(2, 2, []float64{
		0.2, 0.3, // near class 0
		9.8, 10.2, // near class 1
	})

	preds, err := pred.Predict(testX)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	if preds[0] != 0 {
		t.Errorf("expected class 0 for point near (0,0), got %g", preds[0])
	}
	if preds[1] != 1 {
		t.Errorf("expected class 1 for point near (10,10), got %g", preds[1])
	}
}

func TestKNeighborsClassifierK1PerfectTraining(t *testing.T) {
	// K=1 should give perfect accuracy on training data.
	X := mat.NewDense(6, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
		9, 10,
		11, 12,
	})
	y := []float64{0, 1, 0, 1, 0, 1}

	cfg := NewKNeighborsClassifier(WithClassifierK(1))
	pred, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	preds, err := pred.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	for i := range y {
		if preds[i] != y[i] {
			t.Errorf("K=1 training prediction[%d]: expected %g, got %g", i, y[i], preds[i])
		}
	}
}

func TestKNeighborsClassifierDimensionMismatch(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	y := []float64{0, 1, 0, 1}

	cfg := NewKNeighborsClassifier(WithClassifierK(2))
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

func TestKNeighborsClassifierDistanceWeights(t *testing.T) {
	// Two class-0 points far away, one class-1 point very close to query.
	X := mat.NewDense(3, 1, []float64{
		0,    // class 0: far from query
		100,  // class 0: far from query
		10.0, // class 1: right at query
	})
	y := []float64{0, 0, 1}

	// With K=3 uniform: 2 class-0 vs 1 class-1 -> predict 0
	cfg := NewKNeighborsClassifier(WithClassifierK(3), WithClassifierWeights(WeightsUniform))
	pred, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	testX := mat.NewDense(1, 1, []float64{10.01})
	preds, err := pred.Predict(testX)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	if preds[0] != 0 {
		t.Errorf("uniform weights: expected class 0, got %g", preds[0])
	}

	// With K=3 distance: class 1 at distance 0.01 dominates over class 0 at distances 10.01 and 89.99.
	cfgDist := NewKNeighborsClassifier(WithClassifierK(3), WithClassifierWeights(WeightsDistance))
	predDist, err := cfgDist.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	predsDist, err := predDist.Predict(testX)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}
	if predsDist[0] != 1 {
		t.Errorf("distance weights: expected class 1, got %g", predsDist[0])
	}
}

func TestKNeighborsClassifierScore(t *testing.T) {
	X := mat.NewDense(6, 2, []float64{
		0, 0,
		0.5, 0.5,
		1, 0,
		10, 10,
		10.5, 10.5,
		11, 10,
	})
	y := []float64{0, 0, 0, 1, 1, 1}

	cfg := NewKNeighborsClassifier(WithClassifierK(3))
	pred, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	knn := pred.(*KNeighborsClassifier)
	score, err := knn.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}
	if math.Abs(score-1.0) > 1e-10 {
		t.Errorf("expected perfect score on well-separated data, got %g", score)
	}
}

func TestKNeighborsClassifierInvalidK(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	y := []float64{0, 1, 0, 1}

	// K = 0
	cfg := NewKNeighborsClassifier(WithClassifierK(0))
	_, err := cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for K=0, got nil")
	}

	// K > nSamples
	cfg = NewKNeighborsClassifier(WithClassifierK(10))
	_, err = cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for K > nSamples, got nil")
	}
}

func TestKNeighborsClassifierGetClasses(t *testing.T) {
	X := mat.NewDense(6, 2, []float64{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
	})
	y := []float64{0, 1, 2, 0, 1, 2}

	cfg := NewKNeighborsClassifier(WithClassifierK(3))
	pred, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	knn := pred.(*KNeighborsClassifier)
	classes := knn.GetClasses()
	if len(classes) != 3 {
		t.Fatalf("expected 3 classes, got %d", len(classes))
	}
	expected := []float64{0, 1, 2}
	for i, c := range classes {
		if c != expected[i] {
			t.Errorf("class[%d]: expected %g, got %g", i, expected[i], c)
		}
	}
}

func TestKNeighborsClassifierEmptyInput(t *testing.T) {
	t.Skip("gonum panics on zero-dimension matrices; empty input cannot reach Fit")
}

// TestKNeighborsClassifierCompileTimeChecks verifies interface satisfaction.
func TestKNeighborsClassifierCompileTimeChecks(t *testing.T) {
	var _ glearn.Estimator = KNeighborsClassifierConfig{}
	var _ glearn.Predictor = (*KNeighborsClassifier)(nil)
}
