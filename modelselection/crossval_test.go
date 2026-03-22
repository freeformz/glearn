package modelselection

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

// mockEstimator implements glearn.Estimator for testing.
type mockEstimator struct {
	fitIntercept bool
}

func (m mockEstimator) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures := X.Dims()
	if len(y) != nSamples {
		return nil, glearn.ErrDimensionMismatch
	}

	// Simple: store the mean of y as the prediction.
	sum := 0.0
	for _, v := range y {
		sum += v
	}
	meanVal := sum / float64(nSamples)

	return &mockPredictor{meanPred: meanVal, nFeatures: nFeatures}, nil
}

type mockPredictor struct {
	meanPred  float64
	nFeatures int
}

func (m *mockPredictor) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, nFeatures := X.Dims()
	if nFeatures != m.nFeatures {
		return nil, glearn.ErrDimensionMismatch
	}
	preds := make([]float64, nSamples)
	for i := range nSamples {
		preds[i] = m.meanPred
	}
	return preds, nil
}

func r2Scorer(yTrue, yPred []float64) float64 {
	n := len(yTrue)
	meanY := 0.0
	for _, v := range yTrue {
		meanY += v
	}
	meanY /= float64(n)

	ssRes := 0.0
	ssTot := 0.0
	for i := range yTrue {
		diff := yTrue[i] - yPred[i]
		ssRes += diff * diff
		dm := yTrue[i] - meanY
		ssTot += dm * dm
	}
	if ssTot == 0 {
		if ssRes == 0 {
			return 1.0
		}
		return 0.0
	}
	return 1.0 - ssRes/ssTot
}

func TestCrossValScore_Basic(t *testing.T) {
	// Create simple data: y = X[:,0] + X[:,1] with some noise.
	nSamples := 100
	nFeatures := 2
	data := make([]float64, nSamples*nFeatures)
	y := make([]float64, nSamples)
	for i := range nSamples {
		data[i*nFeatures] = float64(i)
		data[i*nFeatures+1] = float64(i * 2)
		y[i] = float64(i) + float64(i*2)
	}
	X := mat.NewDense(nSamples, nFeatures, data)

	est := mockEstimator{}
	scores, err := CrossValScore(t.Context(), est, X, y, 5, r2Scorer, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(scores) != 5 {
		t.Fatalf("expected 5 scores, got %d", len(scores))
	}

	// All scores should be finite numbers.
	for i, s := range scores {
		if math.IsNaN(s) || math.IsInf(s, 0) {
			t.Errorf("score %d is not finite: %g", i, s)
		}
	}
}

func TestCrossValScore_ReturnsCorrectNumberOfScores(t *testing.T) {
	nSamples := 50
	X := mat.NewDense(nSamples, 2, nil)
	y := make([]float64, nSamples)
	for i := range nSamples {
		X.Set(i, 0, float64(i))
		X.Set(i, 1, float64(i+1))
		y[i] = float64(i)
	}

	for _, cv := range []int{2, 3, 5, 10} {
		scores, err := CrossValScore(t.Context(), mockEstimator{}, X, y, cv, r2Scorer, 42)
		if err != nil {
			t.Fatalf("cv=%d: unexpected error: %v", cv, err)
		}
		if len(scores) != cv {
			t.Errorf("cv=%d: expected %d scores, got %d", cv, cv, len(scores))
		}
	}
}

func TestCrossValScore_InvalidInputs(t *testing.T) {
	X := mat.NewDense(10, 2, nil)
	y := make([]float64, 10)

	t.Run("nil estimator", func(t *testing.T) {
		_, err := CrossValScore(t.Context(), nil, X, y, 5, r2Scorer, 42)
		if !errors.Is(err, glearn.ErrInvalidParameter) {
			t.Errorf("expected ErrInvalidParameter, got %v", err)
		}
	})

	t.Run("nil X", func(t *testing.T) {
		_, err := CrossValScore(t.Context(), mockEstimator{}, nil, y, 5, r2Scorer, 42)
		if !errors.Is(err, glearn.ErrEmptyInput) {
			t.Errorf("expected ErrEmptyInput, got %v", err)
		}
	})

	t.Run("nil scorer", func(t *testing.T) {
		_, err := CrossValScore(t.Context(), mockEstimator{}, X, y, 5, nil, 42)
		if !errors.Is(err, glearn.ErrInvalidParameter) {
			t.Errorf("expected ErrInvalidParameter, got %v", err)
		}
	})

	t.Run("mismatched y", func(t *testing.T) {
		_, err := CrossValScore(t.Context(), mockEstimator{}, X, make([]float64, 5), 5, r2Scorer, 42)
		if !errors.Is(err, glearn.ErrDimensionMismatch) {
			t.Errorf("expected ErrDimensionMismatch, got %v", err)
		}
	})
}

func TestCrossValScore_ContextCancellation(t *testing.T) {
	X := mat.NewDense(100, 2, nil)
	y := make([]float64, 100)
	for i := range 100 {
		X.Set(i, 0, float64(i))
		y[i] = float64(i)
	}

	ctx, cancel := context.WithCancel(t.Context())
	cancel() // Cancel immediately.

	_, err := CrossValScore(ctx, mockEstimator{}, X, y, 5, r2Scorer, 42)
	if err == nil {
		t.Fatal("expected error for cancelled context, got nil")
	}
}

func TestCrossValScore_Reproducibility(t *testing.T) {
	nSamples := 50
	X := mat.NewDense(nSamples, 2, nil)
	y := make([]float64, nSamples)
	for i := range nSamples {
		X.Set(i, 0, float64(i))
		X.Set(i, 1, float64(i*3))
		y[i] = float64(i * 2)
	}

	scores1, err := CrossValScore(t.Context(), mockEstimator{}, X, y, 5, r2Scorer, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	scores2, err := CrossValScore(t.Context(), mockEstimator{}, X, y, 5, r2Scorer, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for i := range scores1 {
		if scores1[i] != scores2[i] {
			t.Errorf("score %d differs: %g vs %g", i, scores1[i], scores2[i])
		}
	}
}
