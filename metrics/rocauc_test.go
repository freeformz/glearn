package metrics

import (
	"testing"
)

func TestROCAUC_Perfect(t *testing.T) {
	yTrue := []float64{0, 0, 1, 1}
	yScores := []float64{0.1, 0.2, 0.9, 0.8}
	got := ROCAUC(yTrue, yScores)
	if !almostEqual(got, 1.0, tolerance) {
		t.Errorf("ROCAUC = %v, want 1.0", got)
	}
}

func TestROCAUC_Worst(t *testing.T) {
	// Perfectly reversed predictions.
	yTrue := []float64{0, 0, 1, 1}
	yScores := []float64{0.9, 0.8, 0.1, 0.2}
	got := ROCAUC(yTrue, yScores)
	if !almostEqual(got, 0.0, tolerance) {
		t.Errorf("ROCAUC = %v, want 0.0", got)
	}
}

func TestROCAUC_Random(t *testing.T) {
	// Equal scores for all => should be ~0.5.
	yTrue := []float64{0, 0, 1, 1}
	yScores := []float64{0.5, 0.5, 0.5, 0.5}
	got := ROCAUC(yTrue, yScores)
	if !almostEqual(got, 0.5, tolerance) {
		t.Errorf("ROCAUC = %v, want 0.5", got)
	}
}

func TestROCAUC_KnownValue(t *testing.T) {
	yTrue := []float64{0, 0, 0, 1, 1, 1}
	yScores := []float64{0.1, 0.4, 0.6, 0.3, 0.7, 0.9}
	got := ROCAUC(yTrue, yScores)
	// Sorted by descending score:
	// score=0.9 label=1 => tp=1,fp=0 => (0/3, 1/3)
	// score=0.7 label=1 => tp=2,fp=0 => (0/3, 2/3)
	// score=0.6 label=0 => tp=2,fp=1 => (1/3, 2/3)
	// score=0.4 label=0 => tp=2,fp=2 => (2/3, 2/3)
	// score=0.3 label=1 => tp=3,fp=2 => (2/3, 3/3)
	// score=0.1 label=0 => tp=3,fp=3 => (3/3, 3/3)
	// AUC by trapezoidal rule:
	// (0-0)*(1/3+0)/2 + (0-0)*(2/3+1/3)/2 + (1/3-0)*(2/3+2/3)/2 + (2/3-1/3)*(2/3+2/3)/2 + (2/3-2/3)*(1+2/3)/2 + (1-2/3)*(1+1)/2
	// = 0 + 0 + (1/3)*(4/3)/2 + (1/3)*(4/3)/2 + 0 + (1/3)*2/2
	// = 0 + 0 + 4/18 + 4/18 + 0 + 1/3
	// = 2/9 + 2/9 + 3/9 = 7/9
	want := 7.0 / 9.0
	if !almostEqual(got, want, tolerance) {
		t.Errorf("ROCAUC = %v, want %v", got, want)
	}
}

func TestROCAUC_TwoSamples(t *testing.T) {
	yTrue := []float64{0, 1}
	yScores := []float64{0.2, 0.8}
	got := ROCAUC(yTrue, yScores)
	if !almostEqual(got, 1.0, tolerance) {
		t.Errorf("ROCAUC = %v, want 1.0", got)
	}
}

func TestROCAUC_AllPositive(t *testing.T) {
	yTrue := []float64{1, 1, 1}
	yScores := []float64{0.5, 0.6, 0.7}
	got := ROCAUC(yTrue, yScores)
	// AUC undefined when only one class present; we return 0.
	if got != 0.0 {
		t.Errorf("ROCAUC = %v, want 0.0 (all positive)", got)
	}
}

func TestROCAUC_AllNegative(t *testing.T) {
	yTrue := []float64{0, 0, 0}
	yScores := []float64{0.5, 0.6, 0.7}
	got := ROCAUC(yTrue, yScores)
	if got != 0.0 {
		t.Errorf("ROCAUC = %v, want 0.0 (all negative)", got)
	}
}

func TestROCAUC_LargerExample(t *testing.T) {
	yTrue := []float64{0, 0, 0, 0, 0, 1, 1, 1, 1, 1}
	yScores := []float64{0.1, 0.2, 0.3, 0.5, 0.6, 0.4, 0.55, 0.7, 0.8, 0.9}
	got := ROCAUC(yTrue, yScores)
	// Manually verified: AUC = 0.88
	want := 0.88
	if !almostEqual(got, want, tolerance) {
		t.Errorf("ROCAUC = %v, want %v", got, want)
	}
}

func TestROCAUC_TiedScoresDifferentLabels(t *testing.T) {
	// Test handling of tied scores with different true labels.
	yTrue := []float64{0, 1, 0, 1}
	yScores := []float64{0.5, 0.5, 0.5, 0.5}
	got := ROCAUC(yTrue, yScores)
	// All tied: AUC should be 0.5.
	if !almostEqual(got, 0.5, tolerance) {
		t.Errorf("ROCAUC = %v, want 0.5", got)
	}
}

func TestROCAUC_PanicOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty input")
		}
	}()
	ROCAUC(nil, nil)
}

func TestROCAUC_PanicOnLengthMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on length mismatch")
		}
	}()
	ROCAUC([]float64{0, 1}, []float64{0.5})
}
