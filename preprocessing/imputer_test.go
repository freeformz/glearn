package preprocessing

import (
	"errors"
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface verification.
var (
	_ glearn.Transformer       = SimpleImputerConfig{}
	_ glearn.FitTransformer    = SimpleImputerConfig{}
	_ glearn.FittedTransformer = (*SimpleImputer)(nil)
)

func TestSimpleImputer_Mean(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{
		1, 10,
		math.NaN(), 20,
		3, math.NaN(),
		5, 40,
	})

	cfg := NewSimpleImputer() // Default: StrategyMean
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	imp := ft.(*SimpleImputer)
	// Mean of col 0: (1+3+5)/3 = 3.0
	// Mean of col 1: (10+20+40)/3 = 23.333...
	wantStats := []float64{3.0, 70.0 / 3.0}
	for j, s := range imp.Statistics {
		if math.Abs(s-wantStats[j]) > tolerance {
			t.Errorf("Statistics[%d] = %g, want %g", j, s, wantStats[j])
		}
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	// NaN values should be replaced.
	if math.IsNaN(result.At(1, 0)) {
		t.Error("result[1][0] should not be NaN")
	}
	if math.Abs(result.At(1, 0)-3.0) > tolerance {
		t.Errorf("result[1][0] = %g, want 3.0", result.At(1, 0))
	}

	if math.IsNaN(result.At(2, 1)) {
		t.Error("result[2][1] should not be NaN")
	}
	if math.Abs(result.At(2, 1)-70.0/3.0) > tolerance {
		t.Errorf("result[2][1] = %g, want %g", result.At(2, 1), 70.0/3.0)
	}

	// Non-NaN values should be unchanged.
	if result.At(0, 0) != 1 {
		t.Errorf("result[0][0] = %g, want 1", result.At(0, 0))
	}
	if result.At(0, 1) != 10 {
		t.Errorf("result[0][1] = %g, want 10", result.At(0, 1))
	}
}

func TestSimpleImputer_Median(t *testing.T) {
	X := mat.NewDense(5, 1, []float64{
		1,
		math.NaN(),
		3,
		5,
		7,
	})

	cfg := NewSimpleImputer(WithStrategy(StrategyMedian))
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	imp := ft.(*SimpleImputer)
	// Median of [1, 3, 5, 7] (4 values, even count) = (3+5)/2 = 4.0
	if math.Abs(imp.Statistics[0]-4.0) > tolerance {
		t.Errorf("Statistics[0] = %g, want 4.0", imp.Statistics[0])
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	if math.Abs(result.At(1, 0)-4.0) > tolerance {
		t.Errorf("result[1][0] = %g, want 4.0", result.At(1, 0))
	}
}

func TestSimpleImputer_MedianOddCount(t *testing.T) {
	X := mat.NewDense(4, 1, []float64{
		1,
		math.NaN(),
		3,
		5,
	})

	cfg := NewSimpleImputer(WithStrategy(StrategyMedian))
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	imp := ft.(*SimpleImputer)
	// Median of [1, 3, 5] (3 values, odd count) = 3.0
	if math.Abs(imp.Statistics[0]-3.0) > tolerance {
		t.Errorf("Statistics[0] = %g, want 3.0", imp.Statistics[0])
	}
}

func TestSimpleImputer_Constant(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{
		1, math.NaN(),
		math.NaN(), 2,
		3, 4,
	})

	cfg := NewSimpleImputer(WithStrategy(StrategyConstant), WithFillValue(-999))
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	if result.At(1, 0) != -999 {
		t.Errorf("result[1][0] = %g, want -999", result.At(1, 0))
	}
	if result.At(0, 1) != -999 {
		t.Errorf("result[0][1] = %g, want -999", result.At(0, 1))
	}
	// Non-NaN values unchanged.
	if result.At(0, 0) != 1 {
		t.Errorf("result[0][0] = %g, want 1", result.At(0, 0))
	}
	if result.At(2, 1) != 4 {
		t.Errorf("result[2][1] = %g, want 4", result.At(2, 1))
	}
}

func TestSimpleImputer_NoMissingValues(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
	})

	cfg := NewSimpleImputer()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	rows, cols := result.Dims()
	for i := range rows {
		for j := range cols {
			if result.At(i, j) != X.At(i, j) {
				t.Errorf("result[%d][%d] = %g, want %g", i, j, result.At(i, j), X.At(i, j))
			}
		}
	}
}

func TestSimpleImputer_AllNaN(t *testing.T) {
	X := mat.NewDense(3, 1, []float64{
		math.NaN(),
		math.NaN(),
		math.NaN(),
	})

	cfg := NewSimpleImputer()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	imp := ft.(*SimpleImputer)
	// When all values are NaN, the statistic should be NaN.
	if !math.IsNaN(imp.Statistics[0]) {
		t.Errorf("Statistics[0] = %g, want NaN for all-missing column", imp.Statistics[0])
	}

	// Transform should propagate NaN (NaN replaced with NaN statistic).
	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}
	for i := range 3 {
		if !math.IsNaN(result.At(i, 0)) {
			t.Errorf("result[%d][0] = %g, want NaN", i, result.At(i, 0))
		}
	}
}

func TestSimpleImputer_EmptyInput(t *testing.T) {
	t.Skip("gonum panics on zero-dimension matrices; empty input cannot reach Fit")
}

func TestSimpleImputer_DimensionMismatch(t *testing.T) {
	XTrain := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})
	XTest := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})

	cfg := NewSimpleImputer()
	ft, err := cfg.Fit(t.Context(), XTrain)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	_, err = ft.Transform(XTest)
	if err == nil {
		t.Fatal("Transform should fail with dimension mismatch")
	}
	if !errors.Is(err, glearn.ErrDimensionMismatch) {
		t.Errorf("error should wrap ErrDimensionMismatch, got: %v", err)
	}
}

func TestSimpleImputer_SingleSample(t *testing.T) {
	X := mat.NewDense(1, 2, []float64{math.NaN(), 5})

	cfg := NewSimpleImputer()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	imp := ft.(*SimpleImputer)
	// Single sample with NaN: mean is NaN.
	if !math.IsNaN(imp.Statistics[0]) {
		t.Errorf("Statistics[0] = %g, want NaN", imp.Statistics[0])
	}
	if imp.Statistics[1] != 5 {
		t.Errorf("Statistics[1] = %g, want 5", imp.Statistics[1])
	}
}

func TestSimpleImputer_SingleFeature(t *testing.T) {
	X := mat.NewDense(4, 1, []float64{
		1,
		math.NaN(),
		3,
		5,
	})

	cfg := NewSimpleImputer()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	// Mean of [1, 3, 5] = 3.0
	if math.Abs(result.At(1, 0)-3.0) > tolerance {
		t.Errorf("result[1][0] = %g, want 3.0", result.At(1, 0))
	}
}

func TestSimpleImputer_FitTransform(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{
		1, math.NaN(),
		math.NaN(), 4,
		3, 6,
	})

	cfg := NewSimpleImputer()
	ft, result, err := cfg.FitTransform(t.Context(), X)
	if err != nil {
		t.Fatalf("FitTransform: %v", err)
	}

	ft2, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	result2, err := ft2.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	_ = ft
	rows, cols := result.Dims()
	for i := range rows {
		for j := range cols {
			g := result.At(i, j)
			w := result2.At(i, j)
			if math.IsNaN(g) && math.IsNaN(w) {
				continue
			}
			if math.Abs(g-w) > tolerance {
				t.Errorf("FitTransform[%d][%d] = %g, Fit+Transform = %g", i, j, g, w)
			}
		}
	}
}

func TestSimpleImputer_OriginalUnmodified(t *testing.T) {
	data := []float64{1, math.NaN(), 3, 4}
	X := mat.NewDense(2, 2, data)

	cfg := NewSimpleImputer()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	_, err = ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	// Original data should not be modified.
	if !math.IsNaN(X.At(0, 1)) {
		t.Error("original X was modified by Transform")
	}
}
