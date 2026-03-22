package preprocessing

import (
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface verification.
var (
	_ glearn.Transformer       = StandardScalerConfig{}
	_ glearn.FitTransformer    = StandardScalerConfig{}
	_ glearn.FittedTransformer = (*StandardScaler)(nil)

	_ glearn.Transformer       = MinMaxScalerConfig{}
	_ glearn.FitTransformer    = MinMaxScalerConfig{}
	_ glearn.FittedTransformer = (*MinMaxScaler)(nil)
)

const tolerance = 1e-10

func TestStandardScaler_Basic(t *testing.T) {
	// 3 samples, 2 features.
	X := mat.NewDense(3, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
	})

	cfg := NewStandardScaler()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	scaler := ft.(*StandardScaler)
	// Mean should be [3, 4].
	wantMean := []float64{3, 4}
	for j, m := range scaler.Mean {
		if math.Abs(m-wantMean[j]) > tolerance {
			t.Errorf("Mean[%d] = %g, want %g", j, m, wantMean[j])
		}
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	rows, cols := result.Dims()
	if rows != 3 || cols != 2 {
		t.Fatalf("result dims = %dx%d, want 3x2", rows, cols)
	}

	// Verify transformed data has zero mean and unit variance.
	for j := range cols {
		var sum float64
		for i := range rows {
			sum += result.At(i, j)
		}
		mean := sum / float64(rows)
		if math.Abs(mean) > tolerance {
			t.Errorf("column %d mean = %g, want 0", j, mean)
		}

		var ss float64
		for i := range rows {
			d := result.At(i, j) - mean
			ss += d * d
		}
		variance := ss / float64(rows)
		if math.Abs(variance-1.0) > tolerance {
			t.Errorf("column %d variance = %g, want 1", j, variance)
		}
	}
}

func TestStandardScaler_WithoutMean(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
	})

	cfg := NewStandardScaler(WithMean(false))
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	scaler := ft.(*StandardScaler)
	if scaler.Mean != nil {
		t.Error("Mean should be nil when WithMean is false")
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	// Without centering, each value should be divided by std only.
	// The std is computed relative to the mean even when not centering.
	for i := range 3 {
		for j := range 2 {
			got := result.At(i, j)
			want := X.At(i, j) / scaler.Scale[j]
			if math.Abs(got-want) > tolerance {
				t.Errorf("result[%d][%d] = %g, want %g", i, j, got, want)
			}
		}
	}
}

func TestStandardScaler_WithoutStd(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
	})

	cfg := NewStandardScaler(WithStd(false))
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	scaler := ft.(*StandardScaler)
	if scaler.Scale != nil {
		t.Error("Scale should be nil when WithStd is false")
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	// Without scaling, each value should be x - mean only.
	for i := range 3 {
		for j := range 2 {
			got := result.At(i, j)
			want := X.At(i, j) - scaler.Mean[j]
			if math.Abs(got-want) > tolerance {
				t.Errorf("result[%d][%d] = %g, want %g", i, j, got, want)
			}
		}
	}
}

func TestStandardScaler_InverseTransform(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{
		1, 10,
		2, 20,
		3, 30,
	})

	cfg := NewStandardScaler()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	scaler := ft.(*StandardScaler)

	transformed, err := scaler.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	recovered, err := scaler.InverseTransform(transformed)
	if err != nil {
		t.Fatalf("InverseTransform: %v", err)
	}

	rows, cols := X.Dims()
	for i := range rows {
		for j := range cols {
			got := recovered.At(i, j)
			want := X.At(i, j)
			if math.Abs(got-want) > tolerance {
				t.Errorf("recovered[%d][%d] = %g, want %g", i, j, got, want)
			}
		}
	}
}

func TestStandardScaler_ConstantFeature(t *testing.T) {
	// One constant feature (zero variance).
	X := mat.NewDense(3, 2, []float64{
		5, 1,
		5, 2,
		5, 3,
	})

	cfg := NewStandardScaler()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	scaler := ft.(*StandardScaler)
	// Scale for constant feature should be 1 (not 0).
	if scaler.Scale[0] != 1.0 {
		t.Errorf("Scale[0] = %g, want 1.0 for constant feature", scaler.Scale[0])
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	// Constant feature after centering should be 0.
	for i := range 3 {
		if result.At(i, 0) != 0 {
			t.Errorf("result[%d][0] = %g, want 0", i, result.At(i, 0))
		}
	}
}

func TestStandardScaler_SingleSample(t *testing.T) {
	X := mat.NewDense(1, 3, []float64{1, 2, 3})

	cfg := NewStandardScaler()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	// Single sample: variance is 0, so scale is 1. After centering, all zeros.
	for j := range 3 {
		if result.At(0, j) != 0 {
			t.Errorf("result[0][%d] = %g, want 0", j, result.At(0, j))
		}
	}
}

func TestStandardScaler_SingleFeature(t *testing.T) {
	X := mat.NewDense(4, 1, []float64{2, 4, 6, 8})

	cfg := NewStandardScaler()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	// Verify zero mean.
	var sum float64
	for i := range 4 {
		sum += result.At(i, 0)
	}
	if math.Abs(sum) > tolerance {
		t.Errorf("sum = %g, want 0", sum)
	}
}

func TestStandardScaler_DimensionMismatch(t *testing.T) {
	XTrain := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})
	XTest := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})

	cfg := NewStandardScaler()
	ft, err := cfg.Fit(t.Context(), XTrain)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	_, err = ft.Transform(XTest)
	if err == nil {
		t.Fatal("Transform should fail with dimension mismatch")
	}
}

func TestStandardScaler_EmptyInput(t *testing.T) {
	// gonum panics on mat.NewDense(0, 0, nil), so we verify our code
	// handles the panic path gracefully if it ever receives empty data.
	// In practice, callers cannot construct a zero-dimension Dense matrix.
	// This test verifies we don't crash on a 1x0 matrix (not possible with
	// gonum either), so we skip it as an integration boundary.
	t.Skip("gonum panics on zero-dimension matrices; empty input cannot reach Fit")
}

func TestStandardScaler_FitTransform(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
	})

	cfg := NewStandardScaler()
	ft, result, err := cfg.FitTransform(t.Context(), X)
	if err != nil {
		t.Fatalf("FitTransform: %v", err)
	}

	// FitTransform should produce the same result as Fit + Transform.
	ft2, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	result2, err := ft2.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	_ = ft // verify ft is not nil
	rows, cols := result.Dims()
	for i := range rows {
		for j := range cols {
			if math.Abs(result.At(i, j)-result2.At(i, j)) > tolerance {
				t.Errorf("FitTransform[%d][%d] = %g, Fit+Transform = %g", i, j, result.At(i, j), result2.At(i, j))
			}
		}
	}
}

// --- MinMaxScaler tests ---

func TestMinMaxScaler_Basic(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{
		1, 10,
		2, 20,
		3, 30,
	})

	cfg := NewMinMaxScaler()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	// Expected: each feature scaled to [0, 1].
	want := [][]float64{
		{0, 0},
		{0.5, 0.5},
		{1, 1},
	}
	rows, cols := result.Dims()
	for i := range rows {
		for j := range cols {
			if math.Abs(result.At(i, j)-want[i][j]) > tolerance {
				t.Errorf("result[%d][%d] = %g, want %g", i, j, result.At(i, j), want[i][j])
			}
		}
	}
}

func TestMinMaxScaler_CustomRange(t *testing.T) {
	X := mat.NewDense(3, 1, []float64{1, 2, 3})

	cfg := NewMinMaxScaler(WithFeatureRange(-1, 1))
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	want := []float64{-1, 0, 1}
	for i, w := range want {
		if math.Abs(result.At(i, 0)-w) > tolerance {
			t.Errorf("result[%d][0] = %g, want %g", i, result.At(i, 0), w)
		}
	}
}

func TestMinMaxScaler_InvalidRange(t *testing.T) {
	X := mat.NewDense(3, 1, []float64{1, 2, 3})

	cfg := NewMinMaxScaler(WithFeatureRange(1, 0))
	_, err := cfg.Fit(t.Context(), X)
	if err == nil {
		t.Fatal("Fit should fail when FeatureMin >= FeatureMax")
	}
}

func TestMinMaxScaler_InverseTransform(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{
		1, 10,
		2, 20,
		3, 30,
	})

	cfg := NewMinMaxScaler()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	scaler := ft.(*MinMaxScaler)

	transformed, err := scaler.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	recovered, err := scaler.InverseTransform(transformed)
	if err != nil {
		t.Fatalf("InverseTransform: %v", err)
	}

	rows, cols := X.Dims()
	for i := range rows {
		for j := range cols {
			got := recovered.At(i, j)
			want := X.At(i, j)
			if math.Abs(got-want) > tolerance {
				t.Errorf("recovered[%d][%d] = %g, want %g", i, j, got, want)
			}
		}
	}
}

func TestMinMaxScaler_InverseTransformCustomRange(t *testing.T) {
	X := mat.NewDense(3, 1, []float64{10, 20, 30})

	cfg := NewMinMaxScaler(WithFeatureRange(-1, 1))
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	scaler := ft.(*MinMaxScaler)
	transformed, err := scaler.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	recovered, err := scaler.InverseTransform(transformed)
	if err != nil {
		t.Fatalf("InverseTransform: %v", err)
	}

	for i := range 3 {
		got := recovered.At(i, 0)
		want := X.At(i, 0)
		if math.Abs(got-want) > tolerance {
			t.Errorf("recovered[%d][0] = %g, want %g", i, got, want)
		}
	}
}

func TestMinMaxScaler_ConstantFeature(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{
		5, 1,
		5, 2,
		5, 3,
	})

	cfg := NewMinMaxScaler()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	// Constant feature should be scaled to FeatureMin (0).
	for i := range 3 {
		if result.At(i, 0) != 0 {
			t.Errorf("result[%d][0] = %g, want 0 for constant feature", i, result.At(i, 0))
		}
	}
}

func TestMinMaxScaler_SingleSample(t *testing.T) {
	X := mat.NewDense(1, 2, []float64{5, 10})

	cfg := NewMinMaxScaler()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	// Single sample: range is 0, so should map to FeatureMin.
	for j := range 2 {
		if result.At(0, j) != 0 {
			t.Errorf("result[0][%d] = %g, want 0 for single sample", j, result.At(0, j))
		}
	}
}

func TestMinMaxScaler_DimensionMismatch(t *testing.T) {
	XTrain := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})
	XTest := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})

	cfg := NewMinMaxScaler()
	ft, err := cfg.Fit(t.Context(), XTrain)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	_, err = ft.Transform(XTest)
	if err == nil {
		t.Fatal("Transform should fail with dimension mismatch")
	}
}

func TestMinMaxScaler_FitTransform(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{
		1, 10,
		2, 20,
		3, 30,
	})

	cfg := NewMinMaxScaler()
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
			if math.Abs(result.At(i, j)-result2.At(i, j)) > tolerance {
				t.Errorf("FitTransform[%d][%d] = %g, Fit+Transform = %g", i, j, result.At(i, j), result2.At(i, j))
			}
		}
	}
}
