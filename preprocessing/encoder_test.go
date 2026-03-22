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
	_ glearn.Transformer       = OneHotEncoderConfig{}
	_ glearn.FitTransformer    = OneHotEncoderConfig{}
	_ glearn.FittedTransformer = (*OneHotEncoder)(nil)

	_ glearn.Transformer       = LabelEncoderConfig{}
	_ glearn.FitTransformer    = LabelEncoderConfig{}
	_ glearn.FittedTransformer = (*LabelEncoder)(nil)
)

func TestOneHotEncoder_Basic(t *testing.T) {
	// 4 samples, 1 feature with categories {0, 1, 2}.
	X := mat.NewDense(4, 1, []float64{0, 1, 2, 1})

	cfg := NewOneHotEncoder()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	rows, cols := result.Dims()
	if rows != 4 || cols != 3 {
		t.Fatalf("result dims = %dx%d, want 4x3", rows, cols)
	}

	want := [][]float64{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
		{0, 1, 0},
	}
	for i := range rows {
		for j := range cols {
			if result.At(i, j) != want[i][j] {
				t.Errorf("result[%d][%d] = %g, want %g", i, j, result.At(i, j), want[i][j])
			}
		}
	}
}

func TestOneHotEncoder_MultipleFeatures(t *testing.T) {
	// 3 samples, 2 features.
	// Feature 0: {0, 1} -> 2 columns
	// Feature 1: {0, 1, 2} -> 3 columns
	// Total: 5 output columns.
	X := mat.NewDense(3, 2, []float64{
		0, 0,
		1, 1,
		0, 2,
	})

	cfg := NewOneHotEncoder()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	rows, cols := result.Dims()
	if rows != 3 || cols != 5 {
		t.Fatalf("result dims = %dx%d, want 3x5", rows, cols)
	}

	want := [][]float64{
		{1, 0, 1, 0, 0}, // feat0=0 -> [1,0], feat1=0 -> [1,0,0]
		{0, 1, 0, 1, 0}, // feat0=1 -> [0,1], feat1=1 -> [0,1,0]
		{1, 0, 0, 0, 1}, // feat0=0 -> [1,0], feat1=2 -> [0,0,1]
	}
	for i := range rows {
		for j := range cols {
			if result.At(i, j) != want[i][j] {
				t.Errorf("result[%d][%d] = %g, want %g", i, j, result.At(i, j), want[i][j])
			}
		}
	}
}

func TestOneHotEncoder_DropFirst(t *testing.T) {
	X := mat.NewDense(3, 1, []float64{0, 1, 2})

	cfg := NewOneHotEncoder(WithDropFirst(true))
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	rows, cols := result.Dims()
	if rows != 3 || cols != 2 {
		t.Fatalf("result dims = %dx%d, want 3x2", rows, cols)
	}

	// Categories are [0, 1, 2], dropping first (0):
	// 0 -> [0, 0]
	// 1 -> [1, 0]
	// 2 -> [0, 1]
	want := [][]float64{
		{0, 0},
		{1, 0},
		{0, 1},
	}
	for i := range rows {
		for j := range cols {
			if result.At(i, j) != want[i][j] {
				t.Errorf("result[%d][%d] = %g, want %g", i, j, result.At(i, j), want[i][j])
			}
		}
	}
}

func TestOneHotEncoder_UnknownCategory(t *testing.T) {
	XTrain := mat.NewDense(3, 1, []float64{0, 1, 2})
	XTest := mat.NewDense(1, 1, []float64{3}) // unknown category

	cfg := NewOneHotEncoder()
	ft, err := cfg.Fit(t.Context(), XTrain)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	_, err = ft.Transform(XTest)
	if err == nil {
		t.Fatal("Transform should fail with unknown category")
	}
	if !errors.Is(err, glearn.ErrInvalidParameter) {
		t.Errorf("error should wrap ErrInvalidParameter, got: %v", err)
	}
}

func TestOneHotEncoder_NegativeValue(t *testing.T) {
	X := mat.NewDense(2, 1, []float64{0, -1})

	cfg := NewOneHotEncoder()
	_, err := cfg.Fit(t.Context(), X)
	if err == nil {
		t.Fatal("Fit should fail with negative value")
	}
}

func TestOneHotEncoder_NonInteger(t *testing.T) {
	X := mat.NewDense(2, 1, []float64{0, 1.5})

	cfg := NewOneHotEncoder()
	_, err := cfg.Fit(t.Context(), X)
	if err == nil {
		t.Fatal("Fit should fail with non-integer value")
	}
}

func TestOneHotEncoder_NaN(t *testing.T) {
	X := mat.NewDense(2, 1, []float64{0, math.NaN()})

	cfg := NewOneHotEncoder()
	_, err := cfg.Fit(t.Context(), X)
	if err == nil {
		t.Fatal("Fit should fail with NaN value")
	}
}

func TestOneHotEncoder_EmptyInput(t *testing.T) {
	t.Skip("gonum panics on zero-dimension matrices; empty input cannot reach Fit")
}

func TestOneHotEncoder_DimensionMismatch(t *testing.T) {
	XTrain := mat.NewDense(3, 2, []float64{0, 0, 1, 1, 2, 2})
	XTest := mat.NewDense(2, 1, []float64{0, 1})

	cfg := NewOneHotEncoder()
	ft, err := cfg.Fit(t.Context(), XTrain)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	_, err = ft.Transform(XTest)
	if err == nil {
		t.Fatal("Transform should fail with dimension mismatch")
	}
}

func TestOneHotEncoder_FitTransform(t *testing.T) {
	X := mat.NewDense(3, 1, []float64{0, 1, 2})

	cfg := NewOneHotEncoder()
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
			if result.At(i, j) != result2.At(i, j) {
				t.Errorf("FitTransform[%d][%d] = %g, Fit+Transform = %g", i, j, result.At(i, j), result2.At(i, j))
			}
		}
	}
}

func TestOneHotEncoder_NOutputFeatures(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{
		0, 0,
		1, 1,
		0, 2,
	})

	cfg := NewOneHotEncoder()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	enc := ft.(*OneHotEncoder)
	if enc.NOutputFeatures() != 5 {
		t.Errorf("NOutputFeatures() = %d, want 5", enc.NOutputFeatures())
	}

	// With drop first.
	cfg2 := NewOneHotEncoder(WithDropFirst(true))
	ft2, err := cfg2.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	enc2 := ft2.(*OneHotEncoder)
	if enc2.NOutputFeatures() != 3 {
		t.Errorf("NOutputFeatures() with DropFirst = %d, want 3", enc2.NOutputFeatures())
	}
}

// --- LabelEncoder tests ---

func TestLabelEncoder_Basic(t *testing.T) {
	X := mat.NewDense(5, 1, []float64{3, 1, 2, 1, 3})

	cfg := NewLabelEncoder()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	enc := ft.(*LabelEncoder)
	// Classes should be sorted: [1, 2, 3].
	wantClasses := []float64{1, 2, 3}
	if len(enc.Classes) != len(wantClasses) {
		t.Fatalf("len(Classes) = %d, want %d", len(enc.Classes), len(wantClasses))
	}
	for i, c := range enc.Classes {
		if c != wantClasses[i] {
			t.Errorf("Classes[%d] = %g, want %g", i, c, wantClasses[i])
		}
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	// 3->2, 1->0, 2->1, 1->0, 3->2
	want := []float64{2, 0, 1, 0, 2}
	for i, w := range want {
		if result.At(i, 0) != w {
			t.Errorf("result[%d] = %g, want %g", i, result.At(i, 0), w)
		}
	}
}

func TestLabelEncoder_InverseTransform(t *testing.T) {
	X := mat.NewDense(4, 1, []float64{10, 20, 30, 20})

	cfg := NewLabelEncoder()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	enc := ft.(*LabelEncoder)

	transformed, err := enc.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	recovered, err := enc.InverseTransform(transformed)
	if err != nil {
		t.Fatalf("InverseTransform: %v", err)
	}

	for i := range 4 {
		got := recovered.At(i, 0)
		want := X.At(i, 0)
		if got != want {
			t.Errorf("recovered[%d] = %g, want %g", i, got, want)
		}
	}
}

func TestLabelEncoder_UnknownLabel(t *testing.T) {
	XTrain := mat.NewDense(3, 1, []float64{1, 2, 3})
	XTest := mat.NewDense(1, 1, []float64{4})

	cfg := NewLabelEncoder()
	ft, err := cfg.Fit(t.Context(), XTrain)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	_, err = ft.Transform(XTest)
	if err == nil {
		t.Fatal("Transform should fail with unknown label")
	}
	if !errors.Is(err, glearn.ErrInvalidParameter) {
		t.Errorf("error should wrap ErrInvalidParameter, got: %v", err)
	}
}

func TestLabelEncoder_InverseTransformOutOfRange(t *testing.T) {
	X := mat.NewDense(3, 1, []float64{1, 2, 3})

	cfg := NewLabelEncoder()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	enc := ft.(*LabelEncoder)

	bad := mat.NewDense(1, 1, []float64{5})
	_, err = enc.InverseTransform(bad)
	if err == nil {
		t.Fatal("InverseTransform should fail with out-of-range index")
	}
}

func TestLabelEncoder_MultipleColumns(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})

	cfg := NewLabelEncoder()
	_, err := cfg.Fit(t.Context(), X)
	if err == nil {
		t.Fatal("Fit should fail with multiple columns")
	}
	if !errors.Is(err, glearn.ErrDimensionMismatch) {
		t.Errorf("error should wrap ErrDimensionMismatch, got: %v", err)
	}
}

func TestLabelEncoder_EmptyInput(t *testing.T) {
	t.Skip("gonum panics on zero-dimension matrices; empty input cannot reach Fit")
}

func TestLabelEncoder_FitTransform(t *testing.T) {
	X := mat.NewDense(3, 1, []float64{3, 1, 2})

	cfg := NewLabelEncoder()
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
	for i := range 3 {
		if result.At(i, 0) != result2.At(i, 0) {
			t.Errorf("FitTransform[%d] = %g, Fit+Transform = %g", i, result.At(i, 0), result2.At(i, 0))
		}
	}
}

func TestLabelEncoder_SingleClass(t *testing.T) {
	X := mat.NewDense(3, 1, []float64{5, 5, 5})

	cfg := NewLabelEncoder()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	result, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform: %v", err)
	}

	for i := range 3 {
		if result.At(i, 0) != 0 {
			t.Errorf("result[%d] = %g, want 0", i, result.At(i, 0))
		}
	}
}
