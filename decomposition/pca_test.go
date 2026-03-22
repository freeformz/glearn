package decomposition

import (
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

func TestPCATransformReducesDimensionality(t *testing.T) {
	// 5 samples, 4 features -> reduce to 2 components
	X := mat.NewDense(5, 4, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
		17, 18, 19, 20,
	})

	cfg := NewPCA(WithNComponents(2))
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	transformed, err := ft.Transform(X)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	rows, cols := transformed.Dims()
	if rows != 5 {
		t.Errorf("expected 5 rows, got %d", rows)
	}
	if cols != 2 {
		t.Errorf("expected 2 columns, got %d", cols)
	}
}

func TestPCAComponentsOrthogonal(t *testing.T) {
	// Data with some structure.
	X := mat.NewDense(6, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		2, 4, 6,
		3, 6, 9,
		5, 10, 15,
	})

	cfg := NewPCA(WithNComponents(2))
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	pca := ft.(*PCA)

	// Check orthogonality: dot product of component 0 and component 1 should be ~0.
	comp0 := pca.Components.RawRowView(0)
	comp1 := pca.Components.RawRowView(1)
	dot := 0.0
	for i := range comp0 {
		dot += comp0[i] * comp1[i]
	}
	if math.Abs(dot) > 1e-10 {
		t.Errorf("components not orthogonal: dot product = %g", dot)
	}

	// Check that each component is a unit vector.
	for c := range 2 {
		row := pca.Components.RawRowView(c)
		norm := 0.0
		for _, v := range row {
			norm += v * v
		}
		norm = math.Sqrt(norm)
		if math.Abs(norm-1.0) > 1e-10 {
			t.Errorf("component %d is not unit length: norm = %g", c, norm)
		}
	}
}

func TestPCAExplainedVarianceRatioSumLessThanOrEqualOne(t *testing.T) {
	X := mat.NewDense(5, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 10,
		2, 3, 5,
		6, 7, 8,
	})

	cfg := NewPCA() // keep all components
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	pca := ft.(*PCA)
	sum := 0.0
	for _, r := range pca.ExplainedVarianceRatio {
		if r < 0 {
			t.Errorf("negative explained variance ratio: %g", r)
		}
		sum += r
	}
	if sum > 1.0+1e-10 {
		t.Errorf("explained variance ratios sum to %g, expected <= 1.0", sum)
	}
}

func TestPCAInverseTransformRecovers(t *testing.T) {
	X := mat.NewDense(5, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 10,
		2, 3, 5,
		6, 7, 8,
	})

	// Keep all components -> inverse transform should be exact.
	cfg := NewPCA()
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	pca := ft.(*PCA)

	transformed, err := pca.Transform(X)
	if err != nil {
		t.Fatalf("Transform failed: %v", err)
	}

	recovered, err := pca.InverseTransform(transformed)
	if err != nil {
		t.Fatalf("InverseTransform failed: %v", err)
	}

	rows, cols := X.Dims()
	for i := range rows {
		for j := range cols {
			diff := math.Abs(X.At(i, j) - recovered.At(i, j))
			if diff > 1e-10 {
				t.Errorf("InverseTransform mismatch at (%d,%d): original=%g recovered=%g diff=%g",
					i, j, X.At(i, j), recovered.At(i, j), diff)
			}
		}
	}
}

func TestPCAFitTransform(t *testing.T) {
	X := mat.NewDense(5, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 10,
		2, 3, 5,
		6, 7, 8,
	})

	cfg := NewPCA(WithNComponents(2))
	ft, transformed, err := cfg.FitTransform(t.Context(), X)
	if err != nil {
		t.Fatalf("FitTransform failed: %v", err)
	}

	if ft == nil {
		t.Fatal("FitTransform returned nil FittedTransformer")
	}

	rows, cols := transformed.Dims()
	if rows != 5 || cols != 2 {
		t.Errorf("expected (5, 2), got (%d, %d)", rows, cols)
	}
}

func TestPCADimensionMismatch(t *testing.T) {
	X := mat.NewDense(5, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 10,
		2, 3, 5,
		6, 7, 8,
	})

	cfg := NewPCA(WithNComponents(2))
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Transform with wrong number of features.
	badX := mat.NewDense(3, 4, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})

	_, err = ft.Transform(badX)
	if err == nil {
		t.Fatal("expected dimension mismatch error, got nil")
	}
}

func TestPCAInvalidNComponents(t *testing.T) {
	X := mat.NewDense(5, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 10,
		2, 3, 5,
		6, 7, 8,
	})

	// NComponents exceeds max(nSamples, nFeatures) = min(5,3) = 3
	cfg := NewPCA(WithNComponents(10))
	_, err := cfg.Fit(t.Context(), X)
	if err == nil {
		t.Fatal("expected invalid parameter error, got nil")
	}
}

func TestPCADefaultNComponents(t *testing.T) {
	X := mat.NewDense(5, 3, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 10,
		2, 3, 5,
		6, 7, 8,
	})

	cfg := NewPCA() // default: keep all components
	ft, err := cfg.Fit(t.Context(), X)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	pca := ft.(*PCA)
	if pca.NComponents != 3 {
		t.Errorf("expected 3 components (min of 5 samples, 3 features), got %d", pca.NComponents)
	}
}

func TestPCAEmptyInput(t *testing.T) {
	t.Skip("gonum panics on zero-dimension matrices; empty input cannot reach Fit")
}

// TestPCACompileTimeChecks verifies that PCA types satisfy the expected interfaces.
func TestPCACompileTimeChecks(t *testing.T) {
	var _ glearn.Transformer = PCAConfig{}
	var _ glearn.FitTransformer = PCAConfig{}
	var _ glearn.FittedTransformer = (*PCA)(nil)
}
