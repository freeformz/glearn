package sparse

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// Compile-time verification that CSR satisfies mat.Matrix.
var _ mat.Matrix = (*CSR)(nil)

// helper to compare float64 with relative tolerance.
func approxEqual(a, b, tol float64) bool {
	if a == b {
		return true
	}
	diff := math.Abs(a - b)
	largest := math.Max(math.Abs(a), math.Abs(b))
	if largest == 0 {
		return diff < tol
	}
	return diff/largest < tol
}

func TestNewCSR(t *testing.T) {
	// 3x4 matrix:
	//  [1 0 0 2]
	//  [0 0 3 0]
	//  [4 0 5 6]
	data := []float64{1, 2, 3, 4, 5, 6}
	indices := []int{0, 3, 2, 0, 2, 3}
	indptr := []int{0, 2, 3, 6}

	m := NewCSR(3, 4, data, indices, indptr)
	r, c := m.Dims()
	if r != 3 || c != 4 {
		t.Fatalf("Dims() = (%d, %d), want (3, 4)", r, c)
	}
	if m.NNZ() != 6 {
		t.Fatalf("NNZ() = %d, want 6", m.NNZ())
	}

	// Verify the constructor copies input slices.
	data[0] = 999
	if m.Data[0] != 1 {
		t.Fatal("NewCSR did not copy data slice")
	}
}

func TestCSRAt(t *testing.T) {
	// 3x4 matrix:
	//  [1 0 0 2]
	//  [0 0 3 0]
	//  [4 0 5 6]
	data := []float64{1, 2, 3, 4, 5, 6}
	indices := []int{0, 3, 2, 0, 2, 3}
	indptr := []int{0, 2, 3, 6}
	m := NewCSR(3, 4, data, indices, indptr)

	tests := []struct {
		i, j int
		want float64
	}{
		{0, 0, 1},
		{0, 1, 0},
		{0, 2, 0},
		{0, 3, 2},
		{1, 0, 0},
		{1, 1, 0},
		{1, 2, 3},
		{1, 3, 0},
		{2, 0, 4},
		{2, 1, 0},
		{2, 2, 5},
		{2, 3, 6},
	}
	for _, tt := range tests {
		got := m.At(tt.i, tt.j)
		if got != tt.want {
			t.Errorf("At(%d, %d) = %f, want %f", tt.i, tt.j, got, tt.want)
		}
	}
}

func TestCSRAtPanicsOutOfRange(t *testing.T) {
	m := NewCSR(2, 3, nil, nil, []int{0, 0, 0})

	assertPanics(t, "negative row", func() { m.At(-1, 0) })
	assertPanics(t, "row too large", func() { m.At(2, 0) })
	assertPanics(t, "negative col", func() { m.At(0, -1) })
	assertPanics(t, "col too large", func() { m.At(0, 3) })
}

func TestCSRFromDense(t *testing.T) {
	dense := mat.NewDense(3, 4, []float64{
		1, 0, 0, 2,
		0, 0, 3, 0,
		4, 0, 5, 6,
	})

	m := CSRFromDense(dense)
	r, c := m.Dims()
	if r != 3 || c != 4 {
		t.Fatalf("Dims() = (%d, %d), want (3, 4)", r, c)
	}
	if m.NNZ() != 6 {
		t.Fatalf("NNZ() = %d, want 6", m.NNZ())
	}

	// Verify every element matches.
	for i := range r {
		for j := range c {
			if got, want := m.At(i, j), dense.At(i, j); got != want {
				t.Errorf("At(%d, %d) = %f, want %f", i, j, got, want)
			}
		}
	}
}

func TestCSRToDense(t *testing.T) {
	want := mat.NewDense(3, 4, []float64{
		1, 0, 0, 2,
		0, 0, 3, 0,
		4, 0, 5, 6,
	})

	m := CSRFromDense(want)
	got := m.ToDense()

	if !mat.Equal(got, want) {
		t.Errorf("ToDense() roundtrip failed:\ngot:  %v\nwant: %v", mat.Formatted(got), mat.Formatted(want))
	}
}

func TestCSRRoundtrip(t *testing.T) {
	// Roundtrip: Dense -> CSR -> Dense should be identity.
	original := mat.NewDense(2, 3, []float64{
		0, 7, 0,
		3, 0, 9,
	})
	got := CSRFromDense(original).ToDense()
	if !mat.Equal(got, original) {
		t.Errorf("Dense -> CSR -> Dense roundtrip failed")
	}
}

func TestCSRMatrixInterface(t *testing.T) {
	// Verify CSR works with gonum functions that accept mat.Matrix.
	dense := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	sparse := CSRFromDense(dense)

	// mat.Trace requires a square matrix, so test with element access.
	r, c := sparse.Dims()
	if r != 2 || c != 3 {
		t.Fatalf("unexpected dims")
	}

	// Use the mat.Matrix interface via mat.DenseCopyOf.
	denseCopy := mat.DenseCopyOf(sparse)
	if !mat.Equal(denseCopy, dense) {
		t.Errorf("DenseCopyOf(CSR) does not match original dense")
	}
}

func TestCSRTranspose(t *testing.T) {
	dense := mat.NewDense(2, 3, []float64{
		1, 0, 3,
		0, 5, 0,
	})
	m := CSRFromDense(dense)
	tr := m.T()

	r, c := tr.Dims()
	if r != 3 || c != 2 {
		t.Fatalf("T().Dims() = (%d, %d), want (3, 2)", r, c)
	}

	// Verify elements.
	for i := range 3 {
		for j := range 2 {
			if got, want := tr.At(i, j), dense.At(j, i); got != want {
				t.Errorf("T().At(%d, %d) = %f, want %f", i, j, got, want)
			}
		}
	}
}

func TestCSRRowView(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	indices := []int{0, 3, 2, 0, 2, 3}
	indptr := []int{0, 2, 3, 6}
	m := NewCSR(3, 4, data, indices, indptr)

	// Row 0: cols {0, 3}, vals {1, 2}
	idx, vals := m.RowView(0)
	if len(idx) != 2 || idx[0] != 0 || idx[1] != 3 {
		t.Errorf("RowView(0) indices = %v, want [0 3]", idx)
	}
	if len(vals) != 2 || vals[0] != 1 || vals[1] != 2 {
		t.Errorf("RowView(0) values = %v, want [1 2]", vals)
	}

	// Verify returned slices are copies.
	idx[0] = 999
	vals[0] = 999
	origIdx, origVals := m.RowView(0)
	if origIdx[0] != 0 || origVals[0] != 1 {
		t.Fatal("RowView returned slice aliased to internal data")
	}

	// Row 1: cols {2}, vals {3}
	idx, vals = m.RowView(1)
	if len(idx) != 1 || idx[0] != 2 || vals[0] != 3 {
		t.Errorf("RowView(1) = (%v, %v), want ([2], [3])", idx, vals)
	}

	// Row 2: cols {0, 2, 3}, vals {4, 5, 6}
	idx, vals = m.RowView(2)
	if len(idx) != 3 {
		t.Fatalf("RowView(2) length = %d, want 3", len(idx))
	}
	wantIdx := []int{0, 2, 3}
	wantVals := []float64{4, 5, 6}
	for k := range 3 {
		if idx[k] != wantIdx[k] || vals[k] != wantVals[k] {
			t.Errorf("RowView(2)[%d] = (%d, %f), want (%d, %f)", k, idx[k], vals[k], wantIdx[k], wantVals[k])
		}
	}
}

func TestCSRRowViewPanics(t *testing.T) {
	m := NewCSR(2, 3, nil, nil, []int{0, 0, 0})
	assertPanics(t, "negative row", func() { m.RowView(-1) })
	assertPanics(t, "row too large", func() { m.RowView(2) })
}

func TestCSRMulVec(t *testing.T) {
	// [1 0 0 2]   [1]   [1*1 + 2*4]   [9 ]
	// [0 0 3 0] * [2] = [3*3      ] = [9 ]
	// [4 0 5 6]   [3]   [4*1+5*3+6*4] [43]
	//             [4]
	data := []float64{1, 2, 3, 4, 5, 6}
	indices := []int{0, 3, 2, 0, 2, 3}
	indptr := []int{0, 2, 3, 6}
	m := NewCSR(3, 4, data, indices, indptr)

	x := mat.NewVecDense(4, []float64{1, 2, 3, 4})
	y := m.MulVec(x)

	want := []float64{9, 9, 43}
	for i, w := range want {
		if got := y.AtVec(i); !approxEqual(got, w, 1e-12) {
			t.Errorf("MulVec result[%d] = %f, want %f", i, got, w)
		}
	}
}

func TestCSRMulVecPanicsDimensionMismatch(t *testing.T) {
	m := NewCSR(2, 3, nil, nil, []int{0, 0, 0})
	x := mat.NewVecDense(4, []float64{1, 2, 3, 4})
	assertPanics(t, "dimension mismatch", func() { m.MulVec(x) })
}

func TestCSRMulVecAgainstDense(t *testing.T) {
	// Compare CSR MulVec against dense multiplication.
	dense := mat.NewDense(3, 3, []float64{
		2, 0, 1,
		0, 3, 0,
		7, 0, 4,
	})
	sparse := CSRFromDense(dense)

	x := mat.NewVecDense(3, []float64{1, 2, 3})

	// Sparse result.
	sparseResult := sparse.MulVec(x)

	// Dense result.
	denseResult := mat.NewVecDense(3, nil)
	denseResult.MulVec(dense, x)

	for i := range 3 {
		if got, want := sparseResult.AtVec(i), denseResult.AtVec(i); !approxEqual(got, want, 1e-12) {
			t.Errorf("MulVec result[%d]: sparse=%f, dense=%f", i, got, want)
		}
	}
}

func TestCSRDensity(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	indices := []int{0, 3, 2, 0, 2, 3}
	indptr := []int{0, 2, 3, 6}
	m := NewCSR(3, 4, data, indices, indptr)

	got := m.Density()
	want := 6.0 / 12.0 // 6 non-zeros in a 3x4 matrix
	if !approxEqual(got, want, 1e-12) {
		t.Errorf("Density() = %f, want %f", got, want)
	}
}

func TestCSRScale(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	indices := []int{0, 3, 2, 0, 2, 3}
	indptr := []int{0, 2, 3, 6}
	m := NewCSR(3, 4, data, indices, indptr)

	scaled := m.Scale(2.0)

	// Verify the original is unchanged.
	if m.Data[0] != 1 {
		t.Fatal("Scale modified the original matrix")
	}

	// Verify scaled values.
	r, c := scaled.Dims()
	if r != 3 || c != 4 {
		t.Fatalf("Scaled Dims() = (%d, %d), want (3, 4)", r, c)
	}

	want := mat.NewDense(3, 4, []float64{
		2, 0, 0, 4,
		0, 0, 6, 0,
		8, 0, 10, 12,
	})
	got := scaled.ToDense()
	if !mat.Equal(got, want) {
		t.Errorf("Scale(2) result:\ngot:  %v\nwant: %v", mat.Formatted(got), mat.Formatted(want))
	}
}

func TestCSRScaleZero(t *testing.T) {
	m := CSRFromDense(mat.NewDense(2, 2, []float64{1, 2, 3, 4}))
	scaled := m.Scale(0)

	// All values should be zero, but the structure is preserved.
	if scaled.NNZ() != 4 {
		t.Errorf("Scale(0) NNZ = %d, want 4 (structure preserved)", scaled.NNZ())
	}
	for _, v := range scaled.Data {
		if v != 0 {
			t.Errorf("Scale(0) has non-zero data value: %f", v)
		}
	}
}

// --- Edge Cases ---

func TestCSREmptyMatrix(t *testing.T) {
	m := NewCSR(0, 0, nil, nil, []int{0})
	r, c := m.Dims()
	if r != 0 || c != 0 {
		t.Errorf("Dims() = (%d, %d), want (0, 0)", r, c)
	}
	if m.NNZ() != 0 {
		t.Errorf("NNZ() = %d, want 0", m.NNZ())
	}
	if m.Density() != 0 {
		t.Errorf("Density() = %f, want 0", m.Density())
	}

	d := m.ToDense()
	if d != nil {
		t.Errorf("ToDense() for 0x0 matrix should return nil, got %v", d)
	}
}

func TestCSREmptyRowsMatrix(t *testing.T) {
	// A matrix with rows but no non-zero elements.
	m := NewCSR(3, 3, nil, nil, []int{0, 0, 0, 0})
	r, c := m.Dims()
	if r != 3 || c != 3 {
		t.Fatalf("Dims() = (%d, %d), want (3, 3)", r, c)
	}
	if m.NNZ() != 0 {
		t.Errorf("NNZ() = %d, want 0", m.NNZ())
	}

	// All elements should be zero.
	for i := range 3 {
		for j := range 3 {
			if v := m.At(i, j); v != 0 {
				t.Errorf("At(%d, %d) = %f, want 0", i, j, v)
			}
		}
	}

	// ToDense should work and produce a zero matrix.
	d := m.ToDense()
	for i := range 3 {
		for j := range 3 {
			if v := d.At(i, j); v != 0 {
				t.Errorf("ToDense().At(%d, %d) = %f, want 0", i, j, v)
			}
		}
	}
}

func TestCSRSingleElement(t *testing.T) {
	m := NewCSR(1, 1, []float64{42}, []int{0}, []int{0, 1})
	if m.At(0, 0) != 42 {
		t.Errorf("At(0,0) = %f, want 42", m.At(0, 0))
	}
	if m.NNZ() != 1 {
		t.Errorf("NNZ() = %d, want 1", m.NNZ())
	}
	if m.Density() != 1.0 {
		t.Errorf("Density() = %f, want 1.0", m.Density())
	}
}

func TestCSRFullDenseEquivalent(t *testing.T) {
	// A fully dense matrix stored as CSR.
	dense := mat.NewDense(2, 3, []float64{
		1, 2, 3,
		4, 5, 6,
	})
	m := CSRFromDense(dense)

	if m.NNZ() != 6 {
		t.Errorf("NNZ() = %d, want 6", m.NNZ())
	}
	if !approxEqual(m.Density(), 1.0, 1e-12) {
		t.Errorf("Density() = %f, want 1.0", m.Density())
	}

	got := m.ToDense()
	if !mat.Equal(got, dense) {
		t.Errorf("Roundtrip failed for fully dense matrix")
	}
}

func TestCSRAllZeroRows(t *testing.T) {
	// Matrix with rows that have no non-zero elements.
	// [0 0]
	// [1 0]
	// [0 0]
	m := NewCSR(3, 2, []float64{1}, []int{0}, []int{0, 0, 1, 1})

	idx, vals := m.RowView(0)
	if len(idx) != 0 || len(vals) != 0 {
		t.Errorf("RowView(0) should be empty, got %v, %v", idx, vals)
	}

	idx, vals = m.RowView(1)
	if len(idx) != 1 || idx[0] != 0 || vals[0] != 1 {
		t.Errorf("RowView(1) = (%v, %v), want ([0], [1])", idx, vals)
	}

	idx, vals = m.RowView(2)
	if len(idx) != 0 || len(vals) != 0 {
		t.Errorf("RowView(2) should be empty, got %v, %v", idx, vals)
	}
}

func TestCSRMulVecIdentity(t *testing.T) {
	// Identity matrix multiplication should return the input vector.
	// [1 0 0]
	// [0 1 0]
	// [0 0 1]
	m := NewCSR(3, 3,
		[]float64{1, 1, 1},
		[]int{0, 1, 2},
		[]int{0, 1, 2, 3},
	)

	x := mat.NewVecDense(3, []float64{7, 8, 9})
	y := m.MulVec(x)

	for i := range 3 {
		if got, want := y.AtVec(i), x.AtVec(i); got != want {
			t.Errorf("Identity MulVec[%d] = %f, want %f", i, got, want)
		}
	}
}

func TestCSRMulVecZeroMatrix(t *testing.T) {
	// A 2x3 matrix with no non-zero entries.
	m := NewCSR(2, 3, nil, nil, []int{0, 0, 0})
	x := mat.NewVecDense(3, []float64{1, 2, 3})
	y := m.MulVec(x)

	for i := range 2 {
		if got := y.AtVec(i); got != 0 {
			t.Errorf("Zero matrix MulVec[%d] = %f, want 0", i, got)
		}
	}
}

func TestNewCSRPanics(t *testing.T) {
	assertPanics(t, "negative rows", func() { NewCSR(-1, 2, nil, nil, []int{0}) })
	assertPanics(t, "negative cols", func() { NewCSR(2, -1, nil, nil, []int{0, 0, 0}) })
	assertPanics(t, "indptr length mismatch", func() { NewCSR(2, 2, nil, nil, []int{0, 0}) })
	assertPanics(t, "data/indices length mismatch", func() {
		NewCSR(1, 2, []float64{1, 2}, []int{0}, []int{0, 2})
	})
	assertPanics(t, "indptr last != data len", func() {
		NewCSR(1, 2, []float64{1}, []int{0}, []int{0, 5})
	})
	assertPanics(t, "column index out of range", func() {
		NewCSR(1, 2, []float64{1}, []int{5}, []int{0, 1})
	})
}

func TestCSRNegativeValues(t *testing.T) {
	dense := mat.NewDense(2, 2, []float64{
		-1, 0,
		0, -3.5,
	})
	m := CSRFromDense(dense)

	if m.At(0, 0) != -1 {
		t.Errorf("At(0,0) = %f, want -1", m.At(0, 0))
	}
	if m.At(1, 1) != -3.5 {
		t.Errorf("At(1,1) = %f, want -3.5", m.At(1, 1))
	}
}

func TestCSRScaleNegative(t *testing.T) {
	m := NewCSR(1, 3, []float64{1, 2, 3}, []int{0, 1, 2}, []int{0, 3})
	scaled := m.Scale(-1)

	want := []float64{-1, -2, -3}
	for i, w := range want {
		if got := scaled.At(0, i); got != w {
			t.Errorf("Scale(-1) At(0,%d) = %f, want %f", i, got, w)
		}
	}
}

// assertPanics verifies that fn panics.
func assertPanics(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("%s: expected panic, got none", name)
		}
	}()
	fn()
}
