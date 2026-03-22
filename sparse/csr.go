package sparse

import (
	"fmt"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// CSR is a Compressed Sparse Row matrix. It stores non-zero values in three
// arrays following the standard CSR convention:
//
//   - Data: non-zero values, row by row
//   - Indices: column index for each value in Data
//   - IndPtr: row pointers; IndPtr[i] is the start offset in Data/Indices for row i,
//     and IndPtr[rows] == len(Data)
//
// CSR implements gonum's mat.Matrix interface for interop with dense algorithms.
// It is immutable after construction — safe for concurrent reads.
type CSR struct {
	rows int
	cols int

	// Data holds the non-zero values in row-major order.
	Data []float64
	// Indices holds the column index for each corresponding entry in Data.
	Indices []int
	// IndPtr holds row pointers. Row i spans Data[IndPtr[i]:IndPtr[i+1]].
	IndPtr []int
}

// Compile-time check that CSR implements mat.Matrix.
var _ mat.Matrix = (*CSR)(nil)

// NewCSR creates a CSR matrix from raw CSR components.
//
// Parameters:
//   - r, c: number of rows and columns
//   - data: non-zero values in row-major order
//   - indices: column index for each entry in data (same length as data)
//   - indptr: row pointers of length r+1
//
// NewCSR copies the input slices so that the caller can safely modify them
// after construction.
func NewCSR(r, c int, data []float64, indices []int, indptr []int) *CSR {
	if r < 0 || c < 0 {
		panic("sparse: negative dimensions")
	}
	if len(indptr) != r+1 {
		panic(fmt.Sprintf("sparse: indptr length %d does not match rows+1 (%d)", len(indptr), r+1))
	}
	if len(data) != len(indices) {
		panic(fmt.Sprintf("sparse: data length %d does not match indices length %d", len(data), len(indices)))
	}
	if indptr[r] != len(data) {
		panic(fmt.Sprintf("sparse: indptr[%d]=%d does not match data length %d", r, indptr[r], len(data)))
	}

	// Validate column indices are in bounds.
	for i, idx := range indices {
		if idx < 0 || idx >= c {
			panic(fmt.Sprintf("sparse: column index %d at position %d out of range [0, %d)", idx, i, c))
		}
	}

	d := make([]float64, len(data))
	copy(d, data)
	idx := make([]int, len(indices))
	copy(idx, indices)
	ip := make([]int, len(indptr))
	copy(ip, indptr)

	return &CSR{
		rows:    r,
		cols:    c,
		Data:    d,
		Indices: idx,
		IndPtr:  ip,
	}
}

// CSRFromDense converts a gonum Dense matrix to CSR format.
// Zero-valued elements are not stored.
func CSRFromDense(d *mat.Dense) *CSR {
	r, c := d.Dims()

	var data []float64
	var indices []int
	indptr := make([]int, r+1)

	for i := range r {
		indptr[i] = len(data)
		for j := range c {
			v := d.At(i, j)
			if v != 0 {
				data = append(data, v)
				indices = append(indices, j)
			}
		}
	}
	indptr[r] = len(data)

	return &CSR{
		rows:    r,
		cols:    c,
		Data:    data,
		Indices: indices,
		IndPtr:  indptr,
	}
}

// Dims returns the number of rows and columns of the matrix.
func (m *CSR) Dims() (int, int) {
	return m.rows, m.cols
}

// At returns the element at row i, column j.
// It uses binary search within the row for O(log nnz_row) lookup.
func (m *CSR) At(i, j int) float64 {
	if i < 0 || i >= m.rows || j < 0 || j >= m.cols {
		panic(fmt.Sprintf("sparse: index (%d, %d) out of range for %dx%d matrix", i, j, m.rows, m.cols))
	}

	start := m.IndPtr[i]
	end := m.IndPtr[i+1]
	rowIndices := m.Indices[start:end]

	// Binary search for column j in this row's indices.
	pos := sort.SearchInts(rowIndices, j)
	if pos < len(rowIndices) && rowIndices[pos] == j {
		return m.Data[start+pos]
	}
	return 0
}

// T returns the transpose of the matrix as a mat.Matrix.
func (m *CSR) T() mat.Matrix {
	return mat.Transpose{Matrix: m}
}

// NNZ returns the number of stored (non-zero) elements.
func (m *CSR) NNZ() int {
	return len(m.Data)
}

// Density returns the fraction of non-zero elements (NNZ / total elements).
// Returns 0 for an empty (0x0) matrix.
func (m *CSR) Density() float64 {
	total := m.rows * m.cols
	if total == 0 {
		return 0
	}
	return float64(m.NNZ()) / float64(total)
}

// ToDense converts the CSR matrix to a gonum Dense matrix.
// For a 0x0 matrix it returns nil, since gonum does not support zero-dimension Dense matrices.
func (m *CSR) ToDense() *mat.Dense {
	if m.rows == 0 || m.cols == 0 {
		return nil
	}
	d := mat.NewDense(m.rows, m.cols, nil)
	for i := range m.rows {
		start := m.IndPtr[i]
		end := m.IndPtr[i+1]
		for k := start; k < end; k++ {
			d.Set(i, m.Indices[k], m.Data[k])
		}
	}
	return d
}

// RowView returns the column indices and values for the non-zero elements
// in row i. The returned slices are copies — modifying them does not affect
// the matrix.
func (m *CSR) RowView(i int) (indices []int, values []float64) {
	if i < 0 || i >= m.rows {
		panic(fmt.Sprintf("sparse: row index %d out of range [0, %d)", i, m.rows))
	}

	start := m.IndPtr[i]
	end := m.IndPtr[i+1]
	n := end - start

	indices = make([]int, n)
	copy(indices, m.Indices[start:end])
	values = make([]float64, n)
	copy(values, m.Data[start:end])
	return indices, values
}

// MulVec computes the sparse matrix-vector product y = A*x and returns y.
// The length of x must equal the number of columns.
func (m *CSR) MulVec(x *mat.VecDense) *mat.VecDense {
	n := x.Len()
	if n != m.cols {
		panic(fmt.Sprintf("sparse: vector length %d does not match matrix columns %d", n, m.cols))
	}

	result := mat.NewVecDense(m.rows, nil)
	xRaw := x.RawVector()

	for i := range m.rows {
		start := m.IndPtr[i]
		end := m.IndPtr[i+1]
		sum := 0.0
		for k := start; k < end; k++ {
			sum += m.Data[k] * xRaw.Data[m.Indices[k]*xRaw.Inc]
		}
		result.SetVec(i, sum)
	}
	return result
}

// Scale returns a new CSR matrix with every non-zero element multiplied by alpha.
func (m *CSR) Scale(alpha float64) *CSR {
	data := make([]float64, len(m.Data))
	for i, v := range m.Data {
		data[i] = alpha * v
	}

	indices := make([]int, len(m.Indices))
	copy(indices, m.Indices)
	indptr := make([]int, len(m.IndPtr))
	copy(indptr, m.IndPtr)

	return &CSR{
		rows:    m.rows,
		cols:    m.cols,
		Data:    data,
		Indices: indices,
		IndPtr:  indptr,
	}
}
