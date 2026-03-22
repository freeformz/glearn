package preprocessing

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/internal/validate"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface checks.
var (
	_ glearn.Transformer       = OneHotEncoderConfig{}
	_ glearn.FitTransformer    = OneHotEncoderConfig{}
	_ glearn.FittedTransformer = (*OneHotEncoder)(nil)

	_ glearn.Transformer       = LabelEncoderConfig{}
	_ glearn.FitTransformer    = LabelEncoderConfig{}
	_ glearn.FittedTransformer = (*LabelEncoder)(nil)
)

// OneHotEncoderConfig configures a OneHotEncoder that encodes categorical
// integer features as one-hot numeric arrays.
//
// Input features are expected to be non-negative integers represented as float64.
// Each feature column is expanded into multiple binary columns, one per unique category.
type OneHotEncoderConfig struct {
	// DropFirst drops the first category per feature to avoid multicollinearity.
	DropFirst bool
}

// NewOneHotEncoder creates a OneHotEncoderConfig with functional options.
func NewOneHotEncoder(opts ...OneHotEncoderOption) OneHotEncoderConfig {
	cfg := OneHotEncoderConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit learns the categories for each feature from X and returns a fitted
// OneHotEncoder. Values in X must be non-negative integers (as float64).
func (cfg OneHotEncoderConfig) Fit(ctx context.Context, X *mat.Dense) (glearn.FittedTransformer, error) {
	rows, cols, err := validate.Dimensions(X)
	if err != nil {
		return nil, fmt.Errorf("glearn/preprocessing: one-hot encoder fit: %w", err)
	}

	categories := make([][]float64, cols)
	for j := range cols {
		seen := make(map[float64]struct{})
		for i := range rows {
			v := X.At(i, j)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, fmt.Errorf("glearn/preprocessing: one-hot encoder fit: %w: feature %d contains NaN or Inf",
					glearn.ErrInvalidParameter, j)
			}
			if v != math.Floor(v) || v < 0 {
				return nil, fmt.Errorf("glearn/preprocessing: one-hot encoder fit: %w: feature %d contains non-integer or negative value %g",
					glearn.ErrInvalidParameter, j, v)
			}
			seen[v] = struct{}{}
		}
		cats := make([]float64, 0, len(seen))
		for v := range seen {
			cats = append(cats, v)
		}
		sort.Float64s(cats)
		categories[j] = cats
	}

	return &OneHotEncoder{
		Categories:     categories,
		DropFirst:      cfg.DropFirst,
		NInputFeatures: cols,
	}, nil
}

// FitTransform fits the encoder and transforms X in a single step.
func (cfg OneHotEncoderConfig) FitTransform(ctx context.Context, X *mat.Dense) (glearn.FittedTransformer, *mat.Dense, error) {
	ft, err := cfg.Fit(ctx, X)
	if err != nil {
		return nil, nil, err
	}
	result, err := ft.Transform(X)
	if err != nil {
		return nil, nil, err
	}
	return ft, result, nil
}

// OneHotEncoder is a fitted one-hot encoder. It transforms categorical integer
// features into binary one-hot columns.
//
// OneHotEncoder is immutable after construction and safe for concurrent Transform calls.
type OneHotEncoder struct {
	// Categories contains the sorted unique values for each input feature.
	Categories [][]float64
	// DropFirst indicates whether the first category per feature is dropped.
	DropFirst bool
	// NInputFeatures is the number of input features seen during Fit.
	NInputFeatures int
}

// NOutputFeatures returns the total number of output columns after one-hot encoding.
func (e *OneHotEncoder) NOutputFeatures() int {
	total := 0
	for _, cats := range e.Categories {
		n := len(cats)
		if e.DropFirst && n > 0 {
			n--
		}
		total += n
	}
	return total
}

// Transform encodes X into one-hot representation using categories learned during Fit.
// Returns a new matrix; X is not modified. Unknown categories produce an error.
func (e *OneHotEncoder) Transform(X *mat.Dense) (*mat.Dense, error) {
	nSamples, err := validate.PredictInputs(X, e.NInputFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/preprocessing: one-hot encoder transform: %w", err)
	}

	nOut := e.NOutputFeatures()
	result := mat.NewDense(nSamples, nOut, nil)

	for i := range nSamples {
		col := 0
		for j := range e.NInputFeatures {
			v := X.At(i, j)
			catIdx := searchFloat64s(e.Categories[j], v)
			if catIdx < 0 {
				return nil, fmt.Errorf("glearn/preprocessing: one-hot encoder transform: %w: unknown category %g in feature %d",
					glearn.ErrInvalidParameter, v, j)
			}
			nCats := len(e.Categories[j])
			startCat := 0
			if e.DropFirst {
				startCat = 1
				nCats--
			}
			if catIdx >= startCat {
				result.Set(i, col+catIdx-startCat, 1)
			}
			col += nCats
		}
	}
	return result, nil
}

// LabelEncoderConfig configures a LabelEncoder that encodes target labels
// as consecutive integers starting from 0.
//
// LabelEncoder operates on 1-D label data represented as a single-column matrix.
type LabelEncoderConfig struct{}

// NewLabelEncoder creates a LabelEncoderConfig with functional options.
func NewLabelEncoder(opts ...LabelEncoderOption) LabelEncoderConfig {
	cfg := LabelEncoderConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit learns the mapping from labels to integers from X.
// X must be a single-column matrix containing the labels.
func (cfg LabelEncoderConfig) Fit(ctx context.Context, X *mat.Dense) (glearn.FittedTransformer, error) {
	rows, cols, err := validate.Dimensions(X)
	if err != nil {
		return nil, fmt.Errorf("glearn/preprocessing: label encoder fit: %w", err)
	}
	if cols != 1 {
		return nil, fmt.Errorf("glearn/preprocessing: label encoder fit: %w: expected 1 column, got %d",
			glearn.ErrDimensionMismatch, cols)
	}

	seen := make(map[float64]struct{})
	for i := range rows {
		seen[X.At(i, 0)] = struct{}{}
	}

	classes := make([]float64, 0, len(seen))
	for v := range seen {
		classes = append(classes, v)
	}
	sort.Float64s(classes)

	classToIndex := make(map[float64]int, len(classes))
	for i, c := range classes {
		classToIndex[c] = i
	}

	return &LabelEncoder{
		Classes:      classes,
		ClassToIndex: classToIndex,
	}, nil
}

// FitTransform fits the encoder and transforms X in a single step.
func (cfg LabelEncoderConfig) FitTransform(ctx context.Context, X *mat.Dense) (glearn.FittedTransformer, *mat.Dense, error) {
	ft, err := cfg.Fit(ctx, X)
	if err != nil {
		return nil, nil, err
	}
	result, err := ft.Transform(X)
	if err != nil {
		return nil, nil, err
	}
	return ft, result, nil
}

// LabelEncoder is a fitted label encoder. It transforms label values into
// consecutive integers [0, n_classes).
//
// LabelEncoder is immutable after construction and safe for concurrent Transform calls.
type LabelEncoder struct {
	// Classes contains the sorted unique labels seen during Fit.
	Classes []float64
	// ClassToIndex maps each label value to its integer encoding.
	ClassToIndex map[float64]int
}

// Transform encodes labels in X as integers using the mapping learned during Fit.
// X must be a single-column matrix. Returns a new single-column matrix.
// Unknown labels produce an error.
func (e *LabelEncoder) Transform(X *mat.Dense) (*mat.Dense, error) {
	rows, cols := X.Dims()
	if rows == 0 || cols == 0 {
		return nil, fmt.Errorf("glearn/preprocessing: label encoder transform: %w: X has dimensions %dx%d",
			glearn.ErrEmptyInput, rows, cols)
	}
	if cols != 1 {
		return nil, fmt.Errorf("glearn/preprocessing: label encoder transform: %w: expected 1 column, got %d",
			glearn.ErrDimensionMismatch, cols)
	}

	result := mat.NewDense(rows, 1, nil)
	for i := range rows {
		v := X.At(i, 0)
		idx, ok := e.ClassToIndex[v]
		if !ok {
			return nil, fmt.Errorf("glearn/preprocessing: label encoder transform: %w: unknown label %g",
				glearn.ErrInvalidParameter, v)
		}
		result.Set(i, 0, float64(idx))
	}
	return result, nil
}

// InverseTransform converts integer-encoded labels back to the original label values.
// X must be a single-column matrix containing integer indices. Returns a new single-column matrix.
func (e *LabelEncoder) InverseTransform(X *mat.Dense) (*mat.Dense, error) {
	rows, cols := X.Dims()
	if rows == 0 || cols == 0 {
		return nil, fmt.Errorf("glearn/preprocessing: label encoder inverse transform: %w: X has dimensions %dx%d",
			glearn.ErrEmptyInput, rows, cols)
	}
	if cols != 1 {
		return nil, fmt.Errorf("glearn/preprocessing: label encoder inverse transform: %w: expected 1 column, got %d",
			glearn.ErrDimensionMismatch, cols)
	}

	result := mat.NewDense(rows, 1, nil)
	for i := range rows {
		idx := int(X.At(i, 0))
		if idx < 0 || idx >= len(e.Classes) {
			return nil, fmt.Errorf("glearn/preprocessing: label encoder inverse transform: %w: index %d out of range [0, %d)",
				glearn.ErrInvalidParameter, idx, len(e.Classes))
		}
		result.Set(i, 0, e.Classes[idx])
	}
	return result, nil
}

// searchFloat64s returns the index of v in the sorted slice, or -1 if not found.
func searchFloat64s(sorted []float64, v float64) int {
	i := sort.SearchFloat64s(sorted, v)
	if i < len(sorted) && sorted[i] == v {
		return i
	}
	return -1
}
