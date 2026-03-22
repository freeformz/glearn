package preprocessing

import (
	"context"
	"fmt"
	"math"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/internal/validate"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface checks.
var (
	_ glearn.Transformer      = StandardScalerConfig{}
	_ glearn.FitTransformer   = StandardScalerConfig{}
	_ glearn.FittedTransformer = (*StandardScaler)(nil)

	_ glearn.Transformer      = MinMaxScalerConfig{}
	_ glearn.FitTransformer   = MinMaxScalerConfig{}
	_ glearn.FittedTransformer = (*MinMaxScaler)(nil)
)

// StandardScalerConfig configures a StandardScaler that standardizes features
// by removing the mean and scaling to unit variance.
//
// Each feature is transformed as: z = (x - mean) / std
type StandardScalerConfig struct {
	// WithMean centers the data to zero mean when true. Default is true.
	WithMean bool
	// WithStd scales the data to unit variance when true. Default is true.
	WithStd bool
}

// NewStandardScaler creates a StandardScalerConfig with functional options.
// By default, both centering (WithMean) and scaling (WithStd) are enabled.
func NewStandardScaler(opts ...StandardScalerOption) StandardScalerConfig {
	cfg := StandardScalerConfig{
		WithMean: true,
		WithStd:  true,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit computes the mean and standard deviation from X and returns a fitted
// StandardScaler. The input matrix X is not modified.
func (cfg StandardScalerConfig) Fit(ctx context.Context, X *mat.Dense) (glearn.FittedTransformer, error) {
	rows, cols, err := validate.Dimensions(X)
	if err != nil {
		return nil, fmt.Errorf("glearn/preprocessing: standard scaler fit: %w", err)
	}

	s := &StandardScaler{
		NFeatures: cols,
		WithMean:  cfg.WithMean,
		WithStd:   cfg.WithStd,
	}

	if cfg.WithMean {
		s.Mean = columnMeans(X, rows, cols)
	}
	if cfg.WithStd {
		s.Scale = columnStds(X, rows, cols, s.Mean, cfg.WithMean)
	}

	return s, nil
}

// FitTransform fits the scaler and transforms X in a single step.
func (cfg StandardScalerConfig) FitTransform(ctx context.Context, X *mat.Dense) (glearn.FittedTransformer, *mat.Dense, error) {
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

// StandardScaler is a fitted standard scaler. It transforms data by subtracting
// the mean and dividing by the standard deviation learned during Fit.
//
// StandardScaler is immutable after construction and safe for concurrent Transform calls.
type StandardScaler struct {
	// Mean per feature, computed during Fit. Nil if WithMean is false.
	Mean []float64
	// Scale (standard deviation) per feature, computed during Fit. Nil if WithStd is false.
	Scale []float64
	// NFeatures is the number of features seen during Fit.
	NFeatures int
	// WithMean indicates whether centering was applied.
	WithMean bool
	// WithStd indicates whether scaling was applied.
	WithStd bool
}

// Transform standardizes X using the mean and scale learned during Fit.
// Returns a new matrix; X is not modified.
func (s *StandardScaler) Transform(X *mat.Dense) (*mat.Dense, error) {
	nSamples, err := validate.PredictInputs(X, s.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/preprocessing: standard scaler transform: %w", err)
	}

	result := mat.NewDense(nSamples, s.NFeatures, nil)
	for i := range nSamples {
		for j := range s.NFeatures {
			v := X.At(i, j)
			if s.WithMean {
				v -= s.Mean[j]
			}
			if s.WithStd {
				v /= s.Scale[j]
			}
			result.Set(i, j, v)
		}
	}
	return result, nil
}

// InverseTransform reverses the standardization, recovering the original scale.
// Returns a new matrix; X is not modified.
func (s *StandardScaler) InverseTransform(X *mat.Dense) (*mat.Dense, error) {
	nSamples, err := validate.PredictInputs(X, s.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/preprocessing: standard scaler inverse transform: %w", err)
	}

	result := mat.NewDense(nSamples, s.NFeatures, nil)
	for i := range nSamples {
		for j := range s.NFeatures {
			v := X.At(i, j)
			if s.WithStd {
				v *= s.Scale[j]
			}
			if s.WithMean {
				v += s.Mean[j]
			}
			result.Set(i, j, v)
		}
	}
	return result, nil
}

// MinMaxScalerConfig configures a MinMaxScaler that scales features to a given range.
//
// Each feature is transformed as:
//
//	x_scaled = (x - x_min) / (x_max - x_min) * (feature_max - feature_min) + feature_min
type MinMaxScalerConfig struct {
	// FeatureMin is the lower bound of the target range. Default is 0.
	FeatureMin float64
	// FeatureMax is the upper bound of the target range. Default is 1.
	FeatureMax float64
}

// NewMinMaxScaler creates a MinMaxScalerConfig with functional options.
// By default, the target range is [0, 1].
func NewMinMaxScaler(opts ...MinMaxScalerOption) MinMaxScalerConfig {
	cfg := MinMaxScalerConfig{
		FeatureMin: 0,
		FeatureMax: 1,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit computes the per-feature minimum and maximum from X and returns a fitted
// MinMaxScaler. The input matrix X is not modified.
func (cfg MinMaxScalerConfig) Fit(ctx context.Context, X *mat.Dense) (glearn.FittedTransformer, error) {
	if cfg.FeatureMin >= cfg.FeatureMax {
		return nil, fmt.Errorf("glearn/preprocessing: min-max scaler fit: %w: FeatureMin (%g) must be less than FeatureMax (%g)",
			glearn.ErrInvalidParameter, cfg.FeatureMin, cfg.FeatureMax)
	}

	rows, cols, err := validate.Dimensions(X)
	if err != nil {
		return nil, fmt.Errorf("glearn/preprocessing: min-max scaler fit: %w", err)
	}

	dataMin := make([]float64, cols)
	dataMax := make([]float64, cols)

	// Initialize with first row.
	for j := range cols {
		dataMin[j] = X.At(0, j)
		dataMax[j] = X.At(0, j)
	}
	for i := 1; i < rows; i++ {
		for j := range cols {
			v := X.At(i, j)
			if v < dataMin[j] {
				dataMin[j] = v
			}
			if v > dataMax[j] {
				dataMax[j] = v
			}
		}
	}

	// Compute scale and offset for each feature.
	scale := make([]float64, cols)
	offset := make([]float64, cols)
	featureRange := cfg.FeatureMax - cfg.FeatureMin

	for j := range cols {
		dataRange := dataMax[j] - dataMin[j]
		if dataRange == 0 {
			// Constant feature: scale to FeatureMin.
			scale[j] = 0
			offset[j] = cfg.FeatureMin
		} else {
			scale[j] = featureRange / dataRange
			offset[j] = cfg.FeatureMin - dataMin[j]*scale[j]
		}
	}

	return &MinMaxScaler{
		DataMin:    dataMin,
		DataMax:    dataMax,
		Scale:      scale,
		Offset:     offset,
		FeatureMin: cfg.FeatureMin,
		FeatureMax: cfg.FeatureMax,
		NFeatures:  cols,
	}, nil
}

// FitTransform fits the scaler and transforms X in a single step.
func (cfg MinMaxScalerConfig) FitTransform(ctx context.Context, X *mat.Dense) (glearn.FittedTransformer, *mat.Dense, error) {
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

// MinMaxScaler is a fitted min-max scaler. It transforms data by scaling each
// feature to the target range learned during Fit.
//
// MinMaxScaler is immutable after construction and safe for concurrent Transform calls.
type MinMaxScaler struct {
	// DataMin is the per-feature minimum observed during Fit.
	DataMin []float64
	// DataMax is the per-feature maximum observed during Fit.
	DataMax []float64
	// Scale is the per-feature scaling factor: (FeatureMax - FeatureMin) / (DataMax - DataMin).
	Scale []float64
	// Offset is the per-feature offset: FeatureMin - DataMin * Scale.
	Offset []float64
	// FeatureMin is the lower bound of the target range.
	FeatureMin float64
	// FeatureMax is the upper bound of the target range.
	FeatureMax float64
	// NFeatures is the number of features seen during Fit.
	NFeatures int
}

// Transform scales X to the target range using parameters learned during Fit.
// Returns a new matrix; X is not modified.
func (s *MinMaxScaler) Transform(X *mat.Dense) (*mat.Dense, error) {
	nSamples, err := validate.PredictInputs(X, s.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/preprocessing: min-max scaler transform: %w", err)
	}

	result := mat.NewDense(nSamples, s.NFeatures, nil)
	for i := range nSamples {
		for j := range s.NFeatures {
			result.Set(i, j, X.At(i, j)*s.Scale[j]+s.Offset[j])
		}
	}
	return result, nil
}

// InverseTransform reverses the min-max scaling, recovering the original scale.
// Returns a new matrix; X is not modified.
func (s *MinMaxScaler) InverseTransform(X *mat.Dense) (*mat.Dense, error) {
	nSamples, err := validate.PredictInputs(X, s.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/preprocessing: min-max scaler inverse transform: %w", err)
	}

	result := mat.NewDense(nSamples, s.NFeatures, nil)
	for i := range nSamples {
		for j := range s.NFeatures {
			v := X.At(i, j)
			if s.Scale[j] == 0 {
				// Constant feature: inverse is the original constant value.
				result.Set(i, j, s.DataMin[j])
			} else {
				result.Set(i, j, (v-s.Offset[j])/s.Scale[j])
			}
		}
	}
	return result, nil
}

// columnMeans computes the mean of each column in X.
func columnMeans(X *mat.Dense, rows, cols int) []float64 {
	means := make([]float64, cols)
	for j := range cols {
		var sum float64
		for i := range rows {
			sum += X.At(i, j)
		}
		means[j] = sum / float64(rows)
	}
	return means
}

// columnStds computes the population standard deviation of each column in X.
// If withMean is true, uses the provided means; otherwise computes means internally.
// Zero-variance features get a scale of 1.0 to avoid division by zero.
func columnStds(X *mat.Dense, rows, cols int, means []float64, withMean bool) []float64 {
	if !withMean {
		means = columnMeans(X, rows, cols)
	}
	stds := make([]float64, cols)
	for j := range cols {
		var ss float64
		for i := range rows {
			d := X.At(i, j) - means[j]
			ss += d * d
		}
		std := math.Sqrt(ss / float64(rows))
		if std == 0 {
			std = 1.0 // Avoid division by zero for constant features.
		}
		stds[j] = std
	}
	return stds
}
