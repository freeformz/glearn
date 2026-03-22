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
	_ glearn.Transformer       = SimpleImputerConfig{}
	_ glearn.FitTransformer    = SimpleImputerConfig{}
	_ glearn.FittedTransformer = (*SimpleImputer)(nil)
)

// SimpleImputerConfig configures a SimpleImputer that replaces missing values (NaN)
// with a computed statistic (mean, median) or a constant.
type SimpleImputerConfig struct {
	// Strategy determines how missing values are replaced.
	// Default is StrategyMean.
	Strategy Strategy
	// FillValue is the replacement value when Strategy is StrategyConstant.
	FillValue float64
}

// NewSimpleImputer creates a SimpleImputerConfig with functional options.
// The default strategy is StrategyMean.
func NewSimpleImputer(opts ...SimpleImputerOption) SimpleImputerConfig {
	cfg := SimpleImputerConfig{
		Strategy: StrategyMean,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit computes the fill values from X and returns a fitted SimpleImputer.
// NaN values in X are treated as missing and excluded from statistic computation.
func (cfg SimpleImputerConfig) Fit(ctx context.Context, X *mat.Dense) (glearn.FittedTransformer, error) {
	rows, cols, err := validate.Dimensions(X)
	if err != nil {
		return nil, fmt.Errorf("glearn/preprocessing: simple imputer fit: %w", err)
	}

	statistics := make([]float64, cols)

	switch cfg.Strategy {
	case StrategyMean:
		for j := range cols {
			var sum float64
			var count int
			for i := range rows {
				v := X.At(i, j)
				if !math.IsNaN(v) {
					sum += v
					count++
				}
			}
			if count == 0 {
				statistics[j] = math.NaN()
			} else {
				statistics[j] = sum / float64(count)
			}
		}
	case StrategyMedian:
		for j := range cols {
			var values []float64
			for i := range rows {
				v := X.At(i, j)
				if !math.IsNaN(v) {
					values = append(values, v)
				}
			}
			if len(values) == 0 {
				statistics[j] = math.NaN()
			} else {
				sort.Float64s(values)
				n := len(values)
				if n%2 == 0 {
					statistics[j] = (values[n/2-1] + values[n/2]) / 2
				} else {
					statistics[j] = values[n/2]
				}
			}
		}
	case StrategyConstant:
		for j := range cols {
			statistics[j] = cfg.FillValue
		}
	default:
		return nil, fmt.Errorf("glearn/preprocessing: simple imputer fit: %w: unknown strategy %d",
			glearn.ErrInvalidParameter, cfg.Strategy)
	}

	return &SimpleImputer{
		Statistics: statistics,
		Strategy:   cfg.Strategy,
		NFeatures:  cols,
	}, nil
}

// FitTransform fits the imputer and transforms X in a single step.
func (cfg SimpleImputerConfig) FitTransform(ctx context.Context, X *mat.Dense) (glearn.FittedTransformer, *mat.Dense, error) {
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

// SimpleImputer is a fitted imputer that replaces NaN values with statistics
// learned during Fit.
//
// SimpleImputer is immutable after construction and safe for concurrent Transform calls.
type SimpleImputer struct {
	// Statistics contains the fill value for each feature, computed during Fit.
	Statistics []float64
	// Strategy is the imputation strategy used.
	Strategy Strategy
	// NFeatures is the number of features seen during Fit.
	NFeatures int
}

// Transform replaces NaN values in X with the statistics learned during Fit.
// Returns a new matrix; X is not modified.
func (imp *SimpleImputer) Transform(X *mat.Dense) (*mat.Dense, error) {
	nSamples, err := validate.PredictInputs(X, imp.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/preprocessing: simple imputer transform: %w", err)
	}

	result := mat.NewDense(nSamples, imp.NFeatures, nil)
	for i := range nSamples {
		for j := range imp.NFeatures {
			v := X.At(i, j)
			if math.IsNaN(v) {
				v = imp.Statistics[j]
			}
			result.Set(i, j, v)
		}
	}
	return result, nil
}
