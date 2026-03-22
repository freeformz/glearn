package linear

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
	_ glearn.Estimator = ElasticNetConfig{}
	_ glearn.Predictor = (*ElasticNet)(nil)
	_ glearn.Scorer    = (*ElasticNet)(nil)
)

// ElasticNetConfig holds hyperparameters for elastic net regression (combined
// L1 and L2 regularization). It has Fit() but no Predict().
type ElasticNetConfig struct {
	// Alpha is the overall regularization strength. Default is 1.0.
	Alpha float64
	// L1Ratio controls the mix of L1 vs L2 penalty.
	// 0.0 = pure L2 (ridge), 1.0 = pure L1 (lasso). Default is 0.5.
	L1Ratio float64
	// FitIntercept determines whether to calculate the intercept.
	FitIntercept bool
	// MaxIter is the maximum number of coordinate descent iterations. Default is 1000.
	MaxIter int
	// Tolerance is the convergence tolerance. Default is 1e-4.
	Tolerance float64
}

// NewElasticNet creates an ElasticNetConfig with the given options.
// By default, Alpha is 1.0, L1Ratio is 0.5, FitIntercept is true,
// MaxIter is 1000, and Tolerance is 1e-4.
func NewElasticNet(opts ...ElasticNetOption) ElasticNetConfig {
	cfg := ElasticNetConfig{
		Alpha:        1.0,
		L1Ratio:      0.5,
		FitIntercept: true,
		MaxIter:      1000,
		Tolerance:    1e-4,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit trains an elastic net regression model using coordinate descent.
func (cfg ElasticNetConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.FitInputs(X, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/linear: elastic net fit failed: %w", err)
	}

	if cfg.Alpha < 0 {
		return nil, fmt.Errorf("glearn/linear: elastic net fit failed: %w: alpha must be non-negative, got %g",
			glearn.ErrInvalidParameter, cfg.Alpha)
	}
	if cfg.L1Ratio < 0 || cfg.L1Ratio > 1 {
		return nil, fmt.Errorf("glearn/linear: elastic net fit failed: %w: l1_ratio must be in [0, 1], got %g",
			glearn.ErrInvalidParameter, cfg.L1Ratio)
	}

	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("glearn/linear: elastic net fit cancelled: %w", ctx.Err())
	default:
	}

	// Center data if fitting intercept.
	var xMean []float64
	var yMeanVal float64
	xWork, yCentered := prepareData(X, y, nSamples, nFeatures, cfg.FitIntercept, &xMean, &yMeanVal)

	// Precompute column norms.
	colNorms := make([]float64, nFeatures)
	rawX := xWork.RawMatrix()
	for j := 0; j < nFeatures; j++ {
		sum := 0.0
		for i := 0; i < nSamples; i++ {
			v := rawX.Data[i*rawX.Stride+j]
			sum += v * v
		}
		colNorms[j] = sum
	}

	coef := make([]float64, nFeatures)
	residual := make([]float64, nSamples)
	copy(residual, yCentered)

	// Separate L1 and L2 penalty components, scaled by nSamples.
	l1Penalty := cfg.Alpha * cfg.L1Ratio * float64(nSamples)
	l2Penalty := cfg.Alpha * (1 - cfg.L1Ratio) * float64(nSamples)

	for iter := 0; iter < cfg.MaxIter; iter++ {
		if iter%100 == 0 {
			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("glearn/linear: elastic net fit cancelled at iteration %d: %w", iter, ctx.Err())
			default:
			}
		}

		maxChange := 0.0
		for j := 0; j < nFeatures; j++ {
			denom := colNorms[j] + l2Penalty
			if denom == 0 {
				continue
			}

			oldCoef := coef[j]

			// Compute rho = X_j' * residual + colNorm_j * coef_j
			rho := 0.0
			for i := 0; i < nSamples; i++ {
				rho += rawX.Data[i*rawX.Stride+j] * residual[i]
			}
			rho += colNorms[j] * oldCoef

			// Elastic net update: soft-threshold then divide by (colNorm + l2Penalty).
			coef[j] = softThreshold(rho, l1Penalty) / denom

			delta := coef[j] - oldCoef
			if delta != 0 {
				for i := 0; i < nSamples; i++ {
					residual[i] -= delta * rawX.Data[i*rawX.Stride+j]
				}
			}

			change := math.Abs(delta)
			if change > maxChange {
				maxChange = change
			}
		}

		if maxChange < cfg.Tolerance {
			break
		}
	}

	model := &ElasticNet{
		Coefficients: coef,
		NFeatures:    nFeatures,
	}

	if cfg.FitIntercept {
		intercept := yMeanVal
		for j := 0; j < nFeatures; j++ {
			intercept -= xMean[j] * coef[j]
		}
		model.Intercept = intercept
	}

	return model, nil
}

// ElasticNet is a fitted elastic net regression model.
// It has Predict() and Score() but no Fit().
type ElasticNet struct {
	// Coefficients are the learned feature weights.
	Coefficients []float64
	// Intercept is the learned bias term.
	Intercept float64
	// NFeatures is the number of features seen during fitting.
	NFeatures int
}

// Predict computes predictions for X using the fitted elastic net model.
func (en *ElasticNet) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, err := validate.PredictInputs(X, en.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/linear: elastic net predict failed: %w", err)
	}

	return linearPredict(X, en.Coefficients, en.Intercept, nSamples, en.NFeatures), nil
}

// Score returns the R-squared score of the model on the given data.
func (en *ElasticNet) Score(X *mat.Dense, y []float64) (float64, error) {
	preds, err := en.Predict(X)
	if err != nil {
		return 0, err
	}
	return r2Score(y, preds), nil
}

// GetCoefficients returns the learned feature weights.
func (en *ElasticNet) GetCoefficients() []float64 {
	out := make([]float64, len(en.Coefficients))
	copy(out, en.Coefficients)
	return out
}
