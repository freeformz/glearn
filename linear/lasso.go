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
	_ glearn.Estimator = LassoConfig{}
	_ glearn.Predictor = (*Lasso)(nil)
	_ glearn.Scorer    = (*Lasso)(nil)
)

// LassoConfig holds hyperparameters for L1-regularized linear regression.
// It has Fit() but no Predict().
type LassoConfig struct {
	// Alpha is the L1 regularization strength. Default is 1.0.
	Alpha float64
	// FitIntercept determines whether to calculate the intercept.
	FitIntercept bool
	// MaxIter is the maximum number of coordinate descent iterations. Default is 1000.
	MaxIter int
	// Tolerance is the convergence tolerance. Default is 1e-4.
	Tolerance float64
}

// NewLasso creates a LassoConfig with the given options.
// By default, Alpha is 1.0, FitIntercept is true, MaxIter is 1000, and
// Tolerance is 1e-4.
func NewLasso(opts ...LassoOption) LassoConfig {
	cfg := LassoConfig{
		Alpha:        1.0,
		FitIntercept: true,
		MaxIter:      1000,
		Tolerance:    1e-4,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit trains an L1-regularized regression model using coordinate descent
// with soft-thresholding.
func (cfg LassoConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.FitInputs(X, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/linear: lasso fit failed: %w", err)
	}

	if cfg.Alpha < 0 {
		return nil, fmt.Errorf("glearn/linear: lasso fit failed: %w: alpha must be non-negative, got %g",
			glearn.ErrInvalidParameter, cfg.Alpha)
	}

	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("glearn/linear: lasso fit cancelled: %w", ctx.Err())
	default:
	}

	// Center data if fitting intercept.
	var xMean []float64
	var yMeanVal float64
	xWork, yCentered := prepareData(X, y, nSamples, nFeatures, cfg.FitIntercept, &xMean, &yMeanVal)

	// Precompute column norms (sum of squares) for coordinate descent.
	colNorms := make([]float64, nFeatures)
	rawX := xWork.RawMatrix()
	for j := range nFeatures {
		sum := 0.0
		for i := range nSamples {
			v := rawX.Data[i*rawX.Stride+j]
			sum += v * v
		}
		colNorms[j] = sum
	}

	// Initialize coefficients to zero.
	coef := make([]float64, nFeatures)

	// Compute initial residuals: r = y - X*coef (coef is zero, so r = y).
	residual := make([]float64, nSamples)
	copy(residual, yCentered)

	// Regularization parameter scaled by nSamples (match sklearn convention).
	alpha := cfg.Alpha * float64(nSamples)

	// Coordinate descent.
	for iter := 0; iter < cfg.MaxIter; iter++ {
		// Check context cancellation periodically.
		if iter%100 == 0 {
			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("glearn/linear: lasso fit cancelled at iteration %d: %w", iter, ctx.Err())
			default:
			}
		}

		maxChange := 0.0
		for j := range nFeatures {
			if colNorms[j] == 0 {
				continue
			}

			oldCoef := coef[j]

			// Compute partial residual dot product: sum_i x_ij * r_i + colNorm_j * coef_j
			rho := 0.0
			for i := range nSamples {
				rho += rawX.Data[i*rawX.Stride+j] * residual[i]
			}
			rho += colNorms[j] * oldCoef

			// Soft-thresholding.
			coef[j] = softThreshold(rho, alpha) / colNorms[j]

			// Update residuals.
			delta := coef[j] - oldCoef
			if delta != 0 {
				for i := range nSamples {
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

	model := &Lasso{
		Coefficients: coef,
		NFeatures:    nFeatures,
	}

	if cfg.FitIntercept {
		intercept := yMeanVal
		for j := range nFeatures {
			intercept -= xMean[j] * coef[j]
		}
		model.Intercept = intercept
	}

	return model, nil
}

// Lasso is a fitted L1-regularized regression model.
// It has Predict() and Score() but no Fit().
type Lasso struct {
	// Coefficients are the learned feature weights.
	Coefficients []float64
	// Intercept is the learned bias term.
	Intercept float64
	// NFeatures is the number of features seen during fitting.
	NFeatures int
}

// Predict computes predictions for X using the fitted lasso model.
func (l *Lasso) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, err := validate.PredictInputs(X, l.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/linear: lasso predict failed: %w", err)
	}

	return linearPredict(X, l.Coefficients, l.Intercept, nSamples, l.NFeatures), nil
}

// Score returns the R-squared score of the model on the given data.
func (l *Lasso) Score(X *mat.Dense, y []float64) (float64, error) {
	preds, err := l.Predict(X)
	if err != nil {
		return 0, err
	}
	return r2Score(y, preds), nil
}

// GetCoefficients returns the learned feature weights.
func (l *Lasso) GetCoefficients() []float64 {
	out := make([]float64, len(l.Coefficients))
	copy(out, l.Coefficients)
	return out
}

// softThreshold applies the soft-thresholding operator.
// S(x, lambda) = sign(x) * max(|x| - lambda, 0)
func softThreshold(x, lambda float64) float64 {
	if x > lambda {
		return x - lambda
	}
	if x < -lambda {
		return x + lambda
	}
	return 0
}

// prepareData centers X and y if fitIntercept is true, otherwise copies them.
func prepareData(X *mat.Dense, y []float64, nSamples, nFeatures int, fitIntercept bool, xMeanOut *[]float64, yMeanOut *float64) (*mat.Dense, []float64) {
	if fitIntercept {
		xMean := make([]float64, nFeatures)
		raw := X.RawMatrix()
		for j := range nFeatures {
			sum := 0.0
			for i := range nSamples {
				sum += raw.Data[i*raw.Stride+j]
			}
			xMean[j] = sum / float64(nSamples)
		}

		yMean := 0.0
		for _, v := range y {
			yMean += v
		}
		yMean /= float64(nSamples)

		// Center X.
		xData := make([]float64, nSamples*nFeatures)
		for i := range nSamples {
			for j := range nFeatures {
				xData[i*nFeatures+j] = raw.Data[i*raw.Stride+j] - xMean[j]
			}
		}
		xWork := mat.NewDense(nSamples, nFeatures, xData)

		// Center y.
		yCentered := make([]float64, nSamples)
		for i := range y {
			yCentered[i] = y[i] - yMean
		}

		*xMeanOut = xMean
		*yMeanOut = yMean
		return xWork, yCentered
	}

	xWork := mat.NewDense(nSamples, nFeatures, nil)
	xWork.Copy(X)
	yCentered := make([]float64, nSamples)
	copy(yCentered, y)
	*xMeanOut = nil
	*yMeanOut = 0
	return xWork, yCentered
}
