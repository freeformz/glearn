package linear

import (
	"context"
	"fmt"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/internal/validate"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface checks.
var (
	_ glearn.Estimator = RidgeConfig{}
	_ glearn.Predictor = (*Ridge)(nil)
	_ glearn.Scorer    = (*Ridge)(nil)
)

// RidgeConfig holds hyperparameters for L2-regularized linear regression.
// It has Fit() but no Predict().
type RidgeConfig struct {
	// Alpha is the L2 regularization strength. Default is 1.0.
	Alpha float64
	// FitIntercept determines whether to calculate the intercept for the model.
	FitIntercept bool
}

// NewRidge creates a RidgeConfig with the given options.
// By default, Alpha is 1.0 and FitIntercept is true.
func NewRidge(opts ...RidgeOption) RidgeConfig {
	cfg := RidgeConfig{
		Alpha:        1.0,
		FitIntercept: true,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit trains an L2-regularized regression model using the closed-form solution
// (X'X + alpha*I)^{-1} X'y via Cholesky decomposition.
func (cfg RidgeConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.FitInputs(X, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/linear: ridge fit failed: %w", err)
	}

	if cfg.Alpha < 0 {
		return nil, fmt.Errorf("glearn/linear: ridge fit failed: %w: alpha must be non-negative, got %g",
			glearn.ErrInvalidParameter, cfg.Alpha)
	}

	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("glearn/linear: ridge fit cancelled: %w", ctx.Err())
	default:
	}

	// Center data if fitting intercept.
	var xMean []float64
	var yMeanVal float64
	var xWork *mat.Dense
	var yCentered []float64

	if cfg.FitIntercept {
		xMean = make([]float64, nFeatures)
		for j := range nFeatures {
			sum := 0.0
			for i := range nSamples {
				sum += X.At(i, j)
			}
			xMean[j] = sum / float64(nSamples)
		}

		yMeanVal = 0.0
		for _, v := range y {
			yMeanVal += v
		}
		yMeanVal /= float64(nSamples)

		// Center X.
		xData := make([]float64, nSamples*nFeatures)
		raw := X.RawMatrix()
		for i := range nSamples {
			for j := range nFeatures {
				xData[i*nFeatures+j] = raw.Data[i*raw.Stride+j] - xMean[j]
			}
		}
		xWork = mat.NewDense(nSamples, nFeatures, xData)

		// Center y.
		yCentered = make([]float64, nSamples)
		for i := range y {
			yCentered[i] = y[i] - yMeanVal
		}
	} else {
		xWork = mat.NewDense(nSamples, nFeatures, nil)
		xWork.Copy(X)
		yCentered = make([]float64, nSamples)
		copy(yCentered, y)
	}

	// Compute X'X as a dense matrix first, then build a SymDense.
	var xtxDense mat.Dense
	xtxDense.Mul(xWork.T(), xWork)

	// Build SymDense from X'X + alpha*I.
	xtxData := make([]float64, nFeatures*nFeatures)
	for i := range nFeatures {
		for j := range nFeatures {
			xtxData[i*nFeatures+j] = xtxDense.At(i, j)
		}
		xtxData[i*nFeatures+i] += cfg.Alpha
	}
	xtx := mat.NewSymDense(nFeatures, xtxData)

	// Compute X'y.
	yVec := mat.NewVecDense(nSamples, yCentered)
	var xty mat.VecDense
	xty.MulVec(xWork.T(), yVec)

	// Solve via Cholesky: (X'X + alpha*I) * beta = X'y.
	var chol mat.Cholesky
	if ok := chol.Factorize(xtx); !ok {
		return nil, fmt.Errorf("glearn/linear: ridge Cholesky factorization failed: %w", glearn.ErrSingularMatrix)
	}

	beta := mat.NewVecDense(nFeatures, nil)
	if err := chol.SolveVecTo(beta, &xty); err != nil {
		return nil, fmt.Errorf("glearn/linear: ridge solve failed: %w", glearn.ErrSingularMatrix)
	}

	model := &Ridge{
		Coefficients: make([]float64, nFeatures),
		NFeatures:    nFeatures,
	}
	for i := range nFeatures {
		model.Coefficients[i] = beta.AtVec(i)
	}

	if cfg.FitIntercept {
		// intercept = yMean - xMean . coef
		intercept := yMeanVal
		for j := range nFeatures {
			intercept -= xMean[j] * model.Coefficients[j]
		}
		model.Intercept = intercept
	}

	return model, nil
}

// Ridge is a fitted L2-regularized regression model.
// It has Predict() and Score() but no Fit().
type Ridge struct {
	// Coefficients are the learned feature weights.
	Coefficients []float64
	// Intercept is the learned bias term.
	Intercept float64
	// NFeatures is the number of features seen during fitting.
	NFeatures int
}

// Predict computes predictions for X using the fitted ridge regression model.
func (r *Ridge) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, err := validate.PredictInputs(X, r.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/linear: ridge predict failed: %w", err)
	}

	return linearPredict(X, r.Coefficients, r.Intercept, nSamples, r.NFeatures), nil
}

// Score returns the R-squared score of the model on the given data.
func (r *Ridge) Score(X *mat.Dense, y []float64) (float64, error) {
	preds, err := r.Predict(X)
	if err != nil {
		return 0, err
	}
	return r2Score(y, preds), nil
}

// GetCoefficients returns the learned feature weights.
func (r *Ridge) GetCoefficients() []float64 {
	out := make([]float64, len(r.Coefficients))
	copy(out, r.Coefficients)
	return out
}
