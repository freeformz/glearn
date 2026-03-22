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
	_ glearn.Estimator = LinearRegressionConfig{}
	_ glearn.Predictor = (*LinearRegression)(nil)
	_ glearn.Scorer    = (*LinearRegression)(nil)
)

// LinearRegressionConfig holds hyperparameters for ordinary least squares
// regression. It has Fit() but no Predict().
type LinearRegressionConfig struct {
	// FitIntercept determines whether to calculate the intercept for the model.
	// If false, data is expected to be centered.
	FitIntercept bool
}

// NewLinearRegression creates a LinearRegressionConfig with the given options.
// By default, FitIntercept is true.
func NewLinearRegression(opts ...LinearRegressionOption) LinearRegressionConfig {
	cfg := LinearRegressionConfig{
		FitIntercept: true,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit trains an ordinary least squares regression model using QR decomposition.
// It returns a fitted LinearRegression model.
func (cfg LinearRegressionConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.FitInputs(X, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/linear: OLS fit failed: %w", err)
	}

	// Check for context cancellation.
	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("glearn/linear: OLS fit cancelled: %w", ctx.Err())
	default:
	}

	// Build the design matrix, optionally prepending a column of ones for intercept.
	design := buildDesignMatrix(X, nSamples, nFeatures, cfg.FitIntercept)
	_, nCols := design.Dims()

	// Convert y to a column vector.
	yVec := mat.NewVecDense(nSamples, y)

	// Solve via QR decomposition: design * beta = y.
	var qr mat.QR
	qr.Factorize(design)

	beta := mat.NewVecDense(nCols, nil)
	if err := qr.SolveVecTo(beta, false, yVec); err != nil {
		return nil, fmt.Errorf("glearn/linear: OLS QR solve failed: %w", glearn.ErrSingularMatrix)
	}

	// Extract intercept and coefficients.
	model := &LinearRegression{
		NFeatures: nFeatures,
	}
	if cfg.FitIntercept {
		model.Intercept = beta.AtVec(0)
		model.Coefficients = make([]float64, nFeatures)
		for i := 0; i < nFeatures; i++ {
			model.Coefficients[i] = beta.AtVec(i + 1)
		}
	} else {
		model.Intercept = 0
		model.Coefficients = make([]float64, nFeatures)
		for i := 0; i < nFeatures; i++ {
			model.Coefficients[i] = beta.AtVec(i)
		}
	}

	return model, nil
}

// LinearRegression is a fitted ordinary least squares regression model.
// It has Predict() and Score() but no Fit().
type LinearRegression struct {
	// Coefficients are the learned feature weights.
	Coefficients []float64
	// Intercept is the learned bias term.
	Intercept float64
	// NFeatures is the number of features seen during fitting.
	NFeatures int
}

// Predict computes predictions for X using the fitted linear regression model.
func (lr *LinearRegression) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, err := validate.PredictInputs(X, lr.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/linear: OLS predict failed: %w", err)
	}

	return linearPredict(X, lr.Coefficients, lr.Intercept, nSamples, lr.NFeatures), nil
}

// Score returns the R-squared score of the model on the given data.
func (lr *LinearRegression) Score(X *mat.Dense, y []float64) (float64, error) {
	preds, err := lr.Predict(X)
	if err != nil {
		return 0, err
	}
	return r2Score(y, preds), nil
}

// GetCoefficients returns the learned feature weights.
func (lr *LinearRegression) GetCoefficients() []float64 {
	out := make([]float64, len(lr.Coefficients))
	copy(out, lr.Coefficients)
	return out
}

// buildDesignMatrix creates the design matrix, optionally prepending a column
// of ones for the intercept term.
func buildDesignMatrix(X *mat.Dense, nSamples, nFeatures int, fitIntercept bool) *mat.Dense {
	if !fitIntercept {
		// Return a copy to avoid mutating input.
		result := mat.NewDense(nSamples, nFeatures, nil)
		result.Copy(X)
		return result
	}

	// Prepend a column of ones.
	nCols := nFeatures + 1
	data := make([]float64, nSamples*nCols)
	raw := X.RawMatrix()
	for i := 0; i < nSamples; i++ {
		data[i*nCols] = 1.0
		copy(data[i*nCols+1:i*nCols+nCols], raw.Data[i*raw.Stride:i*raw.Stride+nFeatures])
	}
	return mat.NewDense(nSamples, nCols, data)
}

// linearPredict computes X * coef + intercept.
func linearPredict(X *mat.Dense, coef []float64, intercept float64, nSamples, nFeatures int) []float64 {
	preds := make([]float64, nSamples)
	raw := X.RawMatrix()
	for i := 0; i < nSamples; i++ {
		sum := intercept
		row := raw.Data[i*raw.Stride : i*raw.Stride+nFeatures]
		for j := 0; j < nFeatures; j++ {
			sum += row[j] * coef[j]
		}
		preds[i] = sum
	}
	return preds
}

// r2Score computes the R-squared (coefficient of determination) score.
func r2Score(yTrue, yPred []float64) float64 {
	n := len(yTrue)
	mean := 0.0
	for _, v := range yTrue {
		mean += v
	}
	mean /= float64(n)

	ssRes := 0.0
	ssTot := 0.0
	for i := range yTrue {
		diff := yTrue[i] - yPred[i]
		ssRes += diff * diff
		diffMean := yTrue[i] - mean
		ssTot += diffMean * diffMean
	}

	if ssTot == 0 {
		if ssRes == 0 {
			return 1.0
		}
		return 0.0
	}
	return 1.0 - ssRes/ssTot
}

// accuracyScore computes the fraction of correct predictions.
func accuracyScore(yTrue, yPred []float64) float64 {
	correct := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(yTrue))
}
