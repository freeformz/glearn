package linear

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/internal/validate"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

// Compile-time interface checks.
var (
	_ glearn.Estimator  = LogisticRegressionConfig{}
	_ glearn.Predictor  = (*LogisticRegression)(nil)
	_ glearn.Classifier = (*LogisticRegression)(nil)
	_ glearn.Scorer     = (*LogisticRegression)(nil)
)

// LogisticRegressionConfig holds hyperparameters for logistic regression
// classification. It has Fit() but no Predict().
type LogisticRegressionConfig struct {
	// C is the inverse of regularization strength (L2). Default is 1.0.
	// Smaller values mean stronger regularization.
	C float64
	// FitIntercept determines whether to calculate the intercept.
	FitIntercept bool
	// MaxIter is the maximum number of L-BFGS iterations. Default is 100.
	MaxIter int
	// Tolerance is the convergence tolerance for L-BFGS. Default is 1e-4.
	Tolerance float64
}

// NewLogisticRegression creates a LogisticRegressionConfig with the given options.
// By default, C is 1.0, FitIntercept is true, MaxIter is 100, and Tolerance
// is 1e-4.
func NewLogisticRegression(opts ...LogisticRegressionOption) LogisticRegressionConfig {
	cfg := LogisticRegressionConfig{
		C:            1.0,
		FitIntercept: true,
		MaxIter:      100,
		Tolerance:    1e-4,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit trains a logistic regression model using L-BFGS optimization.
// For binary classification it trains a single model. For multiclass
// it uses a one-vs-rest strategy.
func (cfg LogisticRegressionConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.FitInputs(X, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/linear: logistic regression fit failed: %w", err)
	}

	if cfg.C <= 0 {
		return nil, fmt.Errorf("glearn/linear: logistic regression fit failed: %w: C must be positive, got %g",
			glearn.ErrInvalidParameter, cfg.C)
	}

	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("glearn/linear: logistic regression fit cancelled: %w", ctx.Err())
	default:
	}

	// Discover unique classes.
	classes := uniqueSorted(y)

	if len(classes) < 2 {
		return nil, fmt.Errorf("glearn/linear: logistic regression fit failed: %w: need at least 2 classes, got %d",
			glearn.ErrInvalidParameter, len(classes))
	}

	// Number of parameters per binary problem.
	nParams := nFeatures
	if cfg.FitIntercept {
		nParams = nFeatures + 1
	}

	if len(classes) == 2 {
		// Binary classification.
		binaryY := make([]float64, nSamples)
		for i, v := range y {
			if v == classes[1] {
				binaryY[i] = 1.0
			}
		}

		coef, intercept, err := cfg.fitBinary(ctx, X, binaryY, nSamples, nFeatures, nParams)
		if err != nil {
			return nil, err
		}

		model := &LogisticRegression{
			Coefficients: [][]float64{coef},
			Intercept:    []float64{intercept},
			Classes:      classes,
			NFeatures:    nFeatures,
		}
		return model, nil
	}

	// Multiclass: one-vs-rest.
	allCoefs := make([][]float64, len(classes))
	allIntercepts := make([]float64, len(classes))

	for c, classVal := range classes {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("glearn/linear: logistic regression fit cancelled: %w", ctx.Err())
		default:
		}

		binaryY := make([]float64, nSamples)
		for i, v := range y {
			if v == classVal {
				binaryY[i] = 1.0
			}
		}

		coef, intercept, err := cfg.fitBinary(ctx, X, binaryY, nSamples, nFeatures, nParams)
		if err != nil {
			return nil, fmt.Errorf("glearn/linear: logistic regression fit failed for class %g: %w", classVal, err)
		}

		allCoefs[c] = coef
		allIntercepts[c] = intercept
	}

	model := &LogisticRegression{
		Coefficients: allCoefs,
		Intercept:    allIntercepts,
		Classes:      classes,
		NFeatures:    nFeatures,
	}
	return model, nil
}

// fitBinary trains a single binary logistic regression model.
// binaryY must have values 0 or 1.
func (cfg LogisticRegressionConfig) fitBinary(
	ctx context.Context,
	X *mat.Dense,
	binaryY []float64,
	nSamples, nFeatures, nParams int,
) (coef []float64, intercept float64, err error) {
	rawX := X.RawMatrix()
	lambda := 1.0 / cfg.C // regularization strength

	// Define the objective function: negative log-likelihood + L2 penalty.
	problem := optimize.Problem{
		Func: func(params []float64) float64 {
			loss := 0.0
			for i := 0; i < nSamples; i++ {
				z := dotWithParams(rawX.Data[i*rawX.Stride:i*rawX.Stride+nFeatures], params, nFeatures, cfg.FitIntercept)
				// Numerically stable log-loss: log(1 + exp(-y*z)) where y in {-1, +1}
				label := 2*binaryY[i] - 1 // map 0/1 to -1/+1
				margin := label * z
				if margin > 0 {
					loss += math.Log1p(math.Exp(-margin))
				} else {
					loss += -margin + math.Log1p(math.Exp(margin))
				}
			}
			// L2 regularization on coefficients only (not intercept).
			reg := 0.0
			for j := 0; j < nFeatures; j++ {
				idx := j
				if cfg.FitIntercept {
					idx = j + 1
				}
				reg += params[idx] * params[idx]
			}
			return loss/float64(nSamples) + 0.5*lambda*reg
		},
		Grad: func(grad, params []float64) {
			for k := range grad {
				grad[k] = 0
			}
			for i := 0; i < nSamples; i++ {
				z := dotWithParams(rawX.Data[i*rawX.Stride:i*rawX.Stride+nFeatures], params, nFeatures, cfg.FitIntercept)
				prob := sigmoid(z)
				diff := (prob - binaryY[i]) / float64(nSamples)

				if cfg.FitIntercept {
					grad[0] += diff
					for j := 0; j < nFeatures; j++ {
						grad[j+1] += diff * rawX.Data[i*rawX.Stride+j]
					}
				} else {
					for j := 0; j < nFeatures; j++ {
						grad[j] += diff * rawX.Data[i*rawX.Stride+j]
					}
				}
			}
			// L2 regularization gradient.
			for j := 0; j < nFeatures; j++ {
				idx := j
				if cfg.FitIntercept {
					idx = j + 1
				}
				grad[idx] += lambda * params[idx]
			}
		},
	}

	// Initialize parameters to zero.
	initParams := make([]float64, nParams)

	settings := &optimize.Settings{
		MajorIterations: cfg.MaxIter,
		GradientThreshold: cfg.Tolerance,
		Converger: &optimize.FunctionConverge{
			Absolute:   1e-12,
			Relative:   cfg.Tolerance,
			Iterations: cfg.MaxIter,
		},
	}

	result, err := optimize.Minimize(problem, initParams, settings, &optimize.LBFGS{})
	if err != nil {
		// Check if we hit max iterations but still got a result.
		if result == nil {
			return nil, 0, fmt.Errorf("glearn/linear: logistic regression L-BFGS failed: %w", err)
		}
		// Use the result even if convergence wasn't perfect.
	}

	params := result.X

	if cfg.FitIntercept {
		intercept = params[0]
		coef = make([]float64, nFeatures)
		copy(coef, params[1:])
	} else {
		intercept = 0
		coef = make([]float64, nFeatures)
		copy(coef, params)
	}

	return coef, intercept, nil
}

// LogisticRegression is a fitted logistic regression classification model.
// It has Predict(), PredictProbabilities(), and Score() but no Fit().
type LogisticRegression struct {
	// Coefficients holds the learned weights. For binary classification,
	// this has one row. For multiclass (OvR), one row per class.
	Coefficients [][]float64
	// Intercept holds the learned bias terms, one per binary classifier.
	Intercept []float64
	// Classes are the unique class labels in sorted order.
	Classes []float64
	// NFeatures is the number of features seen during fitting.
	NFeatures int
}

// Predict returns class label predictions for X.
func (lr *LogisticRegression) Predict(X *mat.Dense) ([]float64, error) {
	probs, err := lr.PredictProbabilities(X)
	if err != nil {
		return nil, err
	}

	nSamples, nClasses := probs.Dims()
	preds := make([]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		bestClass := 0
		bestProb := probs.At(i, 0)
		for c := 1; c < nClasses; c++ {
			if probs.At(i, c) > bestProb {
				bestProb = probs.At(i, c)
				bestClass = c
			}
		}
		preds[i] = lr.Classes[bestClass]
	}

	return preds, nil
}

// PredictProbabilities returns class probability estimates for X.
// The returned matrix has shape (nSamples, nClasses).
func (lr *LogisticRegression) PredictProbabilities(X *mat.Dense) (*mat.Dense, error) {
	nSamples, err := validate.PredictInputs(X, lr.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/linear: logistic regression predict failed: %w", err)
	}

	rawX := X.RawMatrix()
	nClasses := len(lr.Classes)

	if nClasses == 2 {
		// Binary: use single set of coefficients.
		probs := mat.NewDense(nSamples, 2, nil)
		for i := 0; i < nSamples; i++ {
			z := lr.Intercept[0]
			for j := 0; j < lr.NFeatures; j++ {
				z += rawX.Data[i*rawX.Stride+j] * lr.Coefficients[0][j]
			}
			p := sigmoid(z)
			probs.Set(i, 0, 1-p)
			probs.Set(i, 1, p)
		}
		return probs, nil
	}

	// Multiclass: OvR with softmax-like normalization.
	probs := mat.NewDense(nSamples, nClasses, nil)
	for i := 0; i < nSamples; i++ {
		sum := 0.0
		for c := 0; c < nClasses; c++ {
			z := lr.Intercept[c]
			for j := 0; j < lr.NFeatures; j++ {
				z += rawX.Data[i*rawX.Stride+j] * lr.Coefficients[c][j]
			}
			p := sigmoid(z)
			probs.Set(i, c, p)
			sum += p
		}
		// Normalize to sum to 1.
		if sum > 0 {
			for c := 0; c < nClasses; c++ {
				probs.Set(i, c, probs.At(i, c)/sum)
			}
		}
	}

	return probs, nil
}

// Score returns the accuracy of the model on the given data.
func (lr *LogisticRegression) Score(X *mat.Dense, y []float64) (float64, error) {
	preds, err := lr.Predict(X)
	if err != nil {
		return 0, err
	}
	return accuracyScore(y, preds), nil
}

// GetCoefficients returns the learned coefficients. For binary classification,
// returns the single set of coefficients. For multiclass, returns concatenated
// coefficients for all classifiers.
func (lr *LogisticRegression) GetCoefficients() []float64 {
	total := 0
	for _, c := range lr.Coefficients {
		total += len(c)
	}
	out := make([]float64, 0, total)
	for _, c := range lr.Coefficients {
		out = append(out, c...)
	}
	return out
}

// GetClasses returns the unique class labels.
func (lr *LogisticRegression) GetClasses() []float64 {
	out := make([]float64, len(lr.Classes))
	copy(out, lr.Classes)
	return out
}

// sigmoid computes the logistic sigmoid function.
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1.0 / (1.0 + math.Exp(-x))
	}
	// For negative x, use the equivalent form to avoid overflow.
	ez := math.Exp(x)
	return ez / (1.0 + ez)
}

// dotWithParams computes the dot product of a feature row with the parameter
// vector, handling intercept if present.
func dotWithParams(row []float64, params []float64, nFeatures int, fitIntercept bool) float64 {
	z := 0.0
	if fitIntercept {
		z = params[0]
		for j := 0; j < nFeatures; j++ {
			z += row[j] * params[j+1]
		}
	} else {
		for j := 0; j < nFeatures; j++ {
			z += row[j] * params[j]
		}
	}
	return z
}

// uniqueSorted returns the sorted unique values in s.
func uniqueSorted(s []float64) []float64 {
	set := make(map[float64]struct{})
	for _, v := range s {
		set[v] = struct{}{}
	}
	result := make([]float64, 0, len(set))
	for v := range set {
		result = append(result, v)
	}
	sort.Float64s(result)
	return result
}
