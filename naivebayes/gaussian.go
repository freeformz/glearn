package naivebayes

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
	_ glearn.Estimator  = GaussianNBConfig{}
	_ glearn.Predictor  = (*GaussianNB)(nil)
	_ glearn.Classifier = (*GaussianNB)(nil)
)

// GaussianNBConfig holds hyperparameters for Gaussian Naive Bayes classification.
// It has Fit() but no Predict().
type GaussianNBConfig struct {
	// VarSmoothing is the portion of the largest variance of all features
	// that is added to variances for calculation stability. Default is 1e-9.
	VarSmoothing float64
}

// NewGaussianNB creates a GaussianNBConfig with the given options.
func NewGaussianNB(opts ...GaussianNBOption) GaussianNBConfig {
	cfg := GaussianNBConfig{
		VarSmoothing: 1e-9,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit trains a Gaussian Naive Bayes classifier on X and y.
// Returns a fitted GaussianNB model.
func (cfg GaussianNBConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.FitInputs(X, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/naivebayes: gaussian fit failed: %w", err)
	}

	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("glearn/naivebayes: gaussian fit cancelled: %w", ctx.Err())
	default:
	}

	// Discover unique classes and their indices.
	classIndices := make(map[float64][]int)
	for i, label := range y {
		classIndices[label] = append(classIndices[label], i)
	}

	nClasses := len(classIndices)
	classes := make([]float64, 0, nClasses)
	for c := range classIndices {
		classes = append(classes, c)
	}
	// Sort classes for deterministic ordering.
	sortFloat64s(classes)

	raw := X.RawMatrix()

	// Compute per-class mean and variance for each feature, plus class priors.
	classPriors := make([]float64, nClasses)
	theta := make([][]float64, nClasses) // class means
	sigma := make([][]float64, nClasses) // class variances

	for ci, class := range classes {
		indices := classIndices[class]
		count := float64(len(indices))
		classPriors[ci] = count / float64(nSamples)

		means := make([]float64, nFeatures)
		for _, idx := range indices {
			for j := range nFeatures {
				means[j] += raw.Data[idx*raw.Stride+j]
			}
		}
		for j := range nFeatures {
			means[j] /= count
		}

		variances := make([]float64, nFeatures)
		for _, idx := range indices {
			for j := range nFeatures {
				d := raw.Data[idx*raw.Stride+j] - means[j]
				variances[j] += d * d
			}
		}
		for j := range nFeatures {
			variances[j] /= count
		}

		theta[ci] = means
		sigma[ci] = variances
	}

	// Compute smoothing: VarSmoothing * max variance across all features/classes.
	maxVar := 0.0
	for _, vars := range sigma {
		for _, v := range vars {
			if v > maxVar {
				maxVar = v
			}
		}
	}
	smoothing := cfg.VarSmoothing * maxVar

	// Add smoothing to all variances.
	for ci := range sigma {
		for j := range sigma[ci] {
			sigma[ci][j] += smoothing
		}
	}

	return &GaussianNB{
		ClassPriors: classPriors,
		Theta:       theta,
		Sigma:       sigma,
		Classes:     classes,
		NFeatures:   nFeatures,
	}, nil
}

// GaussianNB is a fitted Gaussian Naive Bayes classifier.
// It has Predict() and PredictProbabilities() but no Fit().
type GaussianNB struct {
	// ClassPriors are the prior probabilities of each class.
	ClassPriors []float64
	// Theta holds the mean of each feature per class, shape [nClasses][nFeatures].
	Theta [][]float64
	// Sigma holds the variance of each feature per class, shape [nClasses][nFeatures].
	Sigma [][]float64
	// Classes are the unique class labels seen during fitting.
	Classes []float64
	// NFeatures is the number of features seen during fitting.
	NFeatures int
}

// Predict returns the most likely class label for each row of X.
func (gnb *GaussianNB) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, err := validate.PredictInputs(X, gnb.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/naivebayes: gaussian predict failed: %w", err)
	}

	logProbs := gnb.computeLogPosterior(X, nSamples)

	result := make([]float64, nSamples)
	nClasses := len(gnb.Classes)
	for i := range nSamples {
		bestClass := 0
		bestLogProb := logProbs[i*nClasses]
		for ci := 1; ci < nClasses; ci++ {
			if logProbs[i*nClasses+ci] > bestLogProb {
				bestLogProb = logProbs[i*nClasses+ci]
				bestClass = ci
			}
		}
		result[i] = gnb.Classes[bestClass]
	}

	return result, nil
}

// PredictProbabilities returns the posterior probability of each class for each
// row of X. Returns a matrix of shape (nSamples, nClasses).
func (gnb *GaussianNB) PredictProbabilities(X *mat.Dense) (*mat.Dense, error) {
	nSamples, err := validate.PredictInputs(X, gnb.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/naivebayes: gaussian predict_proba failed: %w", err)
	}

	logProbs := gnb.computeLogPosterior(X, nSamples)
	nClasses := len(gnb.Classes)

	// Convert log-probabilities to probabilities using log-sum-exp for stability.
	probs := make([]float64, nSamples*nClasses)
	for i := range nSamples {
		offset := i * nClasses

		// Find max log-prob for this sample (for numerical stability).
		maxLP := logProbs[offset]
		for ci := 1; ci < nClasses; ci++ {
			if logProbs[offset+ci] > maxLP {
				maxLP = logProbs[offset+ci]
			}
		}

		// Compute exp(logProb - maxLP) and sum.
		sum := 0.0
		for ci := range nClasses {
			probs[offset+ci] = math.Exp(logProbs[offset+ci] - maxLP)
			sum += probs[offset+ci]
		}

		// Normalize.
		for ci := range nClasses {
			probs[offset+ci] /= sum
		}
	}

	return mat.NewDense(nSamples, nClasses, probs), nil
}

// GetClasses returns the unique class labels.
func (gnb *GaussianNB) GetClasses() []float64 {
	out := make([]float64, len(gnb.Classes))
	copy(out, gnb.Classes)
	return out
}

// computeLogPosterior computes unnormalized log posterior probabilities.
// Returns a flat slice of shape [nSamples * nClasses].
func (gnb *GaussianNB) computeLogPosterior(X *mat.Dense, nSamples int) []float64 {
	nClasses := len(gnb.Classes)
	raw := X.RawMatrix()
	logProbs := make([]float64, nSamples*nClasses)

	for i := range nSamples {
		for ci := range nClasses {
			logProb := math.Log(gnb.ClassPriors[ci])
			for j := range gnb.NFeatures {
				x := raw.Data[i*raw.Stride+j]
				mean := gnb.Theta[ci][j]
				variance := gnb.Sigma[ci][j]
				// Log of Gaussian PDF: -0.5 * (log(2*pi*var) + (x-mean)^2/var)
				d := x - mean
				logProb += -0.5 * (math.Log(2*math.Pi*variance) + d*d/variance)
			}
			logProbs[i*nClasses+ci] = logProb
		}
	}

	return logProbs
}

// sortFloat64s sorts a slice of float64 values in ascending order.
func sortFloat64s(a []float64) {
	for i := 1; i < len(a); i++ {
		key := a[i]
		j := i - 1
		for j >= 0 && a[j] > key {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = key
	}
}
