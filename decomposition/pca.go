package decomposition

import (
	"context"
	"fmt"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/internal/validate"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface checks.
var (
	_ glearn.Transformer       = PCAConfig{}
	_ glearn.FitTransformer    = PCAConfig{}
	_ glearn.FittedTransformer = (*PCA)(nil)
)

// PCAConfig holds hyperparameters for Principal Component Analysis.
// It has Fit() but no Transform().
type PCAConfig struct {
	// NComponents is the number of principal components to keep.
	// If zero, defaults to min(nSamples, nFeatures).
	NComponents int
}

// NewPCA creates a PCAConfig with the given options.
// By default, NComponents is 0, meaning min(nSamples, nFeatures).
func NewPCA(opts ...PCAOption) PCAConfig {
	cfg := PCAConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit computes the principal components from X and returns a fitted PCA.
// The input matrix X is not modified.
func (cfg PCAConfig) Fit(ctx context.Context, X *mat.Dense) (glearn.FittedTransformer, error) {
	nSamples, nFeatures, err := validate.Dimensions(X)
	if err != nil {
		return nil, fmt.Errorf("glearn/decomposition: PCA fit: %w", err)
	}

	// Check for context cancellation.
	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("glearn/decomposition: PCA fit cancelled: %w", ctx.Err())
	default:
	}

	// Determine number of components.
	nComponents := cfg.NComponents
	maxComponents := nSamples
	if nFeatures < maxComponents {
		maxComponents = nFeatures
	}
	if nComponents <= 0 {
		nComponents = maxComponents
	}
	if nComponents > maxComponents {
		return nil, fmt.Errorf("glearn/decomposition: PCA fit: %w: NComponents=%d exceeds max(%d samples, %d features)=%d",
			glearn.ErrInvalidParameter, nComponents, nSamples, nFeatures, maxComponents)
	}

	// Center data: subtract the column means.
	mean := columnMeans(X, nSamples, nFeatures)
	centered := mat.NewDense(nSamples, nFeatures, nil)
	for i := range nSamples {
		for j := range nFeatures {
			centered.Set(i, j, X.At(i, j)-mean[j])
		}
	}

	// Compute SVD of centered data.
	var svd mat.SVD
	ok := svd.Factorize(centered, mat.SVDThin)
	if !ok {
		return nil, fmt.Errorf("glearn/decomposition: PCA fit: SVD factorization failed: %w", glearn.ErrSingularMatrix)
	}

	// Extract singular values and right singular vectors (V).
	singularValues := make([]float64, nComponents)
	allSingular := svd.Values(nil)
	copy(singularValues, allSingular[:nComponents])

	var vt mat.Dense
	svd.VTo(&vt)

	// Components = first nComponents rows of V^T (each row is a principal component).
	components := mat.NewDense(nComponents, nFeatures, nil)
	for i := range nComponents {
		for j := range nFeatures {
			components.Set(i, j, vt.At(i, j))
		}
	}

	// Explained variance = singular_values^2 / (n_samples - 1).
	explainedVariance := make([]float64, nComponents)
	for i := range nComponents {
		explainedVariance[i] = singularValues[i] * singularValues[i] / float64(nSamples-1)
	}

	// Total variance from all singular values.
	totalVariance := 0.0
	for _, sv := range allSingular {
		totalVariance += sv * sv / float64(nSamples-1)
	}

	// Explained variance ratio.
	explainedVarianceRatio := make([]float64, nComponents)
	if totalVariance > 0 {
		for i := range nComponents {
			explainedVarianceRatio[i] = explainedVariance[i] / totalVariance
		}
	}

	return &PCA{
		Components:             components,
		ExplainedVariance:      explainedVariance,
		ExplainedVarianceRatio: explainedVarianceRatio,
		Mean:                   mean,
		NComponents:            nComponents,
		NFeatures:              nFeatures,
		SingularValues:         singularValues,
	}, nil
}

// FitTransform fits PCA on X and returns both the fitted transformer and the
// transformed data in a single step. This is more efficient than calling Fit
// followed by Transform because it avoids recomputing the projection.
func (cfg PCAConfig) FitTransform(ctx context.Context, X *mat.Dense) (glearn.FittedTransformer, *mat.Dense, error) {
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

// PCA is a fitted Principal Component Analysis model. It transforms data by
// projecting onto the learned principal components.
//
// PCA is immutable after construction and safe for concurrent Transform calls.
type PCA struct {
	// Components are the principal axes (NComponents x NFeatures).
	// Each row is a principal component direction.
	Components *mat.Dense
	// ExplainedVariance is the variance explained by each component.
	ExplainedVariance []float64
	// ExplainedVarianceRatio is the proportion of total variance explained by each component.
	ExplainedVarianceRatio []float64
	// Mean is the per-feature mean from training data, used for centering.
	Mean []float64
	// NComponents is the number of principal components kept.
	NComponents int
	// NFeatures is the number of features seen during fitting.
	NFeatures int
	// SingularValues are the singular values corresponding to each component.
	SingularValues []float64
}

// Transform projects X onto the principal components learned during Fit.
// Returns a new matrix with shape (nSamples, NComponents); X is not modified.
func (pca *PCA) Transform(X *mat.Dense) (*mat.Dense, error) {
	nSamples, err := validate.PredictInputs(X, pca.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/decomposition: PCA transform: %w", err)
	}

	// Center the input data using the training mean.
	centered := mat.NewDense(nSamples, pca.NFeatures, nil)
	for i := range nSamples {
		for j := range pca.NFeatures {
			centered.Set(i, j, X.At(i, j)-pca.Mean[j])
		}
	}

	// Project: result = centered * Components^T
	result := mat.NewDense(nSamples, pca.NComponents, nil)
	result.Mul(centered, pca.Components.T())
	return result, nil
}

// InverseTransform maps data from the reduced space back to the original space.
// Returns a new matrix with shape (nSamples, NFeatures); X is not modified.
// Note: this is an approximation if NComponents < NFeatures.
func (pca *PCA) InverseTransform(X *mat.Dense) (*mat.Dense, error) {
	nSamples, nCols, err := validate.Dimensions(X)
	if err != nil {
		return nil, fmt.Errorf("glearn/decomposition: PCA inverse transform: %w", err)
	}
	if nCols != pca.NComponents {
		return nil, fmt.Errorf("glearn/decomposition: PCA inverse transform: %w: expected %d components but got %d",
			glearn.ErrDimensionMismatch, pca.NComponents, nCols)
	}

	// Reconstruct: result = X * Components + Mean
	result := mat.NewDense(nSamples, pca.NFeatures, nil)
	result.Mul(X, pca.Components)

	// Add back the mean.
	for i := range nSamples {
		for j := range pca.NFeatures {
			result.Set(i, j, result.At(i, j)+pca.Mean[j])
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
