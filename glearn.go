// Package glearn is a comprehensive classical machine learning library for Go.
//
// glearn provides scikit-learn-style algorithms with compile-time safety:
// unfitted config types have Fit() but no Predict(), and fitted model types
// have Predict() but no Fit(). This prevents predict-before-fit errors at
// compile time.
//
// Core interfaces:
//   - [Estimator]: unfitted model with Fit() -> [Predictor]
//   - [Predictor]: fitted model with Predict()
//   - [Transformer]: unfitted transformer with Fit() -> [FittedTransformer]
//   - [FittedTransformer]: fitted transformer with Transform()
//   - [Scorer]: evaluates a fitted model on test data
package glearn

import (
	"context"

	"gonum.org/v1/gonum/mat"
)

// Estimator is an unfitted model that can be trained on data.
// Fit returns a Predictor — a different type that can make predictions.
// The Estimator itself has no Predict method, ensuring compile-time safety.
type Estimator interface {
	Fit(ctx context.Context, X *mat.Dense, y []float64) (Predictor, error)
}

// Predictor is a fitted model that can make predictions on new data.
// Predictors are immutable after construction and safe for concurrent use.
type Predictor interface {
	Predict(X *mat.Dense) ([]float64, error)
}

// Classifier is a fitted model that can predict class probabilities.
type Classifier interface {
	Predictor
	PredictProbabilities(X *mat.Dense) (*mat.Dense, error)
}

// Transformer is an unfitted data transformer (e.g., StandardScaler, PCA).
// Fit returns a FittedTransformer that can transform new data.
type Transformer interface {
	Fit(ctx context.Context, X *mat.Dense) (FittedTransformer, error)
}

// FittedTransformer transforms data using parameters learned during Fit.
// FittedTransformers are immutable after construction and safe for concurrent use.
type FittedTransformer interface {
	Transform(X *mat.Dense) (*mat.Dense, error)
}

// FitTransformer combines Fit and Transform in a single step for efficiency.
// Transformers that implement this can avoid redundant computation when the
// caller needs both the fitted transformer and the transformed data.
type FitTransformer interface {
	FitTransform(ctx context.Context, X *mat.Dense) (FittedTransformer, *mat.Dense, error)
}

// SupervisedTransformer is a transformer that uses target labels during fitting
// (e.g., SelectKBest, LDA).
type SupervisedTransformer interface {
	Fit(ctx context.Context, X *mat.Dense, y []float64) (FittedTransformer, error)
}

// Scorer evaluates a fitted model's performance on test data.
type Scorer interface {
	Score(X *mat.Dense, y []float64) (float64, error)
}

// Cloneable creates an unfitted copy with the same hyperparameters.
// Used by GridSearchCV and cross-validation to create fresh estimators.
type Cloneable interface {
	Clone() Estimator
}

// HasCoefficients provides access to a fitted model's learned coefficients.
type HasCoefficients interface {
	GetCoefficients() []float64
}

// HasFeatureImportances provides access to feature importance scores.
type HasFeatureImportances interface {
	GetFeatureImportances() []float64
}

// HasClasses provides access to the classes a classifier was trained on.
type HasClasses interface {
	GetClasses() []float64
}
