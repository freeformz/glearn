package modelselection

import (
	"context"
	"fmt"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

// CrossValScore evaluates an estimator using K-Fold cross-validation.
//
// The estimator is fit on each training fold and scored on the corresponding
// test fold using the provided scorer function. Since config types in glearn
// are value types (structs), the same estimator config can be safely reused
// across folds without cloning.
//
// Parameters:
//   - est: the estimator to evaluate (must implement glearn.Estimator)
//   - X: the feature matrix
//   - y: the target values
//   - cv: the number of cross-validation folds (must be >= 2)
//   - scorer: a function that takes (yTrue, yPred) and returns a score
//   - seed: random seed for shuffling fold indices
//
// Returns a slice of scores, one per fold.
func CrossValScore(
	ctx context.Context,
	est glearn.Estimator,
	X *mat.Dense, y []float64,
	cv int,
	scorer func(yTrue, yPred []float64) float64,
	seed int64,
) ([]float64, error) {
	if est == nil {
		return nil, fmt.Errorf("glearn/modelselection: %w: estimator is nil", glearn.ErrInvalidParameter)
	}
	if X == nil {
		return nil, fmt.Errorf("glearn/modelselection: %w: X is nil", glearn.ErrEmptyInput)
	}
	if scorer == nil {
		return nil, fmt.Errorf("glearn/modelselection: %w: scorer is nil", glearn.ErrInvalidParameter)
	}

	nSamples, nFeatures := X.Dims()
	if len(y) != nSamples {
		return nil, fmt.Errorf("glearn/modelselection: %w: X has %d samples but y has %d elements",
			glearn.ErrDimensionMismatch, nSamples, len(y))
	}

	kf := KFold{NSplits: cv, Shuffle: true, Seed: seed}
	folds, err := kf.Split(nSamples)
	if err != nil {
		return nil, fmt.Errorf("glearn/modelselection: cross-validation split failed: %w", err)
	}

	scores := make([]float64, len(folds))
	for i, fold := range folds {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("glearn/modelselection: cross-validation cancelled: %w", ctx.Err())
		default:
		}

		XTrain := extractRows(X, fold.TrainIndices, nFeatures)
		yTrain := extractElements(y, fold.TrainIndices)
		XTest := extractRows(X, fold.TestIndices, nFeatures)
		yTest := extractElements(y, fold.TestIndices)

		model, err := est.Fit(ctx, XTrain, yTrain)
		if err != nil {
			return nil, fmt.Errorf("glearn/modelselection: fold %d fit failed: %w", i, err)
		}

		preds, err := model.Predict(XTest)
		if err != nil {
			return nil, fmt.Errorf("glearn/modelselection: fold %d predict failed: %w", i, err)
		}

		scores[i] = scorer(yTest, preds)
	}

	return scores, nil
}
