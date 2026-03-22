package pipeline

import (
	"context"
	"fmt"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

// Step represents a single step in a pipeline. Each step has a name and either
// a Transformer (for intermediate steps) or an Estimator (for the final step).
type Step struct {
	Name        string
	Transformer glearn.Transformer // set for intermediate steps
	Estimator   glearn.Estimator   // set for the final step only
}

// TransformStep creates a pipeline step that applies a transformer.
func TransformStep(name string, t glearn.Transformer) Step {
	return Step{Name: name, Transformer: t}
}

// EstimatorStep creates a pipeline step with a final estimator.
func EstimatorStep(name string, e glearn.Estimator) Step {
	return Step{Name: name, Estimator: e}
}

// Pipeline is an unfitted pipeline that chains transformers with a final
// estimator. It has Fit() but no Predict(), following the config/fitted pattern.
type Pipeline struct {
	Steps []Step
}

// New creates a new Pipeline from the given steps.
//
// The last step must be an EstimatorStep. All preceding steps must be
// TransformSteps. Validation happens at Fit time, not construction time.
func New(steps ...Step) Pipeline {
	return Pipeline{Steps: steps}
}

// Fit trains the pipeline: fits and transforms through each transformer step,
// then fits the final estimator on the transformed data.
//
// For efficiency, if a transformer implements glearn.FitTransformer, its
// FitTransform method is used instead of separate Fit + Transform calls.
func (p Pipeline) Fit(ctx context.Context, X *mat.Dense, y []float64) (*FittedPipeline, error) {
	if len(p.Steps) == 0 {
		return nil, fmt.Errorf("glearn/pipeline: %w: pipeline has no steps", glearn.ErrInvalidParameter)
	}

	// Validate: last step must be an estimator, all others must be transformers.
	lastIdx := len(p.Steps) - 1
	for i := range lastIdx {
		if p.Steps[i].Transformer == nil {
			return nil, fmt.Errorf("glearn/pipeline: %w: step %d (%q) must be a transformer step",
				glearn.ErrInvalidParameter, i, p.Steps[i].Name)
		}
	}
	if p.Steps[lastIdx].Estimator == nil {
		return nil, fmt.Errorf("glearn/pipeline: %w: last step (%q) must be an estimator step",
			glearn.ErrInvalidParameter, p.Steps[lastIdx].Name)
	}

	// Fit and transform through each transformer step.
	fittedTransformers := make([]glearn.FittedTransformer, 0, lastIdx)
	current := X
	for i := range lastIdx {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("glearn/pipeline: fit cancelled at step %d (%q): %w",
				i, p.Steps[i].Name, ctx.Err())
		default:
		}

		t := p.Steps[i].Transformer

		// Use FitTransform if available for efficiency.
		if ft, ok := t.(glearn.FitTransformer); ok {
			fitted, transformed, err := ft.FitTransform(ctx, current)
			if err != nil {
				return nil, fmt.Errorf("glearn/pipeline: step %d (%q) FitTransform failed: %w",
					i, p.Steps[i].Name, err)
			}
			fittedTransformers = append(fittedTransformers, fitted)
			current = transformed
		} else {
			fitted, err := t.Fit(ctx, current)
			if err != nil {
				return nil, fmt.Errorf("glearn/pipeline: step %d (%q) Fit failed: %w",
					i, p.Steps[i].Name, err)
			}
			transformed, err := fitted.Transform(current)
			if err != nil {
				return nil, fmt.Errorf("glearn/pipeline: step %d (%q) Transform failed: %w",
					i, p.Steps[i].Name, err)
			}
			fittedTransformers = append(fittedTransformers, fitted)
			current = transformed
		}
	}

	// Fit the final estimator on the transformed data.
	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("glearn/pipeline: fit cancelled at final step (%q): %w",
			p.Steps[lastIdx].Name, ctx.Err())
	default:
	}

	predictor, err := p.Steps[lastIdx].Estimator.Fit(ctx, current, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/pipeline: final step (%q) Fit failed: %w",
			p.Steps[lastIdx].Name, err)
	}

	return &FittedPipeline{
		Transformers: fittedTransformers,
		Predictor:    predictor,
	}, nil
}

// FittedPipeline is a fitted pipeline that chains fitted transformers with a
// fitted predictor. It has Predict() and Score() but no Fit().
type FittedPipeline struct {
	// Transformers are the fitted transformers from intermediate steps.
	Transformers []glearn.FittedTransformer
	// Predictor is the fitted predictor from the final step.
	Predictor glearn.Predictor
}

// Predict transforms X through each fitted transformer, then predicts using
// the fitted estimator.
func (fp *FittedPipeline) Predict(X *mat.Dense) ([]float64, error) {
	current, err := fp.transform(X)
	if err != nil {
		return nil, err
	}
	preds, err := fp.Predictor.Predict(current)
	if err != nil {
		return nil, fmt.Errorf("glearn/pipeline: predict failed: %w", err)
	}
	return preds, nil
}

// Score transforms X through each fitted transformer, predicts, and computes
// the R-squared score against y.
//
// If the predictor implements glearn.Scorer, its Score method is used directly.
// Otherwise, Score computes R-squared from predictions.
func (fp *FittedPipeline) Score(X *mat.Dense, y []float64) (float64, error) {
	current, err := fp.transform(X)
	if err != nil {
		return 0, err
	}

	// Prefer the predictor's own Score method if available.
	if scorer, ok := fp.Predictor.(glearn.Scorer); ok {
		score, err := scorer.Score(current, y)
		if err != nil {
			return 0, fmt.Errorf("glearn/pipeline: score failed: %w", err)
		}
		return score, nil
	}

	// Fallback: compute R-squared manually.
	preds, err := fp.Predictor.Predict(current)
	if err != nil {
		return 0, fmt.Errorf("glearn/pipeline: score predict failed: %w", err)
	}
	return r2Score(y, preds), nil
}

// transform applies each fitted transformer in sequence.
func (fp *FittedPipeline) transform(X *mat.Dense) (*mat.Dense, error) {
	current := X
	for i, t := range fp.Transformers {
		var err error
		current, err = t.Transform(current)
		if err != nil {
			return nil, fmt.Errorf("glearn/pipeline: transform step %d failed: %w", i, err)
		}
	}
	return current, nil
}

// r2Score computes the R-squared (coefficient of determination) score.
func r2Score(yTrue, yPred []float64) float64 {
	n := len(yTrue)
	if n == 0 {
		return 0
	}
	m := 0.0
	for _, v := range yTrue {
		m += v
	}
	m /= float64(n)

	ssRes := 0.0
	ssTot := 0.0
	for i := range yTrue {
		diff := yTrue[i] - yPred[i]
		ssRes += diff * diff
		diffMean := yTrue[i] - m
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
