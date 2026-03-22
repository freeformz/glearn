package modelselection

import (
	"context"
	"fmt"
	"math"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

// ParamSetter is implemented by estimators that support hyperparameter tuning
// via GridSearchCV. SetParams modifies the estimator's hyperparameters in place
// and returns an error if any parameter is unknown or invalid.
type ParamSetter interface {
	SetParams(params map[string]any) error
}

// GridSearchCVConfig configures an exhaustive grid search over parameter
// combinations with cross-validation.
type GridSearchCVConfig struct {
	// Estimator is the base estimator to tune. It must implement ParamSetter.
	Estimator glearn.Estimator
	// ParamGrid is the list of parameter combinations to evaluate.
	// Each map represents one combination of hyperparameters.
	ParamGrid []map[string]any
	// CV is the number of cross-validation folds (must be >= 2).
	CV int
	// Scorer is the scoring function: higher is better.
	Scorer func(yTrue, yPred []float64) float64
	// Seed controls randomness for cross-validation splits.
	Seed int64
}

// GridSearchCV holds the results of an exhaustive grid search.
type GridSearchCV struct {
	// BestParams is the parameter combination that achieved the highest mean CV score.
	BestParams map[string]any
	// BestScore is the highest mean cross-validation score.
	BestScore float64
	// BestModel is the predictor fitted on all data with BestParams.
	BestModel glearn.Predictor
	// CVResults contains the detailed results for each parameter combination.
	CVResults []CVResult
}

// CVResult holds cross-validation results for a single parameter combination.
type CVResult struct {
	// Params is the parameter combination evaluated.
	Params map[string]any
	// MeanScore is the mean of the per-fold scores.
	MeanScore float64
	// StdScore is the standard deviation of the per-fold scores.
	StdScore float64
	// Scores is the per-fold scores.
	Scores []float64
}

// Fit runs the grid search: for each parameter combination, it performs
// cross-validation, tracks the best result, and refits the best model on all data.
func (cfg GridSearchCVConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (*GridSearchCV, error) {
	if cfg.Estimator == nil {
		return nil, fmt.Errorf("glearn/modelselection: %w: estimator is nil", glearn.ErrInvalidParameter)
	}
	if len(cfg.ParamGrid) == 0 {
		return nil, fmt.Errorf("glearn/modelselection: %w: ParamGrid is empty", glearn.ErrInvalidParameter)
	}
	if cfg.Scorer == nil {
		return nil, fmt.Errorf("glearn/modelselection: %w: scorer is nil", glearn.ErrInvalidParameter)
	}
	if cfg.CV < 2 {
		return nil, fmt.Errorf("glearn/modelselection: %w: CV must be >= 2, got %d",
			glearn.ErrInvalidParameter, cfg.CV)
	}

	// Verify estimator implements ParamSetter.
	setter, ok := cfg.Estimator.(ParamSetter)
	if !ok {
		return nil, fmt.Errorf("glearn/modelselection: %w: estimator does not implement ParamSetter",
			glearn.ErrInvalidParameter)
	}

	results := make([]CVResult, 0, len(cfg.ParamGrid))
	bestScore := math.Inf(-1)
	bestIdx := -1

	for i, params := range cfg.ParamGrid {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("glearn/modelselection: grid search cancelled: %w", ctx.Err())
		default:
		}

		// Apply parameters to the estimator.
		if err := setter.SetParams(params); err != nil {
			return nil, fmt.Errorf("glearn/modelselection: grid search SetParams failed for combination %d: %w", i, err)
		}

		// Run cross-validation.
		scores, err := CrossValScore(ctx, cfg.Estimator, X, y, cfg.CV, cfg.Scorer, cfg.Seed)
		if err != nil {
			return nil, fmt.Errorf("glearn/modelselection: grid search CV failed for combination %d: %w", i, err)
		}

		meanScore := mean(scores)
		stdScore := std(scores, meanScore)

		result := CVResult{
			Params:    copyParams(params),
			MeanScore: meanScore,
			StdScore:  stdScore,
			Scores:    scores,
		}
		results = append(results, result)

		if meanScore > bestScore {
			bestScore = meanScore
			bestIdx = i
		}
	}

	// Refit on all data with the best parameters.
	if err := setter.SetParams(cfg.ParamGrid[bestIdx]); err != nil {
		return nil, fmt.Errorf("glearn/modelselection: grid search refit SetParams failed: %w", err)
	}
	bestModel, err := cfg.Estimator.Fit(ctx, X, y)
	if err != nil {
		return nil, fmt.Errorf("glearn/modelselection: grid search refit failed: %w", err)
	}

	return &GridSearchCV{
		BestParams: copyParams(cfg.ParamGrid[bestIdx]),
		BestScore:  bestScore,
		BestModel:  bestModel,
		CVResults:  results,
	}, nil
}

// mean computes the arithmetic mean of a slice.
func mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}

// std computes the population standard deviation given a precomputed mean.
func std(vals []float64, m float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	ss := 0.0
	for _, v := range vals {
		d := v - m
		ss += d * d
	}
	return math.Sqrt(ss / float64(len(vals)))
}

// copyParams creates a shallow copy of a parameter map.
func copyParams(params map[string]any) map[string]any {
	cp := make(map[string]any, len(params))
	for k, v := range params {
		cp[k] = v
	}
	return cp
}
