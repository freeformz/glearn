package modelselection

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

// paramEstimator is a simple estimator that implements ParamSetter for testing.
// It predicts y = constant * mean(X row) where constant is a tunable parameter.
type paramEstimator struct {
	Constant float64
}

var _ ParamSetter = (*paramEstimator)(nil)

func (pe *paramEstimator) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures := X.Dims()
	if len(y) != nSamples {
		return nil, glearn.ErrDimensionMismatch
	}

	// Simple linear predictor: predict mean(y) * constant_influence.
	sum := 0.0
	for _, v := range y {
		sum += v
	}
	meanY := sum / float64(nSamples)

	return &paramPredictor{
		meanPred:  meanY,
		constant:  pe.Constant,
		nFeatures: nFeatures,
	}, nil
}

func (pe *paramEstimator) SetParams(params map[string]any) error {
	for k, v := range params {
		switch k {
		case "Constant":
			val, ok := v.(float64)
			if !ok {
				return fmt.Errorf("glearn/modelselection: %w: Constant must be float64, got %T",
					glearn.ErrInvalidParameter, v)
			}
			pe.Constant = val
		default:
			return fmt.Errorf("glearn/modelselection: %w: unknown parameter %q",
				glearn.ErrInvalidParameter, k)
		}
	}
	return nil
}

type paramPredictor struct {
	meanPred  float64
	constant  float64
	nFeatures int
}

func (pp *paramPredictor) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, nFeatures := X.Dims()
	if nFeatures != pp.nFeatures {
		return nil, glearn.ErrDimensionMismatch
	}
	preds := make([]float64, nSamples)
	for i := range nSamples {
		// Predict constant * meanPred (independent of input, but tunable).
		preds[i] = pp.constant * pp.meanPred
	}
	return preds, nil
}

// negMSE returns negative mean squared error (higher is better).
func negMSE(yTrue, yPred []float64) float64 {
	sum := 0.0
	for i := range yTrue {
		d := yTrue[i] - yPred[i]
		sum += d * d
	}
	return -sum / float64(len(yTrue))
}

func TestGridSearchCV_Basic(t *testing.T) {
	// Create data where y = mean(y) exactly, so constant=1 should be best.
	nSamples := 60
	X := mat.NewDense(nSamples, 2, nil)
	y := make([]float64, nSamples)
	for i := range nSamples {
		X.Set(i, 0, float64(i))
		X.Set(i, 1, float64(i+1))
		y[i] = 5.0 // constant y
	}

	est := &paramEstimator{Constant: 1.0}
	grid := GridSearchCVConfig{
		Estimator: est,
		ParamGrid: []map[string]any{
			{"Constant": 0.5},
			{"Constant": 1.0},
			{"Constant": 2.0},
		},
		CV:     3,
		Scorer: negMSE,
		Seed:   42,
	}

	result, err := grid.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// With constant y=5, meanPred=5, constant=1 gives pred=5 (perfect).
	// constant=0.5 gives pred=2.5, constant=2.0 gives pred=10.
	if result.BestParams["Constant"] != 1.0 {
		t.Errorf("expected best Constant=1.0, got %v", result.BestParams["Constant"])
	}

	if len(result.CVResults) != 3 {
		t.Errorf("expected 3 CV results, got %d", len(result.CVResults))
	}

	if result.BestModel == nil {
		t.Error("BestModel is nil")
	}

	// Best score should be 0 (negative MSE of 0).
	if result.BestScore != 0 {
		t.Errorf("expected best score 0 (negMSE of perfect prediction), got %g", result.BestScore)
	}
}

func TestGridSearchCV_CVResultsOrder(t *testing.T) {
	nSamples := 40
	X := mat.NewDense(nSamples, 1, nil)
	y := make([]float64, nSamples)
	for i := range nSamples {
		X.Set(i, 0, float64(i))
		y[i] = 10.0
	}

	est := &paramEstimator{Constant: 1.0}
	grid := GridSearchCVConfig{
		Estimator: est,
		ParamGrid: []map[string]any{
			{"Constant": 0.1},
			{"Constant": 1.0},
			{"Constant": 5.0},
		},
		CV:     3,
		Scorer: negMSE,
		Seed:   99,
	}

	result, err := grid.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Results should be in the same order as ParamGrid.
	if len(result.CVResults) != 3 {
		t.Fatalf("expected 3 results, got %d", len(result.CVResults))
	}
	for i, res := range result.CVResults {
		expectedConstant := grid.ParamGrid[i]["Constant"]
		if res.Params["Constant"] != expectedConstant {
			t.Errorf("result %d: expected Constant=%v, got %v", i, expectedConstant, res.Params["Constant"])
		}
		if len(res.Scores) != 3 {
			t.Errorf("result %d: expected 3 fold scores, got %d", i, len(res.Scores))
		}
	}
}

func TestGridSearchCV_InvalidInputs(t *testing.T) {
	X := mat.NewDense(20, 2, nil)
	y := make([]float64, 20)

	t.Run("nil estimator", func(t *testing.T) {
		grid := GridSearchCVConfig{
			Estimator: nil,
			ParamGrid: []map[string]any{{"Constant": 1.0}},
			CV:        3,
			Scorer:    negMSE,
		}
		_, err := grid.Fit(t.Context(), X, y)
		if !errors.Is(err, glearn.ErrInvalidParameter) {
			t.Errorf("expected ErrInvalidParameter, got %v", err)
		}
	})

	t.Run("empty param grid", func(t *testing.T) {
		grid := GridSearchCVConfig{
			Estimator: &paramEstimator{},
			ParamGrid: nil,
			CV:        3,
			Scorer:    negMSE,
		}
		_, err := grid.Fit(t.Context(), X, y)
		if !errors.Is(err, glearn.ErrInvalidParameter) {
			t.Errorf("expected ErrInvalidParameter, got %v", err)
		}
	})

	t.Run("nil scorer", func(t *testing.T) {
		grid := GridSearchCVConfig{
			Estimator: &paramEstimator{},
			ParamGrid: []map[string]any{{"Constant": 1.0}},
			CV:        3,
			Scorer:    nil,
		}
		_, err := grid.Fit(t.Context(), X, y)
		if !errors.Is(err, glearn.ErrInvalidParameter) {
			t.Errorf("expected ErrInvalidParameter, got %v", err)
		}
	})

	t.Run("CV < 2", func(t *testing.T) {
		grid := GridSearchCVConfig{
			Estimator: &paramEstimator{},
			ParamGrid: []map[string]any{{"Constant": 1.0}},
			CV:        1,
			Scorer:    negMSE,
		}
		_, err := grid.Fit(t.Context(), X, y)
		if !errors.Is(err, glearn.ErrInvalidParameter) {
			t.Errorf("expected ErrInvalidParameter, got %v", err)
		}
	})

	t.Run("estimator not ParamSetter", func(t *testing.T) {
		grid := GridSearchCVConfig{
			Estimator: mockEstimator{}, // does not implement ParamSetter
			ParamGrid: []map[string]any{{"Constant": 1.0}},
			CV:        3,
			Scorer:    negMSE,
		}
		_, err := grid.Fit(t.Context(), X, y)
		if !errors.Is(err, glearn.ErrInvalidParameter) {
			t.Errorf("expected ErrInvalidParameter, got %v", err)
		}
	})
}

func TestGridSearchCV_UnknownParam(t *testing.T) {
	X := mat.NewDense(20, 2, nil)
	y := make([]float64, 20)

	grid := GridSearchCVConfig{
		Estimator: &paramEstimator{},
		ParamGrid: []map[string]any{{"UnknownParam": 1.0}},
		CV:        3,
		Scorer:    negMSE,
		Seed:      42,
	}

	_, err := grid.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for unknown parameter, got nil")
	}
}

func TestGridSearchCV_BestModelPredict(t *testing.T) {
	nSamples := 40
	X := mat.NewDense(nSamples, 2, nil)
	y := make([]float64, nSamples)
	for i := range nSamples {
		X.Set(i, 0, float64(i))
		X.Set(i, 1, float64(i))
		y[i] = 7.0
	}

	est := &paramEstimator{Constant: 1.0}
	grid := GridSearchCVConfig{
		Estimator: est,
		ParamGrid: []map[string]any{
			{"Constant": 1.0},
			{"Constant": 0.5},
		},
		CV:     2,
		Scorer: negMSE,
		Seed:   42,
	}

	result, err := grid.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Best model should predict correctly.
	testX := mat.NewDense(3, 2, nil)
	preds, err := result.BestModel.Predict(testX)
	if err != nil {
		t.Fatalf("predict error: %v", err)
	}
	if len(preds) != 3 {
		t.Fatalf("expected 3 predictions, got %d", len(preds))
	}
}

func TestGridSearchCV_ContextCancellation(t *testing.T) {
	X := mat.NewDense(40, 2, nil)
	y := make([]float64, 40)

	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	grid := GridSearchCVConfig{
		Estimator: &paramEstimator{},
		ParamGrid: []map[string]any{{"Constant": 1.0}},
		CV:        3,
		Scorer:    negMSE,
		Seed:      42,
	}

	_, err := grid.Fit(ctx, X, y)
	if err == nil {
		t.Fatal("expected error for cancelled context, got nil")
	}
}
