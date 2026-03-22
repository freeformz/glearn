package pipeline

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

// testTransformer is a simple transformer that scales each element by a factor.
type testTransformerConfig struct {
	Scale float64
}

func (t testTransformerConfig) Fit(_ context.Context, X *mat.Dense) (glearn.FittedTransformer, error) {
	_, nFeatures := X.Dims()
	return &testFittedTransformer{Scale: t.Scale, NFeatures: nFeatures}, nil
}

type testFittedTransformer struct {
	Scale     float64
	NFeatures int
}

func (t *testFittedTransformer) Transform(X *mat.Dense) (*mat.Dense, error) {
	nSamples, nFeatures := X.Dims()
	if nFeatures != t.NFeatures {
		return nil, glearn.ErrDimensionMismatch
	}
	result := mat.NewDense(nSamples, nFeatures, nil)
	for i := range nSamples {
		for j := range nFeatures {
			result.Set(i, j, X.At(i, j)*t.Scale)
		}
	}
	return result, nil
}

// testFitTransformerConfig implements both Transformer and FitTransformer.
type testFitTransformerConfig struct {
	Scale float64
}

func (t testFitTransformerConfig) Fit(_ context.Context, X *mat.Dense) (glearn.FittedTransformer, error) {
	_, nFeatures := X.Dims()
	return &testFittedTransformer{Scale: t.Scale, NFeatures: nFeatures}, nil
}

func (t testFitTransformerConfig) FitTransform(_ context.Context, X *mat.Dense) (glearn.FittedTransformer, *mat.Dense, error) {
	nSamples, nFeatures := X.Dims()
	ft := &testFittedTransformer{Scale: t.Scale, NFeatures: nFeatures}
	result := mat.NewDense(nSamples, nFeatures, nil)
	for i := range nSamples {
		for j := range nFeatures {
			result.Set(i, j, X.At(i, j)*t.Scale)
		}
	}
	return ft, result, nil
}

// testEstimator is a simple estimator that learns the mean of y.
type testEstimator struct{}

func (te testEstimator) Fit(_ context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures := X.Dims()
	if len(y) != nSamples {
		return nil, glearn.ErrDimensionMismatch
	}
	sum := 0.0
	for _, v := range y {
		sum += v
	}
	return &testPredictor{meanPred: sum / float64(nSamples), nFeatures: nFeatures}, nil
}

type testPredictor struct {
	meanPred  float64
	nFeatures int
}

func (tp *testPredictor) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, nFeatures := X.Dims()
	if nFeatures != tp.nFeatures {
		return nil, glearn.ErrDimensionMismatch
	}
	preds := make([]float64, nSamples)
	for i := range nSamples {
		preds[i] = tp.meanPred
	}
	return preds, nil
}

// testScoringPredictor also implements Scorer.
type testScoringPredictor struct {
	testPredictor
}

func (sp *testScoringPredictor) Score(X *mat.Dense, y []float64) (float64, error) {
	preds, err := sp.Predict(X)
	if err != nil {
		return 0, err
	}
	return r2Score(y, preds), nil
}

// testScoringEstimator returns a testScoringPredictor.
type testScoringEstimator struct{}

func (te testScoringEstimator) Fit(_ context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures := X.Dims()
	if len(y) != nSamples {
		return nil, glearn.ErrDimensionMismatch
	}
	sum := 0.0
	for _, v := range y {
		sum += v
	}
	return &testScoringPredictor{
		testPredictor: testPredictor{meanPred: sum / float64(nSamples), nFeatures: nFeatures},
	}, nil
}

func TestPipeline_TransformerAndEstimator(t *testing.T) {
	X := mat.NewDense(10, 2, nil)
	y := make([]float64, 10)
	for i := range 10 {
		X.Set(i, 0, float64(i))
		X.Set(i, 1, float64(i*2))
		y[i] = float64(i * 3)
	}

	pipe := New(
		TransformStep("scale", testTransformerConfig{Scale: 2.0}),
		EstimatorStep("predict", testEstimator{}),
	)

	fitted, err := pipe.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(fitted.Transformers) != 1 {
		t.Errorf("expected 1 fitted transformer, got %d", len(fitted.Transformers))
	}
	if fitted.Predictor == nil {
		t.Error("predictor is nil")
	}

	preds, err := fitted.Predict(X)
	if err != nil {
		t.Fatalf("predict error: %v", err)
	}
	if len(preds) != 10 {
		t.Errorf("expected 10 predictions, got %d", len(preds))
	}
}

func TestPipeline_MultipleTransformers(t *testing.T) {
	X := mat.NewDense(5, 3, nil)
	y := make([]float64, 5)
	for i := range 5 {
		for j := range 3 {
			X.Set(i, j, float64(i+j+1))
		}
		y[i] = float64(i + 1)
	}

	pipe := New(
		TransformStep("scale1", testTransformerConfig{Scale: 2.0}),
		TransformStep("scale2", testTransformerConfig{Scale: 3.0}),
		EstimatorStep("predict", testEstimator{}),
	)

	fitted, err := pipe.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(fitted.Transformers) != 2 {
		t.Errorf("expected 2 fitted transformers, got %d", len(fitted.Transformers))
	}
}

func TestPipeline_FitTransformerOptimization(t *testing.T) {
	// Use a FitTransformer to verify the optimization path works.
	X := mat.NewDense(5, 2, nil)
	y := make([]float64, 5)
	for i := range 5 {
		X.Set(i, 0, float64(i))
		X.Set(i, 1, float64(i))
		y[i] = float64(i)
	}

	pipe := New(
		TransformStep("ft", testFitTransformerConfig{Scale: 2.0}),
		EstimatorStep("predict", testEstimator{}),
	)

	fitted, err := pipe.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	preds, err := fitted.Predict(X)
	if err != nil {
		t.Fatalf("predict error: %v", err)
	}
	if len(preds) != 5 {
		t.Errorf("expected 5 predictions, got %d", len(preds))
	}
}

func TestPipeline_Predict(t *testing.T) {
	// Verify that Predict applies the transforms.
	// With scale=2, X values are doubled before being passed to the estimator.
	X := mat.NewDense(4, 1, []float64{1, 2, 3, 4})
	y := []float64{10, 20, 30, 40}

	pipe := New(
		TransformStep("scale", testTransformerConfig{Scale: 2.0}),
		EstimatorStep("predict", testEstimator{}),
	)

	fitted, err := pipe.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("fit error: %v", err)
	}

	// The estimator was trained on transformed data (scale*X) with y,
	// so it learned meanPred = mean(y) = 25.
	testX := mat.NewDense(2, 1, []float64{5, 6})
	preds, err := fitted.Predict(testX)
	if err != nil {
		t.Fatalf("predict error: %v", err)
	}
	for i, p := range preds {
		if p != 25.0 {
			t.Errorf("prediction %d: expected 25.0, got %g", i, p)
		}
	}
}

func TestPipeline_Score(t *testing.T) {
	X := mat.NewDense(4, 1, []float64{1, 2, 3, 4})
	y := []float64{10, 10, 10, 10} // constant y

	pipe := New(
		TransformStep("scale", testTransformerConfig{Scale: 1.0}),
		EstimatorStep("predict", testScoringEstimator{}),
	)

	fitted, err := pipe.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("fit error: %v", err)
	}

	// With constant y=10 and predictor predicting mean=10, score should be perfect.
	score, err := fitted.Score(X, y)
	if err != nil {
		t.Fatalf("score error: %v", err)
	}
	if score != 1.0 {
		t.Errorf("expected R2 score 1.0, got %g", score)
	}
}

func TestPipeline_ScoreFallback(t *testing.T) {
	// Using testEstimator which does not implement Scorer — fallback R2.
	X := mat.NewDense(4, 1, []float64{1, 2, 3, 4})
	y := []float64{5, 5, 5, 5}

	pipe := New(
		TransformStep("scale", testTransformerConfig{Scale: 1.0}),
		EstimatorStep("predict", testEstimator{}),
	)

	fitted, err := pipe.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("fit error: %v", err)
	}

	score, err := fitted.Score(X, y)
	if err != nil {
		t.Fatalf("score error: %v", err)
	}
	// With constant y=5 and predictor predicting mean=5, R2 should be 1.
	if score != 1.0 {
		t.Errorf("expected R2 score 1.0, got %g", score)
	}
}

func TestPipeline_DimensionMismatch(t *testing.T) {
	X := mat.NewDense(5, 2, nil)
	y := make([]float64, 5)
	for i := range 5 {
		X.Set(i, 0, float64(i))
		X.Set(i, 1, float64(i))
		y[i] = float64(i)
	}

	pipe := New(
		TransformStep("scale", testTransformerConfig{Scale: 1.0}),
		EstimatorStep("predict", testEstimator{}),
	)

	fitted, err := pipe.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("fit error: %v", err)
	}

	// Predict with wrong number of features.
	wrongX := mat.NewDense(3, 5, nil)
	_, err = fitted.Predict(wrongX)
	if err == nil {
		t.Fatal("expected dimension mismatch error, got nil")
	}
}

func TestPipeline_NoSteps(t *testing.T) {
	pipe := New()
	X := mat.NewDense(5, 2, nil)
	y := make([]float64, 5)

	_, err := pipe.Fit(t.Context(), X, y)
	if !errors.Is(err, glearn.ErrInvalidParameter) {
		t.Errorf("expected ErrInvalidParameter for empty pipeline, got %v", err)
	}
}

func TestPipeline_MissingEstimator(t *testing.T) {
	// Last step is a transformer, not an estimator.
	pipe := New(
		TransformStep("scale", testTransformerConfig{Scale: 1.0}),
	)
	X := mat.NewDense(5, 2, nil)
	y := make([]float64, 5)

	_, err := pipe.Fit(t.Context(), X, y)
	if !errors.Is(err, glearn.ErrInvalidParameter) {
		t.Errorf("expected ErrInvalidParameter for missing estimator, got %v", err)
	}
}

func TestPipeline_IntermediateEstimator(t *testing.T) {
	// An estimator in an intermediate position should fail.
	pipe := Pipeline{
		Steps: []Step{
			{Name: "est", Estimator: testEstimator{}},
			{Name: "est2", Estimator: testEstimator{}},
		},
	}
	X := mat.NewDense(5, 2, nil)
	y := make([]float64, 5)

	_, err := pipe.Fit(t.Context(), X, y)
	if !errors.Is(err, glearn.ErrInvalidParameter) {
		t.Errorf("expected ErrInvalidParameter for intermediate estimator, got %v", err)
	}
}

func TestPipeline_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	X := mat.NewDense(5, 2, nil)
	y := make([]float64, 5)

	pipe := New(
		TransformStep("scale", testTransformerConfig{Scale: 1.0}),
		EstimatorStep("predict", testEstimator{}),
	)

	_, err := pipe.Fit(ctx, X, y)
	if err == nil {
		t.Fatal("expected error for cancelled context, got nil")
	}
}

func TestPipeline_OnlyEstimator(t *testing.T) {
	// Pipeline with just an estimator step (no transformers).
	X := mat.NewDense(5, 2, nil)
	y := make([]float64, 5)
	for i := range 5 {
		X.Set(i, 0, float64(i))
		X.Set(i, 1, float64(i))
		y[i] = float64(i)
	}

	pipe := New(
		EstimatorStep("predict", testEstimator{}),
	)

	fitted, err := pipe.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("fit error: %v", err)
	}

	if len(fitted.Transformers) != 0 {
		t.Errorf("expected 0 transformers, got %d", len(fitted.Transformers))
	}

	preds, err := fitted.Predict(X)
	if err != nil {
		t.Fatalf("predict error: %v", err)
	}
	if len(preds) != 5 {
		t.Errorf("expected 5 predictions, got %d", len(preds))
	}
}

func TestPipeline_TransformChainCorrectness(t *testing.T) {
	// Verify that transforms chain correctly: scale by 2 then scale by 3 = scale by 6.
	X := mat.NewDense(3, 1, []float64{1.0, 2.0, 3.0})
	y := []float64{6.0, 12.0, 18.0} // = X * 6

	pipe := New(
		TransformStep("scale2", testTransformerConfig{Scale: 2.0}),
		TransformStep("scale3", testTransformerConfig{Scale: 3.0}),
		EstimatorStep("predict", testEstimator{}),
	)

	fitted, err := pipe.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("fit error: %v", err)
	}

	// After scaling by 2 then 3, the estimator sees [6, 12, 18].
	// It learns mean(y) = 12.
	// When predicting, X=[1,2,3] is scaled to [6,12,18], and pred = 12.
	preds, err := fitted.Predict(X)
	if err != nil {
		t.Fatalf("predict error: %v", err)
	}
	for i, p := range preds {
		if math.Abs(p-12.0) > 1e-10 {
			t.Errorf("prediction %d: expected 12.0, got %g", i, p)
		}
	}
}
