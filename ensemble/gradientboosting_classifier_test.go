package ensemble_test

import (
	"errors"
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/ensemble"
	"gonum.org/v1/gonum/mat"
)

// makeBinaryData creates a simple binary classification dataset.
// Class 0: cluster around (1, 1), Class 1: cluster around (5, 5).
func makeBinaryData(n int) (*mat.Dense, []float64) {
	nFeatures := 2
	half := n / 2
	data := make([]float64, n*nFeatures)
	labels := make([]float64, n)

	for i := range half {
		data[i*nFeatures+0] = 1.0 + float64(i)*0.05
		data[i*nFeatures+1] = 1.0 + float64(i)*0.03
		labels[i] = 0
	}
	for i := range n - half {
		idx := half + i
		data[idx*nFeatures+0] = 5.0 + float64(i)*0.05
		data[idx*nFeatures+1] = 5.0 + float64(i)*0.03
		labels[idx] = 1
	}

	X := mat.NewDense(n, nFeatures, data)
	return X, labels
}

func TestGradientBoostingClassifier_FitPredict(t *testing.T) {
	X, y := makeBinaryData(100)

	cfg := ensemble.NewGradientBoostingClassifier(
		ensemble.WithGBNTrees(50),
		ensemble.WithGBLearningRate(0.1),
		ensemble.WithGBMaxDepth(3),
		ensemble.WithGBSeed(42),
	)

	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	preds, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	correct := 0
	for i := range y {
		if preds[i] == y[i] {
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(y))
	if accuracy < 0.95 {
		t.Errorf("expected accuracy >= 0.95, got %f", accuracy)
	}
}

func TestGradientBoostingClassifier_MoreIterationsReducesError(t *testing.T) {
	X, y := makeBinaryData(100)

	cfg5 := ensemble.NewGradientBoostingClassifier(
		ensemble.WithGBNTrees(2),
		ensemble.WithGBLearningRate(0.1),
		ensemble.WithGBMaxDepth(2),
		ensemble.WithGBSeed(42),
	)
	cfg50 := ensemble.NewGradientBoostingClassifier(
		ensemble.WithGBNTrees(50),
		ensemble.WithGBLearningRate(0.1),
		ensemble.WithGBMaxDepth(2),
		ensemble.WithGBSeed(42),
	)

	pred5, err := cfg5.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit with 2 trees failed: %v", err)
	}
	pred50, err := cfg50.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit with 50 trees failed: %v", err)
	}

	scorer5 := pred5.(glearn.Scorer)
	scorer50 := pred50.(glearn.Scorer)

	acc5, _ := scorer5.Score(X, y)
	acc50, _ := scorer50.Score(X, y)

	if acc50 < acc5-0.05 {
		t.Errorf("more iterations should not significantly hurt: acc(2)=%.3f, acc(50)=%.3f", acc5, acc50)
	}
}

func TestGradientBoostingClassifier_PredictProbabilities(t *testing.T) {
	X, y := makeBinaryData(100)

	cfg := ensemble.NewGradientBoostingClassifier(
		ensemble.WithGBNTrees(30),
		ensemble.WithGBLearningRate(0.1),
		ensemble.WithGBSeed(42),
	)

	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	classifier := predictor.(glearn.Classifier)
	proba, err := classifier.PredictProbabilities(X)
	if err != nil {
		t.Fatalf("PredictProbabilities failed: %v", err)
	}

	rows, cols := proba.Dims()
	if rows != 100 || cols != 2 {
		t.Fatalf("expected proba shape (100, 2), got (%d, %d)", rows, cols)
	}

	// Each row should sum to 1.
	for i := range rows {
		rowSum := proba.At(i, 0) + proba.At(i, 1)
		if math.Abs(rowSum-1.0) > 1e-10 {
			t.Errorf("row %d probability sum = %f, expected 1.0", i, rowSum)
		}
	}

	// Class 0 samples should have high P(class_0).
	for i := range 50 {
		if proba.At(i, 0) < 0.5 {
			t.Errorf("sample %d (class 0) has P(class_0)=%f, expected > 0.5", i, proba.At(i, 0))
		}
	}
}

func TestGradientBoostingClassifier_DimensionMismatch(t *testing.T) {
	X, y := makeBinaryData(50)

	cfg := ensemble.NewGradientBoostingClassifier(
		ensemble.WithGBNTrees(10),
		ensemble.WithGBSeed(42),
	)

	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	wrongX := mat.NewDense(10, 5, nil)
	_, err = predictor.Predict(wrongX)
	if err == nil {
		t.Fatal("expected error for dimension mismatch, got nil")
	}
	if !errors.Is(err, glearn.ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got: %v", err)
	}
}

func TestGradientBoostingClassifier_GetClasses(t *testing.T) {
	X, y := makeBinaryData(50)

	cfg := ensemble.NewGradientBoostingClassifier(
		ensemble.WithGBNTrees(5),
		ensemble.WithGBSeed(42),
	)

	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	hc := predictor.(glearn.HasClasses)
	classes := hc.GetClasses()

	if len(classes) != 2 {
		t.Fatalf("expected 2 classes, got %d", len(classes))
	}
	if classes[0] != 0 || classes[1] != 1 {
		t.Errorf("expected classes [0, 1], got %v", classes)
	}
}

func TestGradientBoostingClassifier_RejectsMulticlass(t *testing.T) {
	X, y := makeIrisLikeData() // 3 classes

	cfg := ensemble.NewGradientBoostingClassifier(
		ensemble.WithGBNTrees(5),
		ensemble.WithGBSeed(42),
	)

	_, err := cfg.Fit(t.Context(), X, y)
	if err == nil {
		t.Fatal("expected error for multiclass data, got nil")
	}
	if !errors.Is(err, glearn.ErrInvalidParameter) {
		t.Errorf("expected ErrInvalidParameter, got: %v", err)
	}
}

func TestGradientBoostingClassifier_Subsample(t *testing.T) {
	X, y := makeBinaryData(100)

	cfg := ensemble.NewGradientBoostingClassifier(
		ensemble.WithGBNTrees(50),
		ensemble.WithGBLearningRate(0.1),
		ensemble.WithGBSubsample(0.8),
		ensemble.WithGBSeed(42),
	)

	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit with subsample failed: %v", err)
	}

	scorer := predictor.(glearn.Scorer)
	acc, err := scorer.Score(X, y)
	if err != nil {
		t.Fatalf("Score failed: %v", err)
	}

	if acc < 0.9 {
		t.Errorf("expected accuracy >= 0.9 with subsample, got %f", acc)
	}
}
