package ensemble_test

import (
	"errors"
	"math"
	"testing"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/ensemble"
	"gonum.org/v1/gonum/mat"
)

// makeIrisLikeData creates a simple 3-class classification dataset.
// 150 samples, 4 features, 3 classes (0, 1, 2), 50 each.
func makeIrisLikeData() (*mat.Dense, []float64) {
	n := 150
	nFeatures := 4
	data := make([]float64, n*nFeatures)
	labels := make([]float64, n)

	// Class 0: cluster around (1, 1, 1, 1)
	for i := range 50 {
		data[i*nFeatures+0] = 1.0 + float64(i)*0.02
		data[i*nFeatures+1] = 1.0 + float64(i)*0.01
		data[i*nFeatures+2] = 1.0 + float64(i)*0.03
		data[i*nFeatures+3] = 1.0 + float64(i)*0.01
		labels[i] = 0
	}
	// Class 1: cluster around (5, 5, 5, 5)
	for i := range 50 {
		idx := 50 + i
		data[idx*nFeatures+0] = 5.0 + float64(i)*0.02
		data[idx*nFeatures+1] = 5.0 + float64(i)*0.01
		data[idx*nFeatures+2] = 5.0 + float64(i)*0.03
		data[idx*nFeatures+3] = 5.0 + float64(i)*0.01
		labels[idx] = 1
	}
	// Class 2: cluster around (9, 9, 9, 9)
	for i := range 50 {
		idx := 100 + i
		data[idx*nFeatures+0] = 9.0 + float64(i)*0.02
		data[idx*nFeatures+1] = 9.0 + float64(i)*0.01
		data[idx*nFeatures+2] = 9.0 + float64(i)*0.03
		data[idx*nFeatures+3] = 9.0 + float64(i)*0.01
		labels[idx] = 2
	}

	X := mat.NewDense(n, nFeatures, data)
	return X, labels
}

func TestRandomForestClassifier_FitPredict(t *testing.T) {
	X, y := makeIrisLikeData()

	cfg := ensemble.NewRandomForestClassifier(
		ensemble.WithNTrees(20),
		ensemble.WithMaxDepth(5),
		ensemble.WithSeed(42),
		ensemble.WithNJobs(2),
	)

	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	preds, err := predictor.Predict(X)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	// Training accuracy should be high on this well-separated dataset.
	correct := 0
	for i := range y {
		if preds[i] == y[i] {
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(y))
	if accuracy < 0.9 {
		t.Errorf("expected training accuracy >= 0.9, got %f", accuracy)
	}
}

func TestRandomForestClassifier_MoreTreesImproves(t *testing.T) {
	X, y := makeIrisLikeData()

	// Shallow trees to make the difference visible.
	cfg5 := ensemble.NewRandomForestClassifier(
		ensemble.WithNTrees(2),
		ensemble.WithMaxDepth(2),
		ensemble.WithSeed(42),
	)
	cfg50 := ensemble.NewRandomForestClassifier(
		ensemble.WithNTrees(50),
		ensemble.WithMaxDepth(2),
		ensemble.WithSeed(42),
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

	// More trees should generally not hurt accuracy.
	if acc50 < acc5-0.05 {
		t.Errorf("50 trees (acc=%.3f) significantly worse than 2 trees (acc=%.3f)", acc50, acc5)
	}
}

func TestRandomForestClassifier_FeatureImportances(t *testing.T) {
	X, y := makeIrisLikeData()

	cfg := ensemble.NewRandomForestClassifier(
		ensemble.WithNTrees(30),
		ensemble.WithSeed(42),
	)

	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	fi := predictor.(glearn.HasFeatureImportances)
	importances := fi.GetFeatureImportances()

	if len(importances) != 4 {
		t.Fatalf("expected 4 feature importances, got %d", len(importances))
	}

	sum := 0.0
	for _, v := range importances {
		sum += v
		if v < 0 {
			t.Errorf("feature importance should be non-negative, got %f", v)
		}
	}

	// Sum should be approximately 1 (allow tolerance for float imprecision).
	if math.Abs(sum-1.0) > 0.05 {
		t.Errorf("feature importances should sum to ~1.0, got %f", sum)
	}
}

func TestRandomForestClassifier_PredictProbabilities(t *testing.T) {
	X, y := makeIrisLikeData()

	cfg := ensemble.NewRandomForestClassifier(
		ensemble.WithNTrees(20),
		ensemble.WithSeed(42),
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
	if rows != 150 || cols != 3 {
		t.Fatalf("expected proba shape (150, 3), got (%d, %d)", rows, cols)
	}

	// Each row should sum to approximately 1.
	for i := range rows {
		rowSum := 0.0
		for j := range cols {
			rowSum += proba.At(i, j)
		}
		if math.Abs(rowSum-1.0) > 0.01 {
			t.Errorf("row %d probability sum = %f, expected ~1.0", i, rowSum)
		}
	}
}

func TestRandomForestClassifier_DimensionMismatch(t *testing.T) {
	X, y := makeIrisLikeData()

	cfg := ensemble.NewRandomForestClassifier(
		ensemble.WithNTrees(5),
		ensemble.WithSeed(42),
	)

	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Predict with wrong number of features.
	wrongX := mat.NewDense(10, 3, nil)
	_, err = predictor.Predict(wrongX)
	if err == nil {
		t.Fatal("expected error for dimension mismatch, got nil")
	}
	if !errors.Is(err, glearn.ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got: %v", err)
	}
}

func TestRandomForestClassifier_GetClasses(t *testing.T) {
	X, y := makeIrisLikeData()

	cfg := ensemble.NewRandomForestClassifier(
		ensemble.WithNTrees(5),
		ensemble.WithSeed(42),
	)

	predictor, err := cfg.Fit(t.Context(), X, y)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	hc := predictor.(glearn.HasClasses)
	classes := hc.GetClasses()

	if len(classes) != 3 {
		t.Fatalf("expected 3 classes, got %d", len(classes))
	}
	expected := []float64{0, 1, 2}
	for i := range expected {
		if classes[i] != expected[i] {
			t.Errorf("class[%d] = %f, expected %f", i, classes[i], expected[i])
		}
	}
}
