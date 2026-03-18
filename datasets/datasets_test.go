package datasets

import (
	"math"
	"testing"
)

// --- Iris ---

func TestLoadIris(t *testing.T) {
	X, y, err := LoadIris()
	if err != nil {
		t.Fatalf("LoadIris() returned error: %v", err)
	}

	rows, cols := X.Dims()
	if rows != 150 {
		t.Errorf("expected 150 samples, got %d", rows)
	}
	if cols != 4 {
		t.Errorf("expected 4 features, got %d", cols)
	}
	if len(y) != 150 {
		t.Errorf("expected 150 targets, got %d", len(y))
	}

	// Check class distribution: 50 of each class.
	classCounts := make(map[float64]int)
	for _, v := range y {
		classCounts[v]++
	}
	if classCounts[0] != 50 {
		t.Errorf("expected 50 class-0 samples, got %d", classCounts[0])
	}
	if classCounts[1] != 50 {
		t.Errorf("expected 50 class-1 samples, got %d", classCounts[1])
	}
	if classCounts[2] != 50 {
		t.Errorf("expected 50 class-2 samples, got %d", classCounts[2])
	}

	// Check feature value ranges (sepal length should be between ~4 and ~8).
	for i := range rows {
		sl := X.At(i, 0)
		if sl < 4.0 || sl > 8.0 {
			t.Errorf("sample %d: sepal length %f out of expected range [4, 8]", i, sl)
		}
	}

	// Verify first sample values.
	expectedFirst := []float64{5.1, 3.5, 1.4, 0.2}
	for j, want := range expectedFirst {
		got := X.At(0, j)
		if math.Abs(got-want) > 1e-10 {
			t.Errorf("X[0][%d] = %f, want %f", j, got, want)
		}
	}

	// Verify the returned data is a copy (mutation safety).
	X.Set(0, 0, -999.0)
	X2, _, err := LoadIris()
	if err != nil {
		t.Fatalf("second LoadIris() returned error: %v", err)
	}
	if X2.At(0, 0) == -999.0 {
		t.Error("LoadIris() data is not a copy; mutation leaked to subsequent calls")
	}
}

// --- Diabetes ---

func TestLoadDiabetes(t *testing.T) {
	X, y, err := LoadDiabetes()
	if err != nil {
		t.Fatalf("LoadDiabetes() returned error: %v", err)
	}

	rows, cols := X.Dims()
	if rows != 442 {
		t.Errorf("expected 442 samples, got %d", rows)
	}
	if cols != 10 {
		t.Errorf("expected 10 features, got %d", cols)
	}
	if len(y) != 442 {
		t.Errorf("expected 442 targets, got %d", len(y))
	}

	// Diabetes features are standardized; check that they are in a reasonable range.
	for i := range rows {
		for j := range cols {
			v := X.At(i, j)
			if v < -0.2 || v > 0.2 {
				t.Errorf("sample %d feature %d: value %f out of expected standardized range [-0.2, 0.2]", i, j, v)
			}
		}
	}

	// Target values should be positive (disease progression measure).
	for i, v := range y {
		if v < 25 || v > 350 {
			t.Errorf("target[%d] = %f, expected in range [25, 350]", i, v)
		}
	}

	// Verify first sample's first feature.
	got := X.At(0, 0)
	want := 0.03807590643
	if math.Abs(got-want) > 1e-8 {
		t.Errorf("X[0][0] = %v, want %v", got, want)
	}
}

// --- Wine ---

func TestLoadWine(t *testing.T) {
	X, y, err := LoadWine()
	if err != nil {
		t.Fatalf("LoadWine() returned error: %v", err)
	}

	rows, cols := X.Dims()
	if rows != 178 {
		t.Errorf("expected 178 samples, got %d", rows)
	}
	if cols != 13 {
		t.Errorf("expected 13 features, got %d", cols)
	}
	if len(y) != 178 {
		t.Errorf("expected 178 targets, got %d", len(y))
	}

	// Check class distribution: 59 class-0, 71 class-1, 48 class-2.
	classCounts := make(map[float64]int)
	for _, v := range y {
		classCounts[v]++
	}
	if classCounts[0] != 59 {
		t.Errorf("expected 59 class-0 samples, got %d", classCounts[0])
	}
	if classCounts[1] != 71 {
		t.Errorf("expected 71 class-1 samples, got %d", classCounts[1])
	}
	if classCounts[2] != 48 {
		t.Errorf("expected 48 class-2 samples, got %d", classCounts[2])
	}

	// Check that alcohol (feature 0) is in a reasonable range.
	for i := range rows {
		alc := X.At(i, 0)
		if alc < 11.0 || alc > 15.0 {
			t.Errorf("sample %d: alcohol %f out of expected range [11, 15]", i, alc)
		}
	}

	// Verify first sample values.
	expectedFirst := []float64{14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0}
	for j, want := range expectedFirst {
		got := X.At(0, j)
		if math.Abs(got-want) > 1e-10 {
			t.Errorf("X[0][%d] = %f, want %f", j, got, want)
		}
	}
}

// --- MakeBlobs ---

func TestMakeBlobs(t *testing.T) {
	X, y, err := MakeBlobs(
		WithBlobsNSamples(300),
		WithBlobsNFeatures(3),
		WithBlobsNClusters(4),
		WithBlobsClusterStd(0.5),
		WithBlobsSeed(42),
	)
	if err != nil {
		t.Fatalf("MakeBlobs() returned error: %v", err)
	}

	rows, cols := X.Dims()
	if rows != 300 {
		t.Errorf("expected 300 samples, got %d", rows)
	}
	if cols != 3 {
		t.Errorf("expected 3 features, got %d", cols)
	}
	if len(y) != 300 {
		t.Errorf("expected 300 targets, got %d", len(y))
	}

	// Check that all classes are represented.
	classCounts := make(map[float64]int)
	for _, v := range y {
		classCounts[v]++
	}
	if len(classCounts) != 4 {
		t.Errorf("expected 4 unique classes, got %d", len(classCounts))
	}
	for cls, count := range classCounts {
		if count != 75 {
			t.Errorf("class %v: expected 75 samples, got %d", cls, count)
		}
	}
}

func TestMakeBlobsReproducibility(t *testing.T) {
	X1, y1, err := MakeBlobs(WithBlobsSeed(123))
	if err != nil {
		t.Fatalf("MakeBlobs() returned error: %v", err)
	}
	X2, y2, err := MakeBlobs(WithBlobsSeed(123))
	if err != nil {
		t.Fatalf("MakeBlobs() returned error: %v", err)
	}

	rows, cols := X1.Dims()
	for i := range rows {
		if y1[i] != y2[i] {
			t.Fatalf("targets differ at index %d: %f vs %f", i, y1[i], y2[i])
		}
		for j := range cols {
			if X1.At(i, j) != X2.At(i, j) {
				t.Fatalf("features differ at [%d][%d]: %f vs %f", i, j, X1.At(i, j), X2.At(i, j))
			}
		}
	}
}

func TestMakeBlobsValidation(t *testing.T) {
	tests := []struct {
		name string
		opts []BlobsOption
	}{
		{"zero samples", []BlobsOption{WithBlobsNSamples(0)}},
		{"negative features", []BlobsOption{WithBlobsNFeatures(-1)}},
		{"zero clusters", []BlobsOption{WithBlobsNClusters(0)}},
		{"negative std", []BlobsOption{WithBlobsClusterStd(-1.0)}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := MakeBlobs(tt.opts...)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

// --- MakeClassification ---

func TestMakeClassification(t *testing.T) {
	X, y, err := MakeClassification(
		WithClassificationNSamples(200),
		WithClassificationNFeatures(10),
		WithClassificationNInformative(5),
		WithClassificationNRedundant(3),
		WithClassificationNClasses(3),
		WithClassificationSeed(42),
	)
	if err != nil {
		t.Fatalf("MakeClassification() returned error: %v", err)
	}

	rows, cols := X.Dims()
	if rows != 200 {
		t.Errorf("expected 200 samples, got %d", rows)
	}
	if cols != 10 {
		t.Errorf("expected 10 features, got %d", cols)
	}
	if len(y) != 200 {
		t.Errorf("expected 200 targets, got %d", len(y))
	}

	// Check that all 3 classes are present.
	classCounts := make(map[float64]int)
	for _, v := range y {
		classCounts[v]++
	}
	if len(classCounts) != 3 {
		t.Errorf("expected 3 unique classes, got %d: %v", len(classCounts), classCounts)
	}
}

func TestMakeClassificationReproducibility(t *testing.T) {
	X1, y1, err := MakeClassification(WithClassificationSeed(99))
	if err != nil {
		t.Fatalf("MakeClassification() returned error: %v", err)
	}
	X2, y2, err := MakeClassification(WithClassificationSeed(99))
	if err != nil {
		t.Fatalf("MakeClassification() returned error: %v", err)
	}

	rows, cols := X1.Dims()
	for i := range rows {
		if y1[i] != y2[i] {
			t.Fatalf("targets differ at index %d: %f vs %f", i, y1[i], y2[i])
		}
		for j := range cols {
			if X1.At(i, j) != X2.At(i, j) {
				t.Fatalf("features differ at [%d][%d]: %f vs %f", i, j, X1.At(i, j), X2.At(i, j))
			}
		}
	}
}

func TestMakeClassificationValidation(t *testing.T) {
	tests := []struct {
		name string
		opts []ClassificationOption
	}{
		{"zero samples", []ClassificationOption{WithClassificationNSamples(0)}},
		{"zero features", []ClassificationOption{WithClassificationNFeatures(0)}},
		{"zero informative", []ClassificationOption{WithClassificationNInformative(0)}},
		{
			"informative > features",
			[]ClassificationOption{
				WithClassificationNFeatures(5),
				WithClassificationNInformative(10),
			},
		},
		{
			"informative + redundant > features",
			[]ClassificationOption{
				WithClassificationNFeatures(5),
				WithClassificationNInformative(3),
				WithClassificationNRedundant(4),
			},
		},
		{"zero classes", []ClassificationOption{WithClassificationNClasses(0)}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := MakeClassification(tt.opts...)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

// --- MakeRegression ---

func TestMakeRegression(t *testing.T) {
	X, y, err := MakeRegression(
		WithRegressionNSamples(150),
		WithRegressionNFeatures(20),
		WithRegressionNInformative(5),
		WithRegressionNoise(0.1),
		WithRegressionBias(1.0),
		WithRegressionSeed(42),
	)
	if err != nil {
		t.Fatalf("MakeRegression() returned error: %v", err)
	}

	rows, cols := X.Dims()
	if rows != 150 {
		t.Errorf("expected 150 samples, got %d", rows)
	}
	if cols != 20 {
		t.Errorf("expected 20 features, got %d", cols)
	}
	if len(y) != 150 {
		t.Errorf("expected 150 targets, got %d", len(y))
	}

	// Target should have non-trivial variance.
	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= float64(len(y))

	variance := 0.0
	for _, v := range y {
		d := v - mean
		variance += d * d
	}
	variance /= float64(len(y))

	if variance < 0.01 {
		t.Errorf("target variance %f is suspiciously low", variance)
	}
}

func TestMakeRegressionNoiseless(t *testing.T) {
	// Without noise, the same seed should produce identical results.
	X1, y1, err := MakeRegression(
		WithRegressionNSamples(50),
		WithRegressionNFeatures(5),
		WithRegressionNInformative(3),
		WithRegressionSeed(7),
	)
	if err != nil {
		t.Fatalf("MakeRegression() returned error: %v", err)
	}
	X2, y2, err := MakeRegression(
		WithRegressionNSamples(50),
		WithRegressionNFeatures(5),
		WithRegressionNInformative(3),
		WithRegressionSeed(7),
	)
	if err != nil {
		t.Fatalf("MakeRegression() returned error: %v", err)
	}

	rows, cols := X1.Dims()
	for i := range rows {
		if y1[i] != y2[i] {
			t.Fatalf("targets differ at index %d: %f vs %f", i, y1[i], y2[i])
		}
		for j := range cols {
			if X1.At(i, j) != X2.At(i, j) {
				t.Fatalf("features differ at [%d][%d]: %f vs %f", i, j, X1.At(i, j), X2.At(i, j))
			}
		}
	}
}

func TestMakeRegressionValidation(t *testing.T) {
	tests := []struct {
		name string
		opts []RegressionOption
	}{
		{"zero samples", []RegressionOption{WithRegressionNSamples(0)}},
		{"zero features", []RegressionOption{WithRegressionNFeatures(0)}},
		{"zero informative", []RegressionOption{WithRegressionNInformative(0)}},
		{
			"informative > features",
			[]RegressionOption{
				WithRegressionNFeatures(5),
				WithRegressionNInformative(10),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := MakeRegression(tt.opts...)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

// --- MakeRegression bias test ---

func TestMakeRegressionBias(t *testing.T) {
	// With zero noise and a known bias, the mean of y should be approximately the bias.
	X, y, err := MakeRegression(
		WithRegressionNSamples(10000),
		WithRegressionNFeatures(1),
		WithRegressionNInformative(1),
		WithRegressionBias(5.0),
		WithRegressionSeed(42),
	)
	if err != nil {
		t.Fatalf("MakeRegression() returned error: %v", err)
	}

	rows, _ := X.Dims()
	mean := 0.0
	for _, v := range y {
		mean += v
	}
	mean /= float64(rows)

	// The mean of y should be close to the bias (5.0) since features are zero-mean.
	if math.Abs(mean-5.0) > 0.5 {
		t.Errorf("mean of targets = %f, expected close to 5.0 (bias)", mean)
	}
}
