package modelselection

import (
	"errors"
	"sort"
	"testing"

	"github.com/freeformz/glearn"
)

func TestKFold_AllSamplesAppearOnce(t *testing.T) {
	nSamples := 100
	kf := KFold{NSplits: 5, Shuffle: false}

	folds, err := kf.Split(nSamples)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(folds) != 5 {
		t.Fatalf("expected 5 folds, got %d", len(folds))
	}

	// Collect all test indices and verify each appears exactly once.
	testCount := make(map[int]int)
	for _, fold := range folds {
		for _, idx := range fold.TestIndices {
			testCount[idx]++
		}
	}

	if len(testCount) != nSamples {
		t.Errorf("expected %d unique test indices, got %d", nSamples, len(testCount))
	}
	for idx, count := range testCount {
		if count != 1 {
			t.Errorf("index %d appeared %d times in test folds, expected 1", idx, count)
		}
	}
}

func TestKFold_TrainTestDisjoint(t *testing.T) {
	nSamples := 50
	kf := KFold{NSplits: 5, Shuffle: true, Seed: 42}

	folds, err := kf.Split(nSamples)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for i, fold := range folds {
		trainSet := make(map[int]bool)
		for _, idx := range fold.TrainIndices {
			trainSet[idx] = true
		}
		for _, idx := range fold.TestIndices {
			if trainSet[idx] {
				t.Errorf("fold %d: index %d appears in both train and test", i, idx)
			}
		}
		// Total should be nSamples.
		if len(fold.TrainIndices)+len(fold.TestIndices) != nSamples {
			t.Errorf("fold %d: train (%d) + test (%d) != %d",
				i, len(fold.TrainIndices), len(fold.TestIndices), nSamples)
		}
	}
}

func TestKFold_SplitCount(t *testing.T) {
	for _, nSplits := range []int{2, 3, 5, 10} {
		kf := KFold{NSplits: nSplits, Shuffle: false}
		folds, err := kf.Split(100)
		if err != nil {
			t.Fatalf("unexpected error for NSplits=%d: %v", nSplits, err)
		}
		if len(folds) != nSplits {
			t.Errorf("NSplits=%d: expected %d folds, got %d", nSplits, nSplits, len(folds))
		}
	}
}

func TestKFold_UnevenSplits(t *testing.T) {
	// 10 samples into 3 folds: folds should have sizes 4, 3, 3 or similar.
	kf := KFold{NSplits: 3, Shuffle: false}
	folds, err := kf.Split(10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	totalTest := 0
	for _, fold := range folds {
		totalTest += len(fold.TestIndices)
	}
	if totalTest != 10 {
		t.Errorf("expected 10 total test indices, got %d", totalTest)
	}

	// Check fold sizes are balanced (differ by at most 1).
	sizes := make([]int, len(folds))
	for i, fold := range folds {
		sizes[i] = len(fold.TestIndices)
	}
	sort.Ints(sizes)
	if sizes[len(sizes)-1]-sizes[0] > 1 {
		t.Errorf("fold sizes are unbalanced: %v", sizes)
	}
}

func TestKFold_InvalidParams(t *testing.T) {
	tests := []struct {
		name     string
		nSplits  int
		nSamples int
		wantErr  error
	}{
		{"NSplits < 2", 1, 10, glearn.ErrInvalidParameter},
		{"NSplits > nSamples", 11, 10, glearn.ErrInvalidParameter},
		{"NSplits = 0", 0, 10, glearn.ErrInvalidParameter},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kf := KFold{NSplits: tt.nSplits}
			_, err := kf.Split(tt.nSamples)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("expected error wrapping %v, got %v", tt.wantErr, err)
			}
		})
	}
}

func TestKFold_Shuffle(t *testing.T) {
	kf1 := KFold{NSplits: 3, Shuffle: true, Seed: 42}
	kf2 := KFold{NSplits: 3, Shuffle: false}

	folds1, err := kf1.Split(30)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	folds2, err := kf2.Split(30)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Shuffled and unshuffled folds should generally differ.
	different := false
	for i := range folds1 {
		if len(folds1[i].TestIndices) != len(folds2[i].TestIndices) {
			different = true
			break
		}
		for j := range folds1[i].TestIndices {
			if folds1[i].TestIndices[j] != folds2[i].TestIndices[j] {
				different = true
				break
			}
		}
		if different {
			break
		}
	}
	if !different {
		t.Error("shuffled and unshuffled folds are identical — shuffle may not be working")
	}
}

func TestStratifiedKFold_ClassDistribution(t *testing.T) {
	// 60 samples: 40 class 0, 20 class 1 => ratio 2:1.
	y := make([]float64, 60)
	for i := 40; i < 60; i++ {
		y[i] = 1.0
	}

	skf := StratifiedKFold{NSplits: 3, Shuffle: false}
	folds, err := skf.Split(y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(folds) != 3 {
		t.Fatalf("expected 3 folds, got %d", len(folds))
	}

	for i, fold := range folds {
		class0Count := 0
		class1Count := 0
		for _, idx := range fold.TestIndices {
			if y[idx] == 0 {
				class0Count++
			} else {
				class1Count++
			}
		}
		// Each test fold should have approximately 2:1 ratio.
		// With 60 samples and 3 folds, each fold gets ~20 test samples.
		// Expect ~13-14 class 0 and ~6-7 class 1 per fold.
		if class0Count == 0 || class1Count == 0 {
			t.Errorf("fold %d: missing a class in test set (class0=%d, class1=%d)",
				i, class0Count, class1Count)
		}

		// Check approximate ratio (within a factor of 1.5 of expected).
		ratio := float64(class0Count) / float64(class1Count)
		if ratio < 1.5 || ratio > 2.5 {
			t.Errorf("fold %d: class ratio %g is outside expected range [1.5, 2.5] (class0=%d, class1=%d)",
				i, ratio, class0Count, class1Count)
		}
	}
}

func TestStratifiedKFold_AllSamplesAppearOnce(t *testing.T) {
	y := make([]float64, 90)
	for i := range 90 {
		y[i] = float64(i % 3) // 3 classes
	}

	skf := StratifiedKFold{NSplits: 5, Shuffle: true, Seed: 99}
	folds, err := skf.Split(y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// All test indices should cover every sample exactly once.
	testCount := make(map[int]int)
	for _, fold := range folds {
		for _, idx := range fold.TestIndices {
			testCount[idx]++
		}
	}
	if len(testCount) != 90 {
		t.Errorf("expected 90 unique test indices, got %d", len(testCount))
	}
	for idx, count := range testCount {
		if count != 1 {
			t.Errorf("index %d appeared %d times in test folds, expected 1", idx, count)
		}
	}
}

func TestStratifiedKFold_TrainTestDisjoint(t *testing.T) {
	y := make([]float64, 40)
	for i := range 40 {
		y[i] = float64(i % 2)
	}

	skf := StratifiedKFold{NSplits: 4, Shuffle: false}
	folds, err := skf.Split(y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for i, fold := range folds {
		trainSet := make(map[int]bool)
		for _, idx := range fold.TrainIndices {
			trainSet[idx] = true
		}
		for _, idx := range fold.TestIndices {
			if trainSet[idx] {
				t.Errorf("fold %d: index %d appears in both train and test", i, idx)
			}
		}
		if len(fold.TrainIndices)+len(fold.TestIndices) != 40 {
			t.Errorf("fold %d: train (%d) + test (%d) != 40",
				i, len(fold.TrainIndices), len(fold.TestIndices))
		}
	}
}

func TestStratifiedKFold_InvalidParams(t *testing.T) {
	t.Run("NSplits < 2", func(t *testing.T) {
		skf := StratifiedKFold{NSplits: 1}
		_, err := skf.Split([]float64{0, 0, 1, 1})
		if !errors.Is(err, glearn.ErrInvalidParameter) {
			t.Errorf("expected ErrInvalidParameter, got %v", err)
		}
	})

	t.Run("too few samples per class", func(t *testing.T) {
		// Class 1 has only 1 sample but NSplits = 3.
		skf := StratifiedKFold{NSplits: 3}
		_, err := skf.Split([]float64{0, 0, 0, 1})
		if !errors.Is(err, glearn.ErrInvalidParameter) {
			t.Errorf("expected ErrInvalidParameter, got %v", err)
		}
	})
}

func TestStratifiedKFold_MultiClass(t *testing.T) {
	// 3 classes with different proportions: 50 class 0, 30 class 1, 20 class 2.
	y := make([]float64, 100)
	for i := 50; i < 80; i++ {
		y[i] = 1.0
	}
	for i := 80; i < 100; i++ {
		y[i] = 2.0
	}

	skf := StratifiedKFold{NSplits: 5, Shuffle: true, Seed: 42}
	folds, err := skf.Split(y)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for i, fold := range folds {
		counts := make(map[float64]int)
		for _, idx := range fold.TestIndices {
			counts[y[idx]]++
		}
		// Each fold should have samples from all 3 classes.
		if len(counts) != 3 {
			t.Errorf("fold %d: expected 3 classes in test set, got %d (counts: %v)",
				i, len(counts), counts)
		}
		// Expected per fold: ~10 class 0, ~6 class 1, ~4 class 2.
		if counts[0] < 8 || counts[0] > 12 {
			t.Errorf("fold %d: class 0 count %d outside expected range [8, 12]", i, counts[0])
		}
		if counts[1] < 4 || counts[1] > 8 {
			t.Errorf("fold %d: class 1 count %d outside expected range [4, 8]", i, counts[1])
		}
		if counts[2] < 2 || counts[2] > 6 {
			t.Errorf("fold %d: class 2 count %d outside expected range [2, 6]", i, counts[2])
		}
	}
}
