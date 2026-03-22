package modelselection

import (
	"errors"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

func TestTrainTestSplit_Ratio(t *testing.T) {
	X := mat.NewDense(100, 3, nil)
	for i := range 100 {
		for j := range 3 {
			X.Set(i, j, float64(i*3+j))
		}
	}
	y := make([]float64, 100)
	for i := range 100 {
		y[i] = float64(i)
	}

	XTrain, XTest, yTrain, yTest, err := TrainTestSplit(X, y, 0.2, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rTrain, _ := XTrain.Dims()
	rTest, _ := XTest.Dims()

	if rTrain != 80 {
		t.Errorf("expected 80 training samples, got %d", rTrain)
	}
	if rTest != 20 {
		t.Errorf("expected 20 test samples, got %d", rTest)
	}
	if len(yTrain) != 80 {
		t.Errorf("expected 80 yTrain elements, got %d", len(yTrain))
	}
	if len(yTest) != 20 {
		t.Errorf("expected 20 yTest elements, got %d", len(yTest))
	}

	// Verify total samples are preserved.
	if rTrain+rTest != 100 {
		t.Errorf("total samples %d != 100", rTrain+rTest)
	}
}

func TestTrainTestSplit_Reproducibility(t *testing.T) {
	X := mat.NewDense(50, 2, nil)
	for i := range 50 {
		for j := range 2 {
			X.Set(i, j, float64(i*2+j))
		}
	}
	y := make([]float64, 50)
	for i := range 50 {
		y[i] = float64(i)
	}

	_, _, yTrain1, yTest1, err := TrainTestSplit(X, y, 0.3, 123)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, _, yTrain2, yTest2, err := TrainTestSplit(X, y, 0.3, 123)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Same seed should produce identical splits.
	for i := range yTrain1 {
		if yTrain1[i] != yTrain2[i] {
			t.Fatalf("yTrain differs at index %d: %g vs %g", i, yTrain1[i], yTrain2[i])
		}
	}
	for i := range yTest1 {
		if yTest1[i] != yTest2[i] {
			t.Fatalf("yTest differs at index %d: %g vs %g", i, yTest1[i], yTest2[i])
		}
	}
}

func TestTrainTestSplit_DifferentSeeds(t *testing.T) {
	X := mat.NewDense(50, 2, nil)
	for i := range 50 {
		for j := range 2 {
			X.Set(i, j, float64(i*2+j))
		}
	}
	y := make([]float64, 50)
	for i := range 50 {
		y[i] = float64(i)
	}

	_, _, yTrain1, _, err := TrainTestSplit(X, y, 0.3, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, _, yTrain2, _, err := TrainTestSplit(X, y, 0.3, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Different seeds should (almost certainly) produce different splits.
	same := true
	for i := range yTrain1 {
		if yTrain1[i] != yTrain2[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("different seeds produced identical splits")
	}
}

func TestTrainTestSplit_DataIntegrity(t *testing.T) {
	// Verify that split data matches the original data.
	nSamples := 20
	nFeatures := 3
	X := mat.NewDense(nSamples, nFeatures, nil)
	y := make([]float64, nSamples)
	for i := range nSamples {
		y[i] = float64(i * 10)
		for j := range nFeatures {
			X.Set(i, j, float64(i*100+j))
		}
	}

	XTrain, XTest, yTrain, yTest, err := TrainTestSplit(X, y, 0.25, 7)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Build a set of all original y values.
	allY := make(map[float64]bool)
	for _, v := range yTrain {
		allY[v] = true
	}
	for _, v := range yTest {
		allY[v] = true
	}
	if len(allY) != nSamples {
		t.Errorf("expected %d unique y values, got %d", nSamples, len(allY))
	}

	// Verify X and y correspondence: if y = i*10, then X[row][0] should be i*100.
	rTrain, _ := XTrain.Dims()
	for i := range rTrain {
		idx := int(yTrain[i]) / 10
		expected := float64(idx * 100)
		if XTrain.At(i, 0) != expected {
			t.Errorf("XTrain row %d: expected first feature %g, got %g", i, expected, XTrain.At(i, 0))
		}
	}
	rTest, _ := XTest.Dims()
	for i := range rTest {
		idx := int(yTest[i]) / 10
		expected := float64(idx * 100)
		if XTest.At(i, 0) != expected {
			t.Errorf("XTest row %d: expected first feature %g, got %g", i, expected, XTest.At(i, 0))
		}
	}
}

func TestTrainTestSplit_InvalidInputs(t *testing.T) {
	X := mat.NewDense(10, 2, nil)
	y := make([]float64, 10)

	tests := []struct {
		name     string
		X        *mat.Dense
		y        []float64
		testSize float64
		wantErr  error
	}{
		{
			name:     "nil X",
			X:        nil,
			y:        y,
			testSize: 0.2,
			wantErr:  glearn.ErrEmptyInput,
		},
		{
			name:     "mismatched y length",
			X:        X,
			y:        make([]float64, 5),
			testSize: 0.2,
			wantErr:  glearn.ErrDimensionMismatch,
		},
		{
			name:     "testSize too small",
			X:        X,
			y:        y,
			testSize: 0.0,
			wantErr:  glearn.ErrInvalidParameter,
		},
		{
			name:     "testSize too large",
			X:        X,
			y:        y,
			testSize: 1.0,
			wantErr:  glearn.ErrInvalidParameter,
		},
		{
			name:     "testSize negative",
			X:        X,
			y:        y,
			testSize: -0.1,
			wantErr:  glearn.ErrInvalidParameter,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, _, err := TrainTestSplit(tt.X, tt.y, tt.testSize, 42)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("expected error wrapping %v, got %v", tt.wantErr, err)
			}
		})
	}
}

func TestTrainTestSplit_SmallDataset(t *testing.T) {
	// Even with 2 samples, split should work.
	X := mat.NewDense(2, 1, []float64{1, 2})
	y := []float64{10, 20}

	XTrain, XTest, yTrain, yTest, err := TrainTestSplit(X, y, 0.5, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	rTrain, _ := XTrain.Dims()
	rTest, _ := XTest.Dims()
	if rTrain+rTest != 2 {
		t.Errorf("expected total 2 samples, got %d", rTrain+rTest)
	}
	if len(yTrain)+len(yTest) != 2 {
		t.Errorf("expected total 2 y values, got %d", len(yTrain)+len(yTest))
	}

	// Check that both original values appear.
	allVals := make(map[float64]bool)
	for _, v := range yTrain {
		allVals[v] = true
	}
	for _, v := range yTest {
		allVals[v] = true
	}
	if !allVals[10] || !allVals[20] {
		t.Error("split did not preserve all original y values")
	}
}

func TestTrainTestSplit_NilMatrix(t *testing.T) {
	_, _, _, _, err := TrainTestSplit(nil, nil, 0.2, 42)
	if err == nil {
		t.Fatal("expected error for nil matrix, got nil")
	}
	if !errors.Is(err, glearn.ErrEmptyInput) {
		t.Errorf("expected ErrEmptyInput, got %v", err)
	}
}

