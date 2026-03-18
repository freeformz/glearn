package metrics

import (
	"math"
	"testing"
)

const tolerance = 1e-10

func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}

// --- Accuracy ---

func TestAccuracy_PerfectPredictions(t *testing.T) {
	yTrue := []float64{0, 1, 1, 0, 1}
	yPred := []float64{0, 1, 1, 0, 1}
	got := Accuracy(yTrue, yPred)
	if got != 1.0 {
		t.Errorf("Accuracy = %v, want 1.0", got)
	}
}

func TestAccuracy_AllWrong(t *testing.T) {
	yTrue := []float64{0, 0, 0, 0}
	yPred := []float64{1, 1, 1, 1}
	got := Accuracy(yTrue, yPred)
	if got != 0.0 {
		t.Errorf("Accuracy = %v, want 0.0", got)
	}
}

func TestAccuracy_Partial(t *testing.T) {
	yTrue := []float64{0, 1, 1, 0}
	yPred := []float64{0, 1, 0, 1}
	got := Accuracy(yTrue, yPred)
	if got != 0.5 {
		t.Errorf("Accuracy = %v, want 0.5", got)
	}
}

func TestAccuracy_SingleElement(t *testing.T) {
	got := Accuracy([]float64{1}, []float64{1})
	if got != 1.0 {
		t.Errorf("Accuracy = %v, want 1.0", got)
	}
}

func TestAccuracy_MultiClass(t *testing.T) {
	yTrue := []float64{0, 1, 2, 0, 1, 2}
	yPred := []float64{0, 1, 2, 2, 1, 0}
	got := Accuracy(yTrue, yPred)
	want := 4.0 / 6.0 // 4 correct out of 6
	if !almostEqual(got, want, tolerance) {
		t.Errorf("Accuracy = %v, want %v", got, want)
	}
}

func TestAccuracy_PanicOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty input")
		}
	}()
	Accuracy(nil, nil)
}

func TestAccuracy_PanicOnLengthMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on length mismatch")
		}
	}()
	Accuracy([]float64{1, 0}, []float64{1})
}

// --- Precision ---

func TestPrecision_BinaryDefault(t *testing.T) {
	// TP=2, FP=1 for class 1
	yTrue := []float64{0, 1, 1, 0, 1}
	yPred := []float64{0, 1, 1, 1, 0}
	got := Precision(yTrue, yPred)
	want := 2.0 / 3.0 // TP=2, FP=1
	if !almostEqual(got, want, tolerance) {
		t.Errorf("Precision(binary) = %v, want %v", got, want)
	}
}

func TestPrecision_BinaryAllPositivePredictions(t *testing.T) {
	yTrue := []float64{1, 1, 0, 0}
	yPred := []float64{1, 1, 1, 1}
	got := Precision(yTrue, yPred)
	want := 2.0 / 4.0 // TP=2, FP=2
	if !almostEqual(got, want, tolerance) {
		t.Errorf("Precision = %v, want %v", got, want)
	}
}

func TestPrecision_BinaryNoPositivePredictions(t *testing.T) {
	yTrue := []float64{1, 1, 0, 0}
	yPred := []float64{0, 0, 0, 0}
	got := Precision(yTrue, yPred)
	want := 0.0 // No positive predictions, precision is 0
	if !almostEqual(got, want, tolerance) {
		t.Errorf("Precision = %v, want %v", got, want)
	}
}

func TestPrecision_Micro(t *testing.T) {
	yTrue := []float64{0, 1, 2, 0, 1, 2}
	yPred := []float64{0, 2, 1, 0, 0, 2}
	got := Precision(yTrue, yPred, WithAverage(AverageMicro))
	// Micro precision = total TP / total (TP + FP) = accuracy for multi-class
	want := Accuracy(yTrue, yPred)
	if !almostEqual(got, want, tolerance) {
		t.Errorf("Precision(micro) = %v, want %v", got, want)
	}
}

func TestPrecision_Macro(t *testing.T) {
	yTrue := []float64{0, 1, 2, 0, 1, 2}
	yPred := []float64{0, 2, 1, 0, 0, 2}
	got := Precision(yTrue, yPred, WithAverage(AverageMacro))
	// Class 0: TP=2, FP=1 => 2/3
	// Class 1: TP=0, FP=1 => 0/1 = 0
	// Class 2: TP=1, FP=1 => 1/2
	want := (2.0/3.0 + 0.0 + 1.0/2.0) / 3.0
	if !almostEqual(got, want, tolerance) {
		t.Errorf("Precision(macro) = %v, want %v", got, want)
	}
}

func TestPrecision_Weighted(t *testing.T) {
	yTrue := []float64{0, 1, 2, 0, 1, 2}
	yPred := []float64{0, 2, 1, 0, 0, 2}
	got := Precision(yTrue, yPred, WithAverage(AverageWeighted))
	// Support: class 0=2, class 1=2, class 2=2, total=6
	// Precision per class: 0=2/3, 1=0, 2=1/2
	want := (2.0/3.0*2 + 0.0*2 + 1.0/2.0*2) / 6.0
	if !almostEqual(got, want, tolerance) {
		t.Errorf("Precision(weighted) = %v, want %v", got, want)
	}
}

func TestPrecision_PanicOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty input")
		}
	}()
	Precision(nil, nil)
}

// --- Recall ---

func TestRecall_BinaryDefault(t *testing.T) {
	yTrue := []float64{0, 1, 1, 0, 1}
	yPred := []float64{0, 1, 1, 1, 0}
	got := Recall(yTrue, yPred)
	want := 2.0 / 3.0 // TP=2, FN=1 for class 1
	if !almostEqual(got, want, tolerance) {
		t.Errorf("Recall(binary) = %v, want %v", got, want)
	}
}

func TestRecall_BinaryPerfect(t *testing.T) {
	yTrue := []float64{0, 1, 1, 0}
	yPred := []float64{0, 1, 1, 0}
	got := Recall(yTrue, yPred)
	if got != 1.0 {
		t.Errorf("Recall = %v, want 1.0", got)
	}
}

func TestRecall_Micro(t *testing.T) {
	yTrue := []float64{0, 1, 2, 0, 1, 2}
	yPred := []float64{0, 2, 1, 0, 0, 2}
	got := Recall(yTrue, yPred, WithAverage(AverageMicro))
	// Micro recall = total TP / total (TP + FN) = accuracy for multi-class
	want := Accuracy(yTrue, yPred)
	if !almostEqual(got, want, tolerance) {
		t.Errorf("Recall(micro) = %v, want %v", got, want)
	}
}

func TestRecall_Macro(t *testing.T) {
	yTrue := []float64{0, 1, 2, 0, 1, 2}
	yPred := []float64{0, 2, 1, 0, 0, 2}
	got := Recall(yTrue, yPred, WithAverage(AverageMacro))
	// Class 0: TP=2, FN=0 => 2/2 = 1
	// Class 1: TP=0, FN=2 => 0/2 = 0
	// Class 2: TP=1, FN=1 => 1/2
	want := (1.0 + 0.0 + 0.5) / 3.0
	if !almostEqual(got, want, tolerance) {
		t.Errorf("Recall(macro) = %v, want %v", got, want)
	}
}

func TestRecall_Weighted(t *testing.T) {
	yTrue := []float64{0, 1, 2, 0, 1, 2}
	yPred := []float64{0, 2, 1, 0, 0, 2}
	got := Recall(yTrue, yPred, WithAverage(AverageWeighted))
	// Support: class 0=2, class 1=2, class 2=2, total=6
	// Recall per class: 0=1, 1=0, 2=0.5
	want := (1.0*2 + 0.0*2 + 0.5*2) / 6.0
	if !almostEqual(got, want, tolerance) {
		t.Errorf("Recall(weighted) = %v, want %v", got, want)
	}
}

func TestRecall_PanicOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty input")
		}
	}()
	Recall(nil, nil)
}

// --- F1 ---

func TestF1_BinaryDefault(t *testing.T) {
	yTrue := []float64{0, 1, 1, 0, 1}
	yPred := []float64{0, 1, 1, 1, 0}
	got := F1(yTrue, yPred)
	// P=2/3, R=2/3, F1 = 2*(2/3)*(2/3)/((2/3)+(2/3)) = 2/3
	want := 2.0 / 3.0
	if !almostEqual(got, want, tolerance) {
		t.Errorf("F1(binary) = %v, want %v", got, want)
	}
}

func TestF1_Perfect(t *testing.T) {
	yTrue := []float64{0, 1, 1, 0}
	yPred := []float64{0, 1, 1, 0}
	got := F1(yTrue, yPred)
	if got != 1.0 {
		t.Errorf("F1 = %v, want 1.0", got)
	}
}

func TestF1_Micro(t *testing.T) {
	yTrue := []float64{0, 1, 2, 0, 1, 2}
	yPred := []float64{0, 2, 1, 0, 0, 2}
	got := F1(yTrue, yPred, WithAverage(AverageMicro))
	// For micro, precision = recall = accuracy, so F1 = accuracy.
	want := Accuracy(yTrue, yPred)
	if !almostEqual(got, want, tolerance) {
		t.Errorf("F1(micro) = %v, want %v", got, want)
	}
}

func TestF1_Macro(t *testing.T) {
	yTrue := []float64{0, 1, 2, 0, 1, 2}
	yPred := []float64{0, 2, 1, 0, 0, 2}
	got := F1(yTrue, yPred, WithAverage(AverageMacro))
	// Class 0: P=2/3, R=1 => F1=2*(2/3)*1/((2/3)+1) = (4/3)/(5/3) = 4/5 = 0.8
	// Class 1: P=0, R=0 => F1=0
	// Class 2: P=1/2, R=1/2 => F1=2*(1/2)*(1/2)/((1/2)+(1/2)) = 1/2 = 0.5
	want := (0.8 + 0.0 + 0.5) / 3.0
	if !almostEqual(got, want, tolerance) {
		t.Errorf("F1(macro) = %v, want %v", got, want)
	}
}

func TestF1_Weighted(t *testing.T) {
	yTrue := []float64{0, 1, 2, 0, 1, 2}
	yPred := []float64{0, 2, 1, 0, 0, 2}
	got := F1(yTrue, yPred, WithAverage(AverageWeighted))
	// Support: class 0=2, class 1=2, class 2=2, total=6
	// F1 per class: 0=0.8, 1=0, 2=0.5
	want := (0.8*2 + 0.0*2 + 0.5*2) / 6.0
	if !almostEqual(got, want, tolerance) {
		t.Errorf("F1(weighted) = %v, want %v", got, want)
	}
}

func TestF1_ZeroPrecisionAndRecall(t *testing.T) {
	// All predictions wrong for positive class.
	yTrue := []float64{0, 0, 0}
	yPred := []float64{1, 1, 1}
	got := F1(yTrue, yPred)
	// For class 1: TP=0, FP=3, FN=0 => P=0, R=0 => F1=0
	if got != 0.0 {
		t.Errorf("F1 = %v, want 0.0", got)
	}
}

func TestF1_PanicOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty input")
		}
	}()
	F1(nil, nil)
}

// --- ConfusionMatrix ---

func TestConfusionMatrix_Binary(t *testing.T) {
	yTrue := []float64{0, 0, 1, 1, 1}
	yPred := []float64{0, 1, 1, 1, 0}
	cm := ConfusionMatrix(yTrue, yPred)

	// Classes [0, 1], matrix:
	// TN=1  FP=1
	// FN=1  TP=2
	expected := [][]int{
		{1, 1},
		{1, 2},
	}

	if len(cm) != 2 {
		t.Fatalf("expected 2x2 matrix, got %dx%d", len(cm), len(cm[0]))
	}
	for i := range expected {
		for j := range expected[i] {
			if cm[i][j] != expected[i][j] {
				t.Errorf("cm[%d][%d] = %d, want %d", i, j, cm[i][j], expected[i][j])
			}
		}
	}
}

func TestConfusionMatrix_MultiClass(t *testing.T) {
	yTrue := []float64{0, 1, 2, 0, 1, 2}
	yPred := []float64{0, 2, 1, 0, 0, 2}
	cm := ConfusionMatrix(yTrue, yPred)

	// Classes [0, 1, 2], matrix:
	//       pred0  pred1  pred2
	// true0:  2      0      0
	// true1:  1      0      1
	// true2:  0      1      1
	expected := [][]int{
		{2, 0, 0},
		{1, 0, 1},
		{0, 1, 1},
	}

	if len(cm) != 3 {
		t.Fatalf("expected 3x3 matrix, got %d rows", len(cm))
	}
	for i := range expected {
		for j := range expected[i] {
			if cm[i][j] != expected[i][j] {
				t.Errorf("cm[%d][%d] = %d, want %d", i, j, cm[i][j], expected[i][j])
			}
		}
	}
}

func TestConfusionMatrix_Perfect(t *testing.T) {
	yTrue := []float64{0, 1, 2}
	yPred := []float64{0, 1, 2}
	cm := ConfusionMatrix(yTrue, yPred)

	// Diagonal should all be 1, off-diagonal all 0.
	for i := range cm {
		for j := range cm[i] {
			if i == j {
				if cm[i][j] != 1 {
					t.Errorf("cm[%d][%d] = %d, want 1", i, j, cm[i][j])
				}
			} else {
				if cm[i][j] != 0 {
					t.Errorf("cm[%d][%d] = %d, want 0", i, j, cm[i][j])
				}
			}
		}
	}
}

func TestConfusionMatrix_SingleElement(t *testing.T) {
	cm := ConfusionMatrix([]float64{1}, []float64{1})
	if len(cm) != 1 || cm[0][0] != 1 {
		t.Errorf("expected [[1]], got %v", cm)
	}
}

func TestConfusionMatrix_PanicOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty input")
		}
	}()
	ConfusionMatrix(nil, nil)
}

// --- Helpers ---

func TestUniqueClasses_Sorted(t *testing.T) {
	yTrue := []float64{2, 0, 1, 2}
	yPred := []float64{0, 1, 2, 3}
	classes := uniqueClasses(yTrue, yPred)
	expected := []float64{0, 1, 2, 3}
	if len(classes) != len(expected) {
		t.Fatalf("got %d classes, want %d", len(classes), len(expected))
	}
	for i, c := range classes {
		if c != expected[i] {
			t.Errorf("classes[%d] = %v, want %v", i, c, expected[i])
		}
	}
}

func TestValidateInputs_PanicOnEmptyYTrue(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	validateInputs([]float64{}, []float64{1})
}

func TestValidateInputs_PanicOnEmptyYPred(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	validateInputs([]float64{1}, []float64{})
}

func TestValidateInputs_PanicOnMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	validateInputs([]float64{1, 2}, []float64{1})
}

// --- Edge cases for average modes ---

func TestPrecision_BinaryWithNoPositiveClassInData(t *testing.T) {
	// Only class 0 in truth; class 1 never appears as truth.
	yTrue := []float64{0, 0, 0}
	yPred := []float64{0, 0, 1}
	got := Precision(yTrue, yPred)
	// For class 1: TP=0, FP=1, precision=0
	if got != 0.0 {
		t.Errorf("Precision = %v, want 0.0", got)
	}
}

func TestRecall_BinaryWithNoPositiveClassInData(t *testing.T) {
	yTrue := []float64{0, 0, 0}
	yPred := []float64{0, 0, 1}
	got := Recall(yTrue, yPred)
	// For class 1: TP=0, FN=0, recall=0/0=0 (convention)
	if got != 0.0 {
		t.Errorf("Recall = %v, want 0.0", got)
	}
}
