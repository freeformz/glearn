package metrics

import (
	"math"
	"testing"
)

// --- MAE ---

func TestMAE_Perfect(t *testing.T) {
	yTrue := []float64{1, 2, 3, 4, 5}
	yPred := []float64{1, 2, 3, 4, 5}
	got := MAE(yTrue, yPred)
	if got != 0.0 {
		t.Errorf("MAE = %v, want 0.0", got)
	}
}

func TestMAE_Known(t *testing.T) {
	yTrue := []float64{3, -0.5, 2, 7}
	yPred := []float64{2.5, 0.0, 2, 8}
	got := MAE(yTrue, yPred)
	want := (0.5 + 0.5 + 0.0 + 1.0) / 4.0 // = 0.5
	if !almostEqual(got, want, tolerance) {
		t.Errorf("MAE = %v, want %v", got, want)
	}
}

func TestMAE_SingleElement(t *testing.T) {
	got := MAE([]float64{5.0}, []float64{3.0})
	if !almostEqual(got, 2.0, tolerance) {
		t.Errorf("MAE = %v, want 2.0", got)
	}
}

func TestMAE_Symmetric(t *testing.T) {
	yTrue := []float64{1, 2, 3}
	yPred := []float64{3, 2, 1}
	// |1-3| + |2-2| + |3-1| = 2+0+2 = 4/3
	got := MAE(yTrue, yPred)
	want := 4.0 / 3.0
	if !almostEqual(got, want, tolerance) {
		t.Errorf("MAE = %v, want %v", got, want)
	}
}

func TestMAE_PanicOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty input")
		}
	}()
	MAE(nil, nil)
}

// --- MSE ---

func TestMSE_Perfect(t *testing.T) {
	yTrue := []float64{1, 2, 3}
	yPred := []float64{1, 2, 3}
	got := MSE(yTrue, yPred)
	if got != 0.0 {
		t.Errorf("MSE = %v, want 0.0", got)
	}
}

func TestMSE_Known(t *testing.T) {
	yTrue := []float64{3, -0.5, 2, 7}
	yPred := []float64{2.5, 0.0, 2, 8}
	got := MSE(yTrue, yPred)
	want := (0.25 + 0.25 + 0.0 + 1.0) / 4.0 // = 0.375
	if !almostEqual(got, want, tolerance) {
		t.Errorf("MSE = %v, want %v", got, want)
	}
}

func TestMSE_SingleElement(t *testing.T) {
	got := MSE([]float64{5.0}, []float64{3.0})
	if !almostEqual(got, 4.0, tolerance) {
		t.Errorf("MSE = %v, want 4.0", got)
	}
}

func TestMSE_PanicOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty input")
		}
	}()
	MSE(nil, nil)
}

// --- RMSE ---

func TestRMSE_Perfect(t *testing.T) {
	yTrue := []float64{1, 2, 3}
	yPred := []float64{1, 2, 3}
	got := RMSE(yTrue, yPred)
	if got != 0.0 {
		t.Errorf("RMSE = %v, want 0.0", got)
	}
}

func TestRMSE_Known(t *testing.T) {
	yTrue := []float64{3, -0.5, 2, 7}
	yPred := []float64{2.5, 0.0, 2, 8}
	got := RMSE(yTrue, yPred)
	want := math.Sqrt(0.375)
	if !almostEqual(got, want, tolerance) {
		t.Errorf("RMSE = %v, want %v", got, want)
	}
}

func TestRMSE_IsSqrtOfMSE(t *testing.T) {
	yTrue := []float64{1, 2, 3, 4, 5}
	yPred := []float64{1.1, 2.2, 2.8, 4.1, 5.3}
	rmse := RMSE(yTrue, yPred)
	mse := MSE(yTrue, yPred)
	if !almostEqual(rmse, math.Sqrt(mse), tolerance) {
		t.Errorf("RMSE (%v) != sqrt(MSE) (%v)", rmse, math.Sqrt(mse))
	}
}

func TestRMSE_PanicOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty input")
		}
	}()
	RMSE(nil, nil)
}

// --- R2 ---

func TestR2_Perfect(t *testing.T) {
	yTrue := []float64{1, 2, 3, 4, 5}
	yPred := []float64{1, 2, 3, 4, 5}
	got := R2(yTrue, yPred)
	if got != 1.0 {
		t.Errorf("R2 = %v, want 1.0", got)
	}
}

func TestR2_Known(t *testing.T) {
	yTrue := []float64{3, -0.5, 2, 7}
	yPred := []float64{2.5, 0.0, 2, 8}
	got := R2(yTrue, yPred)
	// ssRes = 0.25+0.25+0+1 = 1.5
	// mean = (3-0.5+2+7)/4 = 11.5/4 = 2.875
	// ssTot = (3-2.875)^2 + (-0.5-2.875)^2 + (2-2.875)^2 + (7-2.875)^2
	//       = 0.015625 + 11.390625 + 0.765625 + 17.015625 = 29.1875
	// R2 = 1 - 1.5/29.1875
	want := 1.0 - 1.5/29.1875
	if !almostEqual(got, want, tolerance) {
		t.Errorf("R2 = %v, want %v", got, want)
	}
}

func TestR2_MeanPredictor(t *testing.T) {
	yTrue := []float64{1, 2, 3, 4, 5}
	mean := 3.0
	yPred := []float64{mean, mean, mean, mean, mean}
	got := R2(yTrue, yPred)
	if !almostEqual(got, 0.0, tolerance) {
		t.Errorf("R2 = %v, want 0.0", got)
	}
}

func TestR2_WorseThanMean(t *testing.T) {
	yTrue := []float64{1, 2, 3}
	yPred := []float64{10, 20, 30} // Very bad predictions.
	got := R2(yTrue, yPred)
	if got >= 0 {
		t.Errorf("R2 = %v, expected negative", got)
	}
}

func TestR2_ConstantTrue(t *testing.T) {
	yTrue := []float64{5, 5, 5}
	yPred := []float64{5, 5, 5}
	got := R2(yTrue, yPred)
	if got != 1.0 {
		t.Errorf("R2 = %v, want 1.0 (constant perfect prediction)", got)
	}
}

func TestR2_ConstantTrueImperfectPred(t *testing.T) {
	yTrue := []float64{5, 5, 5}
	yPred := []float64{4, 5, 6}
	got := R2(yTrue, yPred)
	if got != 0.0 {
		t.Errorf("R2 = %v, want 0.0 (constant true, imperfect pred)", got)
	}
}

func TestR2_SingleElement(t *testing.T) {
	got := R2([]float64{5.0}, []float64{5.0})
	if got != 1.0 {
		t.Errorf("R2 = %v, want 1.0", got)
	}
}

func TestR2_PanicOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty input")
		}
	}()
	R2(nil, nil)
}

// --- Cross-metric consistency ---

func TestMSE_MAE_Relationship(t *testing.T) {
	// For uniform errors, MSE >= MAE^2 (Jensen's inequality).
	yTrue := []float64{1, 2, 3, 4, 5}
	yPred := []float64{1.5, 2.5, 3.5, 4.5, 5.5}
	mae := MAE(yTrue, yPred)
	mse := MSE(yTrue, yPred)
	if mse < mae*mae-tolerance {
		t.Errorf("MSE (%v) < MAE^2 (%v), violates Jensen's inequality", mse, mae*mae)
	}
}
