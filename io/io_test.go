package io

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"math"
	"path/filepath"
	"testing"

	"github.com/vmihailenco/msgpack/v5"
)

// testModel mimics a fitted LinearRegression with simple types ([]float64, float64, int).
type testModel struct {
	Coefficients []float64 `json:"coefficients" msgpack:"coefficients"`
	Intercept    float64   `json:"intercept" msgpack:"intercept"`
	NFeatures    int       `json:"n_features" msgpack:"n_features"`
}

// sampleModel returns a testModel with realistic values for testing.
func sampleModel() *testModel {
	return &testModel{
		Coefficients: []float64{1.5, -2.3, 0.7, 4.1},
		Intercept:    3.14,
		NFeatures:    4,
	}
}

// assertModelEqual checks that two testModels are deeply equal.
func assertModelEqual(t *testing.T, want, got *testModel) {
	t.Helper()
	if got.NFeatures != want.NFeatures {
		t.Errorf("NFeatures: got %d, want %d", got.NFeatures, want.NFeatures)
	}
	if math.Abs(got.Intercept-want.Intercept) > 1e-12 {
		t.Errorf("Intercept: got %g, want %g", got.Intercept, want.Intercept)
	}
	if len(got.Coefficients) != len(want.Coefficients) {
		t.Fatalf("Coefficients length: got %d, want %d", len(got.Coefficients), len(want.Coefficients))
	}
	for i := range want.Coefficients {
		if math.Abs(got.Coefficients[i]-want.Coefficients[i]) > 1e-12 {
			t.Errorf("Coefficients[%d]: got %g, want %g", i, got.Coefficients[i], want.Coefficients[i])
		}
	}
}

// predict computes a simple linear prediction: sum(coef[i] * x[i]) + intercept.
func predict(m *testModel, x []float64) float64 {
	sum := m.Intercept
	for i := range m.NFeatures {
		sum += m.Coefficients[i] * x[i]
	}
	return sum
}

// --- JSON Tests ---

func TestJSONRoundtrip(t *testing.T) {
	original := sampleModel()

	var buf bytes.Buffer
	if err := SaveJSON(&buf, original); err != nil {
		t.Fatalf("SaveJSON failed: %v", err)
	}

	var loaded testModel
	if err := LoadJSON(&buf, &loaded); err != nil {
		t.Fatalf("LoadJSON failed: %v", err)
	}

	assertModelEqual(t, original, &loaded)
}

func TestJSONChecksumMismatch(t *testing.T) {
	original := sampleModel()

	var buf bytes.Buffer
	if err := SaveJSON(&buf, original); err != nil {
		t.Fatalf("SaveJSON failed: %v", err)
	}

	// Decode the envelope, corrupt the data, re-encode.
	var env Envelope
	if err := json.Unmarshal(buf.Bytes(), &env); err != nil {
		t.Fatalf("could not unmarshal envelope: %v", err)
	}

	// Flip a byte in Data to corrupt it.
	if len(env.Data) > 0 {
		env.Data[0] ^= 0xFF
	}

	corrupted, err := json.Marshal(env)
	if err != nil {
		t.Fatalf("could not marshal corrupted envelope: %v", err)
	}

	var loaded testModel
	err = LoadJSON(bytes.NewReader(corrupted), &loaded)
	if err == nil {
		t.Fatal("expected checksum mismatch error, got nil")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("checksum mismatch")) {
		t.Errorf("expected 'checksum mismatch' in error, got: %v", err)
	}
}

func TestJSONFileRoundtrip(t *testing.T) {
	original := sampleModel()
	path := filepath.Join(t.TempDir(), "model.json")

	if err := SaveJSONFile(path, original); err != nil {
		t.Fatalf("SaveJSONFile failed: %v", err)
	}

	var loaded testModel
	if err := LoadJSONFile(path, &loaded); err != nil {
		t.Fatalf("LoadJSONFile failed: %v", err)
	}

	assertModelEqual(t, original, &loaded)
}

func TestJSONPredict(t *testing.T) {
	original := sampleModel()

	var buf bytes.Buffer
	if err := SaveJSON(&buf, original); err != nil {
		t.Fatalf("SaveJSON failed: %v", err)
	}

	var loaded testModel
	if err := LoadJSON(&buf, &loaded); err != nil {
		t.Fatalf("LoadJSON failed: %v", err)
	}

	// Verify prediction still works after roundtrip.
	x := []float64{1.0, 2.0, 3.0, 4.0}
	wantPred := predict(original, x)
	gotPred := predict(&loaded, x)

	if math.Abs(gotPred-wantPred) > 1e-12 {
		t.Errorf("prediction after load: got %g, want %g", gotPred, wantPred)
	}
}

// --- Gob Tests ---

func TestGobRoundtrip(t *testing.T) {
	original := sampleModel()

	var buf bytes.Buffer
	if err := SaveGob(&buf, original); err != nil {
		t.Fatalf("SaveGob failed: %v", err)
	}

	var loaded testModel
	if err := LoadGob(&buf, &loaded); err != nil {
		t.Fatalf("LoadGob failed: %v", err)
	}

	assertModelEqual(t, original, &loaded)
}

func TestGobChecksumMismatch(t *testing.T) {
	original := sampleModel()

	var buf bytes.Buffer
	if err := SaveGob(&buf, original); err != nil {
		t.Fatalf("SaveGob failed: %v", err)
	}

	// Decode the envelope, corrupt the data, re-encode.
	var env Envelope
	decoder := gob.NewDecoder(&buf)
	if err := decoder.Decode(&env); err != nil {
		t.Fatalf("could not decode envelope: %v", err)
	}

	// Flip a byte in Data to corrupt it.
	if len(env.Data) > 0 {
		env.Data[0] ^= 0xFF
	}

	var corrupted bytes.Buffer
	encoder := gob.NewEncoder(&corrupted)
	if err := encoder.Encode(env); err != nil {
		t.Fatalf("could not encode corrupted envelope: %v", err)
	}

	var loaded testModel
	err := LoadGob(&corrupted, &loaded)
	if err == nil {
		t.Fatal("expected checksum mismatch error, got nil")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("checksum mismatch")) {
		t.Errorf("expected 'checksum mismatch' in error, got: %v", err)
	}
}

func TestGobFileRoundtrip(t *testing.T) {
	original := sampleModel()
	path := filepath.Join(t.TempDir(), "model.gob")

	if err := SaveGobFile(path, original); err != nil {
		t.Fatalf("SaveGobFile failed: %v", err)
	}

	var loaded testModel
	if err := LoadGobFile(path, &loaded); err != nil {
		t.Fatalf("LoadGobFile failed: %v", err)
	}

	assertModelEqual(t, original, &loaded)
}

func TestGobPredict(t *testing.T) {
	original := sampleModel()

	var buf bytes.Buffer
	if err := SaveGob(&buf, original); err != nil {
		t.Fatalf("SaveGob failed: %v", err)
	}

	var loaded testModel
	if err := LoadGob(&buf, &loaded); err != nil {
		t.Fatalf("LoadGob failed: %v", err)
	}

	x := []float64{1.0, 2.0, 3.0, 4.0}
	wantPred := predict(original, x)
	gotPred := predict(&loaded, x)

	if math.Abs(gotPred-wantPred) > 1e-12 {
		t.Errorf("prediction after load: got %g, want %g", gotPred, wantPred)
	}
}

// --- Msgpack Tests ---

func TestMsgpackRoundtrip(t *testing.T) {
	original := sampleModel()

	var buf bytes.Buffer
	if err := SaveMsgpack(&buf, original); err != nil {
		t.Fatalf("SaveMsgpack failed: %v", err)
	}

	var loaded testModel
	if err := LoadMsgpack(&buf, &loaded); err != nil {
		t.Fatalf("LoadMsgpack failed: %v", err)
	}

	assertModelEqual(t, original, &loaded)
}

func TestMsgpackChecksumMismatch(t *testing.T) {
	original := sampleModel()

	var buf bytes.Buffer
	if err := SaveMsgpack(&buf, original); err != nil {
		t.Fatalf("SaveMsgpack failed: %v", err)
	}

	// Decode the envelope, corrupt the data, re-encode.
	var env Envelope
	if err := msgpack.Unmarshal(buf.Bytes(), &env); err != nil {
		t.Fatalf("could not unmarshal envelope: %v", err)
	}

	// Flip a byte in Data to corrupt it.
	if len(env.Data) > 0 {
		env.Data[0] ^= 0xFF
	}

	corrupted, err := msgpack.Marshal(env)
	if err != nil {
		t.Fatalf("could not marshal corrupted envelope: %v", err)
	}

	var loaded testModel
	err = LoadMsgpack(bytes.NewReader(corrupted), &loaded)
	if err == nil {
		t.Fatal("expected checksum mismatch error, got nil")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("checksum mismatch")) {
		t.Errorf("expected 'checksum mismatch' in error, got: %v", err)
	}
}

func TestMsgpackFileRoundtrip(t *testing.T) {
	original := sampleModel()
	path := filepath.Join(t.TempDir(), "model.msgpack")

	if err := SaveMsgpackFile(path, original); err != nil {
		t.Fatalf("SaveMsgpackFile failed: %v", err)
	}

	var loaded testModel
	if err := LoadMsgpackFile(path, &loaded); err != nil {
		t.Fatalf("LoadMsgpackFile failed: %v", err)
	}

	assertModelEqual(t, original, &loaded)
}

func TestMsgpackPredict(t *testing.T) {
	original := sampleModel()

	var buf bytes.Buffer
	if err := SaveMsgpack(&buf, original); err != nil {
		t.Fatalf("SaveMsgpack failed: %v", err)
	}

	var loaded testModel
	if err := LoadMsgpack(&buf, &loaded); err != nil {
		t.Fatalf("LoadMsgpack failed: %v", err)
	}

	x := []float64{1.0, 2.0, 3.0, 4.0}
	wantPred := predict(original, x)
	gotPred := predict(&loaded, x)

	if math.Abs(gotPred-wantPred) > 1e-12 {
		t.Errorf("prediction after load: got %g, want %g", gotPred, wantPred)
	}
}

// --- Envelope Tests ---

func TestEnvelopeFields(t *testing.T) {
	original := sampleModel()

	// Test JSON envelope has correct metadata.
	var buf bytes.Buffer
	if err := SaveJSON(&buf, original); err != nil {
		t.Fatalf("SaveJSON failed: %v", err)
	}

	var env Envelope
	if err := json.Unmarshal(buf.Bytes(), &env); err != nil {
		t.Fatalf("could not unmarshal envelope: %v", err)
	}

	if env.Format != "json" {
		t.Errorf("Format: got %q, want %q", env.Format, "json")
	}
	if env.Type != "*io.testModel" {
		t.Errorf("Type: got %q, want %q", env.Type, "*io.testModel")
	}
	if env.Version != Version {
		t.Errorf("Version: got %q, want %q", env.Version, Version)
	}
	if env.Checksum == 0 {
		t.Error("Checksum should not be zero")
	}
	if len(env.Data) == 0 {
		t.Error("Data should not be empty")
	}
}

func TestChecksumComputation(t *testing.T) {
	data := []byte("hello, glearn")
	cs := computeChecksum(data)
	if cs == 0 {
		t.Error("checksum should not be zero for non-empty data")
	}

	// Same data should produce same checksum.
	cs2 := computeChecksum(data)
	if cs != cs2 {
		t.Errorf("checksum not deterministic: %d != %d", cs, cs2)
	}

	// Different data should produce different checksum.
	data2 := []byte("hello, glearn!")
	cs3 := computeChecksum(data2)
	if cs == cs3 {
		t.Error("different data produced same checksum")
	}

	// Verify helper.
	if !verifyChecksum(data, cs) {
		t.Error("verifyChecksum returned false for correct checksum")
	}
	if verifyChecksum(data, cs+1) {
		t.Error("verifyChecksum returned true for incorrect checksum")
	}
}

// --- Cross-format tests ---

func TestLoadJSONFileNotFound(t *testing.T) {
	var loaded testModel
	err := LoadJSONFile("/nonexistent/path/model.json", &loaded)
	if err == nil {
		t.Fatal("expected error for nonexistent file, got nil")
	}
}

func TestLoadGobFileNotFound(t *testing.T) {
	var loaded testModel
	err := LoadGobFile("/nonexistent/path/model.gob", &loaded)
	if err == nil {
		t.Fatal("expected error for nonexistent file, got nil")
	}
}

func TestLoadMsgpackFileNotFound(t *testing.T) {
	var loaded testModel
	err := LoadMsgpackFile("/nonexistent/path/model.msgpack", &loaded)
	if err == nil {
		t.Fatal("expected error for nonexistent file, got nil")
	}
}
