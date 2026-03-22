package cluster

import (
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

func TestDBSCANClusteredDataWithNoise(t *testing.T) {
	// Two dense clusters with a noise point far away.
	data := []float64{
		// Cluster 0: tight group near (0, 0)
		0.0, 0.0,
		0.1, 0.0,
		0.0, 0.1,
		0.1, 0.1,
		0.05, 0.05,
		// Cluster 1: tight group near (5, 5)
		5.0, 5.0,
		5.1, 5.0,
		5.0, 5.1,
		5.1, 5.1,
		5.05, 5.05,
		// Noise point
		50.0, 50.0,
	}
	X := mat.NewDense(11, 2, data)

	cfg := NewDBSCAN(WithEps(0.5), WithMinSamples(3))
	result, err := cfg.Fit(t.Context(), X, nil)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	db := result.(*DBSCAN)

	// Points in first cluster should have the same label.
	cluster0Label := db.Labels[0]
	for i := 1; i < 5; i++ {
		if db.Labels[i] != cluster0Label {
			t.Errorf("expected point %d in same cluster as point 0, got label %d vs %d", i, db.Labels[i], cluster0Label)
		}
	}

	// Points in second cluster should have the same label.
	cluster1Label := db.Labels[5]
	for i := 6; i < 10; i++ {
		if db.Labels[i] != cluster1Label {
			t.Errorf("expected point %d in same cluster as point 5, got label %d vs %d", i, db.Labels[i], cluster1Label)
		}
	}

	// The two clusters should have different labels.
	if cluster0Label == cluster1Label {
		t.Error("expected different labels for the two clusters")
	}

	// The noise point should be labeled -1.
	if db.Labels[10] != -1 {
		t.Errorf("expected noise point label -1, got %d", db.Labels[10])
	}

	// Should have core samples.
	if len(db.CoreSampleIndices) == 0 {
		t.Error("expected at least one core sample")
	}
}

func TestDBSCANEpsEffect(t *testing.T) {
	// Points spaced 1.0 apart.
	data := []float64{
		0, 0,
		1, 0,
		2, 0,
		3, 0,
		4, 0,
	}
	X := mat.NewDense(5, 2, data)

	// With Eps=0.5, no point has enough neighbors -> all noise.
	cfg := NewDBSCAN(WithEps(0.5), WithMinSamples(2))
	result, err := cfg.Fit(t.Context(), X, nil)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	db := result.(*DBSCAN)
	for i, label := range db.Labels {
		if label != -1 {
			t.Errorf("Eps=0.5: expected noise for point %d, got label %d", i, label)
		}
	}

	// With Eps=1.5, all points should be in one cluster.
	cfg = NewDBSCAN(WithEps(1.5), WithMinSamples(2))
	result, err = cfg.Fit(t.Context(), X, nil)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	db = result.(*DBSCAN)
	firstLabel := db.Labels[0]
	if firstLabel == -1 {
		t.Error("Eps=1.5: expected a cluster, but got noise")
	}
	for i, label := range db.Labels {
		if label != firstLabel {
			t.Errorf("Eps=1.5: expected all points in same cluster, point %d has label %d vs %d", i, label, firstLabel)
		}
	}
}

func TestDBSCANMinSamplesEffect(t *testing.T) {
	data := []float64{
		0, 0,
		0.1, 0,
		0, 0.1,
		10, 10,
	}
	X := mat.NewDense(4, 2, data)

	// With MinSamples=3, the cluster of 3 should form; the outlier is noise.
	cfg := NewDBSCAN(WithEps(0.5), WithMinSamples(3))
	result, err := cfg.Fit(t.Context(), X, nil)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	db := result.(*DBSCAN)

	if db.Labels[3] != -1 {
		t.Errorf("expected outlier to be noise, got label %d", db.Labels[3])
	}

	clusterLabel := db.Labels[0]
	if clusterLabel == -1 {
		t.Error("expected first 3 points in a cluster, got noise")
	}
	for i := 1; i < 3; i++ {
		if db.Labels[i] != clusterLabel {
			t.Errorf("expected point %d in same cluster as point 0", i)
		}
	}

	// With MinSamples=4, no point has enough neighbors -> all noise.
	cfg = NewDBSCAN(WithEps(0.5), WithMinSamples(4))
	result, err = cfg.Fit(t.Context(), X, nil)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}
	db = result.(*DBSCAN)
	for i, label := range db.Labels {
		if label != -1 {
			t.Errorf("MinSamples=4: expected noise for point %d, got label %d", i, label)
		}
	}
}

func TestDBSCANPredict(t *testing.T) {
	data := []float64{
		0, 0,
		0.1, 0,
		0, 0.1,
		0.1, 0.1,
		0.05, 0.05,
	}
	X := mat.NewDense(5, 2, data)

	cfg := NewDBSCAN(WithEps(0.5), WithMinSamples(3))
	result, err := cfg.Fit(t.Context(), X, nil)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Predict on a close point and a far point.
	newPoints := mat.NewDense(2, 2, []float64{
		0.05, 0.0, // close to cluster
		100, 100, // far away
	})
	preds, err := result.Predict(newPoints)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	if preds[0] < 0 {
		t.Error("expected close point to be assigned to a cluster")
	}
	if preds[1] != -1 {
		t.Errorf("expected far point to be noise (-1), got %v", preds[1])
	}
}

func TestDBSCANDimensionMismatch(t *testing.T) {
	X := mat.NewDense(5, 2, []float64{0, 0, 1, 0, 0, 1, 1, 1, 0.5, 0.5})
	cfg := NewDBSCAN(WithEps(1.5), WithMinSamples(2))
	result, err := cfg.Fit(t.Context(), X, nil)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	badX := mat.NewDense(1, 3, []float64{1, 2, 3})
	_, err = result.Predict(badX)
	if err == nil {
		t.Fatal("expected error for dimension mismatch in Predict")
	}
}

func TestDBSCANInvalidParameters(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})

	cfg := NewDBSCAN(WithEps(0))
	_, err := cfg.Fit(t.Context(), X, nil)
	if err == nil {
		t.Fatal("expected error for Eps=0")
	}

	cfg = NewDBSCAN(WithMinSamples(0))
	_, err = cfg.Fit(t.Context(), X, nil)
	if err == nil {
		t.Fatal("expected error for MinSamples=0")
	}
}

// Compile-time check that DBSCANConfig implements glearn.Estimator.
var _ glearn.Estimator = DBSCANConfig{}
var _ glearn.Predictor = (*DBSCAN)(nil)
