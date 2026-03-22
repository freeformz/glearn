package cluster

import (
	"math"
	"sort"
	"testing"

	"github.com/freeformz/glearn"
	"gonum.org/v1/gonum/mat"
)

func TestKMeansWellSeparatedBlobs(t *testing.T) {
	// Three well-separated clusters centered at (0,0), (10,10), (20,0).
	data := []float64{
		// Cluster 0: around (0, 0)
		0.1, 0.2,
		-0.1, -0.2,
		0.2, -0.1,
		-0.2, 0.1,
		0.0, 0.0,
		// Cluster 1: around (10, 10)
		10.1, 10.2,
		9.9, 9.8,
		10.2, 9.9,
		9.8, 10.1,
		10.0, 10.0,
		// Cluster 2: around (20, 0)
		20.1, 0.2,
		19.9, -0.2,
		20.2, -0.1,
		19.8, 0.1,
		20.0, 0.0,
	}
	X := mat.NewDense(15, 2, data)

	cfg := NewKMeans(WithNClusters(3), WithSeed(42))
	result, err := cfg.Fit(t.Context(), X, nil)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	km := result.(*KMeans)

	// Verify we got 3 centroids.
	if len(km.Centroids) != 3 {
		t.Fatalf("expected 3 centroids, got %d", len(km.Centroids))
	}

	// Verify centroids are close to the true centers.
	trueCenters := [][]float64{{0, 0}, {10, 10}, {20, 0}}
	used := make([]bool, 3)
	for _, tc := range trueCenters {
		found := false
		for k, centroid := range km.Centroids {
			if used[k] {
				continue
			}
			d := euclideanDistance(tc, centroid)
			if d < 1.0 {
				used[k] = true
				found = true
				break
			}
		}
		if !found {
			t.Errorf("no centroid found near true center %v; centroids: %v", tc, km.Centroids)
		}
	}

	// Verify that points in the same original cluster have the same label.
	for group := 0; group < 3; group++ {
		label := km.Labels[group*5]
		for i := group*5 + 1; i < (group+1)*5; i++ {
			if km.Labels[i] != label {
				t.Errorf("points in group %d have different labels: %d vs %d", group, label, km.Labels[i])
			}
		}
	}

	// Verify inertia is small (well-separated clusters).
	if km.Inertia > 1.0 {
		t.Errorf("expected low inertia for well-separated clusters, got %f", km.Inertia)
	}
}

func TestKMeansPredict(t *testing.T) {
	data := []float64{
		0, 0,
		0, 1,
		10, 10,
		10, 11,
	}
	X := mat.NewDense(4, 2, data)

	cfg := NewKMeans(WithNClusters(2), WithSeed(1))
	result, err := cfg.Fit(t.Context(), X, nil)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Predict on new points.
	newPoints := mat.NewDense(2, 2, []float64{
		0.5, 0.5, // should be near cluster 0
		9.5, 10.5, // should be near cluster 1
	})

	preds, err := result.Predict(newPoints)
	if err != nil {
		t.Fatalf("Predict failed: %v", err)
	}

	// The two new points should be in different clusters.
	if preds[0] == preds[1] {
		t.Errorf("expected points in different clusters, both got label %v", preds[0])
	}
}

func TestKMeansNClusters(t *testing.T) {
	data := mat.NewDense(6, 2, []float64{
		0, 0, 1, 0, 0, 1,
		10, 10, 11, 10, 10, 11,
	})

	for _, k := range []int{2, 3} {
		cfg := NewKMeans(WithNClusters(k), WithSeed(0))
		result, err := cfg.Fit(t.Context(), data, nil)
		if err != nil {
			t.Fatalf("Fit with NClusters=%d failed: %v", k, err)
		}
		km := result.(*KMeans)
		if len(km.Centroids) != k {
			t.Errorf("NClusters=%d: expected %d centroids, got %d", k, k, len(km.Centroids))
		}
	}
}

func TestKMeansConvergence(t *testing.T) {
	// With identical points, KMeans should converge quickly.
	data := mat.NewDense(4, 2, []float64{
		5, 5, 5, 5, 5, 5, 5, 5,
	})

	cfg := NewKMeans(WithNClusters(1), WithSeed(0), WithMaxIter(10))
	result, err := cfg.Fit(t.Context(), data, nil)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	km := result.(*KMeans)
	if km.Inertia != 0 {
		t.Errorf("expected zero inertia for identical points, got %f", km.Inertia)
	}
}

func TestKMeansInvalidNClusters(t *testing.T) {
	X := mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6})

	cfg := NewKMeans(WithNClusters(0))
	_, err := cfg.Fit(t.Context(), X, nil)
	if err == nil {
		t.Fatal("expected error for NClusters=0")
	}

	cfg = NewKMeans(WithNClusters(10))
	_, err = cfg.Fit(t.Context(), X, nil)
	if err == nil {
		t.Fatal("expected error for NClusters > nSamples")
	}
}

func TestKMeansDimensionMismatch(t *testing.T) {
	X := mat.NewDense(4, 2, []float64{0, 0, 1, 1, 2, 2, 3, 3})
	cfg := NewKMeans(WithNClusters(2), WithSeed(0))
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

func TestKMeansNInit(t *testing.T) {
	// With multiple initializations, we should get a reasonable result.
	data := make([]float64, 20*2)
	for i := 0; i < 10; i++ {
		data[i*2] = float64(i) * 0.1
		data[i*2+1] = float64(i) * 0.1
	}
	for i := 10; i < 20; i++ {
		data[i*2] = 10 + float64(i-10)*0.1
		data[i*2+1] = 10 + float64(i-10)*0.1
	}
	X := mat.NewDense(20, 2, data)

	cfg := NewKMeans(WithNClusters(2), WithSeed(7), WithNInit(5))
	result, err := cfg.Fit(t.Context(), X, nil)
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	km := result.(*KMeans)

	// Sort centroids by first coordinate for deterministic comparison.
	sort.Slice(km.Centroids, func(i, j int) bool {
		return km.Centroids[i][0] < km.Centroids[j][0]
	})

	// First centroid should be near (0.45, 0.45), second near (10.45, 10.45).
	if math.Abs(km.Centroids[0][0]-0.45) > 1.0 {
		t.Errorf("first centroid x expected near 0.45, got %f", km.Centroids[0][0])
	}
	if math.Abs(km.Centroids[1][0]-10.45) > 1.0 {
		t.Errorf("second centroid x expected near 10.45, got %f", km.Centroids[1][0])
	}
}

// Compile-time check that KMeansConfig implements glearn.Estimator.
var _ glearn.Estimator = KMeansConfig{}
var _ glearn.Predictor = (*KMeans)(nil)
