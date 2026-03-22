package cluster

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/internal/validate"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface checks.
var (
	_ glearn.Estimator = KMeansConfig{}
	_ glearn.Predictor = (*KMeans)(nil)
)

// KMeansConfig holds hyperparameters for K-Means clustering (Lloyd's algorithm)
// with KMeans++ initialization. It has Fit() but no Predict().
type KMeansConfig struct {
	// NClusters is the number of clusters to form. Default is 8.
	NClusters int
	// MaxIter is the maximum number of Lloyd's algorithm iterations. Default is 300.
	MaxIter int
	// Tolerance is the convergence threshold based on centroid movement. Default is 1e-4.
	Tolerance float64
	// Seed is the random seed for centroid initialization. Default is 0.
	Seed int64
	// NInit is the number of times the algorithm runs with different seeds.
	// The result with the lowest inertia is kept. Default is 10.
	NInit int
}

// NewKMeans creates a KMeansConfig with the given options.
func NewKMeans(opts ...KMeansOption) KMeansConfig {
	cfg := KMeansConfig{
		NClusters: 8,
		MaxIter:   300,
		Tolerance: 1e-4,
		Seed:      0,
		NInit:     10,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit runs K-Means clustering on X. The y parameter is ignored (unsupervised).
// Returns a fitted KMeans model.
func (cfg KMeansConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.Dimensions(X)
	if err != nil {
		return nil, fmt.Errorf("glearn/cluster: kmeans fit failed: %w", err)
	}

	if cfg.NClusters <= 0 {
		return nil, fmt.Errorf("glearn/cluster: kmeans fit failed: %w: NClusters must be positive, got %d",
			glearn.ErrInvalidParameter, cfg.NClusters)
	}
	if cfg.NClusters > nSamples {
		return nil, fmt.Errorf("glearn/cluster: kmeans fit failed: %w: NClusters (%d) > number of samples (%d)",
			glearn.ErrInvalidParameter, cfg.NClusters, nSamples)
	}

	// Extract raw data rows for efficient access.
	data := extractRows(X, nSamples, nFeatures)

	nInit := cfg.NInit
	if nInit < 1 {
		nInit = 1
	}

	var bestModel *KMeans

	for init := range nInit {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("glearn/cluster: kmeans fit cancelled: %w", ctx.Err())
		default:
		}

		// Use a different seed for each initialization.
		rng := rand.New(rand.NewPCG(uint64(cfg.Seed)+uint64(init), uint64(init)))

		centroids := kmeansppInit(data, cfg.NClusters, nFeatures, rng)

		labels := make([]int, nSamples)
		var inertia float64

		for iter := range cfg.MaxIter {
			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("glearn/cluster: kmeans fit cancelled: %w", ctx.Err())
			default:
			}

			// Assignment step: assign each point to the nearest centroid.
			inertia = 0
			for i, row := range data {
				bestDist := math.Inf(1)
				bestCluster := 0
				for k := range cfg.NClusters {
					d := euclideanDistanceSquared(row, centroids[k])
					if d < bestDist {
						bestDist = d
						bestCluster = k
					}
				}
				labels[i] = bestCluster
				inertia += bestDist
			}

			// Update step: recompute centroids.
			newCentroids := make([][]float64, cfg.NClusters)
			counts := make([]int, cfg.NClusters)
			for k := range cfg.NClusters {
				newCentroids[k] = make([]float64, nFeatures)
			}
			for i, row := range data {
				k := labels[i]
				counts[k]++
				for j := range nFeatures {
					newCentroids[k][j] += row[j]
				}
			}
			for k := range cfg.NClusters {
				if counts[k] > 0 {
					for j := range nFeatures {
						newCentroids[k][j] /= float64(counts[k])
					}
				} else {
					// Empty cluster: reinitialize to a random data point.
					newCentroids[k] = make([]float64, nFeatures)
					copy(newCentroids[k], data[rng.IntN(nSamples)])
				}
			}

			// Check convergence: maximum centroid movement.
			maxShift := 0.0
			for k := range cfg.NClusters {
				shift := euclideanDistanceSquared(centroids[k], newCentroids[k])
				if shift > maxShift {
					maxShift = shift
				}
			}
			centroids = newCentroids

			_ = iter // used by range
			if maxShift <= cfg.Tolerance*cfg.Tolerance {
				break
			}
		}

		model := &KMeans{
			Centroids: centroids,
			Labels:    labels,
			Inertia:   inertia,
			NFeatures: nFeatures,
		}

		if bestModel == nil || inertia < bestModel.Inertia {
			bestModel = model
		}
	}

	return bestModel, nil
}

// KMeans is a fitted K-Means clustering model.
// It has Predict() but no Fit().
type KMeans struct {
	// Centroids are the cluster center coordinates, shape [NClusters][NFeatures].
	Centroids [][]float64
	// Labels are the cluster assignments for each training sample.
	Labels []int
	// Inertia is the sum of squared distances of samples to their closest centroid.
	Inertia float64
	// NFeatures is the number of features seen during fitting.
	NFeatures int
}

// Predict assigns each row of X to the nearest centroid.
func (km *KMeans) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, err := validate.PredictInputs(X, km.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/cluster: kmeans predict failed: %w", err)
	}

	data := extractRows(X, nSamples, km.NFeatures)
	result := make([]float64, nSamples)
	for i, row := range data {
		bestDist := math.Inf(1)
		bestCluster := 0
		for k, centroid := range km.Centroids {
			d := euclideanDistanceSquared(row, centroid)
			if d < bestDist {
				bestDist = d
				bestCluster = k
			}
		}
		result[i] = float64(bestCluster)
	}
	return result, nil
}

// kmeansppInit selects initial centroids using the KMeans++ algorithm.
func kmeansppInit(data [][]float64, k, nFeatures int, rng *rand.Rand) [][]float64 {
	nSamples := len(data)
	centroids := make([][]float64, 0, k)

	// Choose the first centroid uniformly at random.
	first := make([]float64, nFeatures)
	copy(first, data[rng.IntN(nSamples)])
	centroids = append(centroids, first)

	// Distance from each point to its nearest existing centroid.
	minDist := make([]float64, nSamples)
	for i, row := range data {
		minDist[i] = euclideanDistanceSquared(row, centroids[0])
	}

	for c := 1; c < k; c++ {
		// Compute cumulative distribution proportional to D(x)^2.
		totalDist := 0.0
		for _, d := range minDist {
			totalDist += d
		}

		// Choose the next centroid with probability proportional to D(x)^2.
		target := rng.Float64() * totalDist
		cumulative := 0.0
		chosen := 0
		for i, d := range minDist {
			cumulative += d
			if cumulative >= target {
				chosen = i
				break
			}
		}

		newCentroid := make([]float64, nFeatures)
		copy(newCentroid, data[chosen])
		centroids = append(centroids, newCentroid)

		// Update minimum distances.
		for i, row := range data {
			d := euclideanDistanceSquared(row, newCentroid)
			if d < minDist[i] {
				minDist[i] = d
			}
		}
	}

	return centroids
}

// extractRows converts a mat.Dense into a slice of row slices for efficient iteration.
func extractRows(X *mat.Dense, nSamples, nFeatures int) [][]float64 {
	raw := X.RawMatrix()
	rows := make([][]float64, nSamples)
	for i := range nSamples {
		rows[i] = make([]float64, nFeatures)
		copy(rows[i], raw.Data[i*raw.Stride:i*raw.Stride+nFeatures])
	}
	return rows
}
