package cluster

import (
	"context"
	"fmt"
	"math"

	"github.com/freeformz/glearn"
	"github.com/freeformz/glearn/internal/validate"
	"gonum.org/v1/gonum/mat"
)

// Compile-time interface checks.
var (
	_ glearn.Estimator = DBSCANConfig{}
	_ glearn.Predictor = (*DBSCAN)(nil)
)

// DBSCANConfig holds hyperparameters for DBSCAN (Density-Based Spatial
// Clustering of Applications with Noise). It has Fit() but no Predict().
type DBSCANConfig struct {
	// Eps is the maximum distance between two samples for one to be considered
	// in the neighborhood of the other. Default is 0.5.
	Eps float64
	// MinSamples is the number of samples in a neighborhood for a point to be
	// considered a core point (including the point itself). Default is 5.
	MinSamples int
}

// NewDBSCAN creates a DBSCANConfig with the given options.
func NewDBSCAN(opts ...DBSCANOption) DBSCANConfig {
	cfg := DBSCANConfig{
		Eps:        0.5,
		MinSamples: 5,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Fit runs DBSCAN clustering on X. The y parameter is ignored (unsupervised).
// Returns a fitted DBSCAN model.
func (cfg DBSCANConfig) Fit(ctx context.Context, X *mat.Dense, y []float64) (glearn.Predictor, error) {
	nSamples, nFeatures, err := validate.Dimensions(X)
	if err != nil {
		return nil, fmt.Errorf("glearn/cluster: dbscan fit failed: %w", err)
	}

	if cfg.Eps <= 0 {
		return nil, fmt.Errorf("glearn/cluster: dbscan fit failed: %w: Eps must be positive, got %f",
			glearn.ErrInvalidParameter, cfg.Eps)
	}
	if cfg.MinSamples <= 0 {
		return nil, fmt.Errorf("glearn/cluster: dbscan fit failed: %w: MinSamples must be positive, got %d",
			glearn.ErrInvalidParameter, cfg.MinSamples)
	}

	data := extractRows(X, nSamples, nFeatures)

	// Find neighbors for each point.
	neighbors := make([][]int, nSamples)
	for i := range nSamples {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("glearn/cluster: dbscan fit cancelled: %w", ctx.Err())
		default:
		}
		for j := range nSamples {
			if euclideanDistance(data[i], data[j]) <= cfg.Eps {
				neighbors[i] = append(neighbors[i], j)
			}
		}
	}

	// Identify core points.
	isCore := make([]bool, nSamples)
	for i := range nSamples {
		if len(neighbors[i]) >= cfg.MinSamples {
			isCore[i] = true
		}
	}

	// Cluster expansion using BFS.
	labels := make([]int, nSamples)
	for i := range nSamples {
		labels[i] = -1 // -1 means noise/unvisited
	}

	clusterID := 0
	for i := range nSamples {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("glearn/cluster: dbscan fit cancelled: %w", ctx.Err())
		default:
		}

		if labels[i] != -1 || !isCore[i] {
			continue
		}

		// Start a new cluster from this core point.
		labels[i] = clusterID
		queue := make([]int, 0, len(neighbors[i]))
		for _, n := range neighbors[i] {
			if n != i {
				queue = append(queue, n)
			}
		}

		for len(queue) > 0 {
			q := queue[0]
			queue = queue[1:]

			if labels[q] == -1 {
				labels[q] = clusterID
				if isCore[q] {
					for _, n := range neighbors[q] {
						if labels[n] == -1 {
							queue = append(queue, n)
						}
					}
				}
			}
		}

		clusterID++
	}

	// Collect core sample indices.
	var coreSampleIndices []int
	for i := range nSamples {
		if isCore[i] {
			coreSampleIndices = append(coreSampleIndices, i)
		}
	}

	// Store core sample data for prediction on new points.
	coreData := make([][]float64, len(coreSampleIndices))
	coreLabels := make([]int, len(coreSampleIndices))
	for i, idx := range coreSampleIndices {
		coreData[i] = data[idx]
		coreLabels[i] = labels[idx]
	}

	return &DBSCAN{
		Labels:            labels,
		CoreSampleIndices: coreSampleIndices,
		NFeatures:         nFeatures,
		coreData:          coreData,
		coreLabels:        coreLabels,
		eps:               cfg.Eps,
	}, nil
}

// DBSCAN is a fitted DBSCAN clustering model.
// It has Predict() but no Fit().
type DBSCAN struct {
	// Labels are the cluster assignments for each training sample. -1 means noise.
	Labels []int
	// CoreSampleIndices are the indices of the core samples in the training data.
	CoreSampleIndices []int
	// NFeatures is the number of features seen during fitting.
	NFeatures int

	// coreData holds the feature vectors of core samples for transductive prediction.
	coreData [][]float64
	// coreLabels holds the cluster labels of core samples.
	coreLabels []int
	// eps is the neighborhood radius used during fitting.
	eps float64
}

// Predict assigns each row of X to the nearest core point's cluster.
// Points farther than Eps from all core points are labeled -1 (noise).
// DBSCAN is transductive; this is an approximate assignment for new data.
func (db *DBSCAN) Predict(X *mat.Dense) ([]float64, error) {
	nSamples, err := validate.PredictInputs(X, db.NFeatures)
	if err != nil {
		return nil, fmt.Errorf("glearn/cluster: dbscan predict failed: %w", err)
	}

	data := extractRows(X, nSamples, db.NFeatures)
	result := make([]float64, nSamples)

	for i, row := range data {
		bestDist := math.Inf(1)
		bestLabel := -1
		for j, core := range db.coreData {
			d := euclideanDistance(row, core)
			if d <= db.eps && d < bestDist {
				bestDist = d
				bestLabel = db.coreLabels[j]
			}
		}
		result[i] = float64(bestLabel)
	}

	return result, nil
}
