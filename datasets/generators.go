package datasets

import (
	"fmt"
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

// MakeBlobs generates isotropic Gaussian blobs for clustering.
//
// Each cluster center is drawn uniformly from [-10, 10] for each feature,
// then samples are drawn from a normal distribution centered at the cluster
// center with standard deviation clusterStd.
func MakeBlobs(opts ...BlobsOption) (*mat.Dense, []float64, error) {
	cfg := defaultBlobsConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	if cfg.nSamples <= 0 {
		return nil, nil, fmt.Errorf("datasets: nSamples must be positive, got %d", cfg.nSamples)
	}
	if cfg.nFeatures <= 0 {
		return nil, nil, fmt.Errorf("datasets: nFeatures must be positive, got %d", cfg.nFeatures)
	}
	if cfg.nClusters <= 0 {
		return nil, nil, fmt.Errorf("datasets: nClusters must be positive, got %d", cfg.nClusters)
	}
	if cfg.clusterStd < 0 {
		return nil, nil, fmt.Errorf("datasets: clusterStd must be non-negative, got %f", cfg.clusterStd)
	}

	rng := rand.New(rand.NewPCG(cfg.seed, 0))

	// Generate cluster centers uniformly in [-10, 10].
	centers := make([][]float64, cfg.nClusters)
	for i := range centers {
		centers[i] = make([]float64, cfg.nFeatures)
		for j := range centers[i] {
			centers[i][j] = rng.Float64()*20.0 - 10.0
		}
	}

	features := make([]float64, cfg.nSamples*cfg.nFeatures)
	targets := make([]float64, cfg.nSamples)

	for i := range cfg.nSamples {
		cluster := i % cfg.nClusters
		targets[i] = float64(cluster)
		for j := range cfg.nFeatures {
			features[i*cfg.nFeatures+j] = centers[cluster][j] + rng.NormFloat64()*cfg.clusterStd
		}
	}

	X := mat.NewDense(cfg.nSamples, cfg.nFeatures, features)
	return X, targets, nil
}

// MakeClassification generates a random n-class classification problem.
//
// It creates nInformative features drawn from a standard normal, computes
// class labels by partitioning the informative feature space, generates
// nRedundant features as random linear combinations of the informative
// features, fills remaining features with noise, and optionally flips
// a fraction of labels.
func MakeClassification(opts ...ClassificationOption) (*mat.Dense, []float64, error) {
	cfg := defaultClassificationConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	if cfg.nSamples <= 0 {
		return nil, nil, fmt.Errorf("datasets: nSamples must be positive, got %d", cfg.nSamples)
	}
	if cfg.nFeatures <= 0 {
		return nil, nil, fmt.Errorf("datasets: nFeatures must be positive, got %d", cfg.nFeatures)
	}
	if cfg.nInformative <= 0 {
		return nil, nil, fmt.Errorf("datasets: nInformative must be positive, got %d", cfg.nInformative)
	}
	if cfg.nInformative > cfg.nFeatures {
		return nil, nil, fmt.Errorf("datasets: nInformative (%d) must be <= nFeatures (%d)", cfg.nInformative, cfg.nFeatures)
	}
	if cfg.nRedundant < 0 {
		return nil, nil, fmt.Errorf("datasets: nRedundant must be non-negative, got %d", cfg.nRedundant)
	}
	if cfg.nInformative+cfg.nRedundant > cfg.nFeatures {
		return nil, nil, fmt.Errorf("datasets: nInformative (%d) + nRedundant (%d) must be <= nFeatures (%d)", cfg.nInformative, cfg.nRedundant, cfg.nFeatures)
	}
	if cfg.nClasses <= 0 {
		return nil, nil, fmt.Errorf("datasets: nClasses must be positive, got %d", cfg.nClasses)
	}

	rng := rand.New(rand.NewPCG(cfg.seed, 0))

	features := make([]float64, cfg.nSamples*cfg.nFeatures)
	targets := make([]float64, cfg.nSamples)

	// Generate informative features from standard normal.
	for i := range cfg.nSamples {
		for j := range cfg.nInformative {
			features[i*cfg.nFeatures+j] = rng.NormFloat64() * cfg.classSep
		}
	}

	// Assign labels based on the sum of informative features.
	// We partition the range of sums into nClasses equal-width bins.
	sums := make([]float64, cfg.nSamples)
	minSum, maxSum := math.Inf(1), math.Inf(-1)
	for i := range cfg.nSamples {
		s := 0.0
		for j := range cfg.nInformative {
			s += features[i*cfg.nFeatures+j]
		}
		sums[i] = s
		if s < minSum {
			minSum = s
		}
		if s > maxSum {
			maxSum = s
		}
	}

	binWidth := (maxSum - minSum) / float64(cfg.nClasses)
	if binWidth == 0 {
		binWidth = 1.0
	}
	for i := range cfg.nSamples {
		bin := int((sums[i] - minSum) / binWidth)
		if bin >= cfg.nClasses {
			bin = cfg.nClasses - 1
		}
		targets[i] = float64(bin)
	}

	// Generate redundant features as random linear combinations of informative features.
	if cfg.nRedundant > 0 {
		// Create a mixing matrix [nRedundant x nInformative].
		mix := make([]float64, cfg.nRedundant*cfg.nInformative)
		for k := range mix {
			mix[k] = rng.NormFloat64()
		}
		for i := range cfg.nSamples {
			for r := range cfg.nRedundant {
				val := 0.0
				for j := range cfg.nInformative {
					val += mix[r*cfg.nInformative+j] * features[i*cfg.nFeatures+j]
				}
				features[i*cfg.nFeatures+cfg.nInformative+r] = val
			}
		}
	}

	// Fill remaining features with noise.
	noiseStart := cfg.nInformative + cfg.nRedundant
	for i := range cfg.nSamples {
		for j := noiseStart; j < cfg.nFeatures; j++ {
			features[i*cfg.nFeatures+j] = rng.NormFloat64()
		}
	}

	// Flip labels.
	if cfg.flipY > 0 {
		for i := range cfg.nSamples {
			if rng.Float64() < cfg.flipY {
				targets[i] = float64(rng.IntN(cfg.nClasses))
			}
		}
	}

	X := mat.NewDense(cfg.nSamples, cfg.nFeatures, features)
	return X, targets, nil
}

// MakeRegression generates a random regression problem.
//
// It creates features from a standard normal distribution and a target
// as a linear combination of nInformative features plus optional Gaussian noise
// and a bias term. The ground truth coefficients are drawn from a standard normal.
func MakeRegression(opts ...RegressionOption) (*mat.Dense, []float64, error) {
	cfg := defaultRegressionConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	if cfg.nSamples <= 0 {
		return nil, nil, fmt.Errorf("datasets: nSamples must be positive, got %d", cfg.nSamples)
	}
	if cfg.nFeatures <= 0 {
		return nil, nil, fmt.Errorf("datasets: nFeatures must be positive, got %d", cfg.nFeatures)
	}
	if cfg.nInformative <= 0 {
		return nil, nil, fmt.Errorf("datasets: nInformative must be positive, got %d", cfg.nInformative)
	}
	if cfg.nInformative > cfg.nFeatures {
		return nil, nil, fmt.Errorf("datasets: nInformative (%d) must be <= nFeatures (%d)", cfg.nInformative, cfg.nFeatures)
	}

	rng := rand.New(rand.NewPCG(cfg.seed, 0))

	features := make([]float64, cfg.nSamples*cfg.nFeatures)
	targets := make([]float64, cfg.nSamples)

	// Generate all features from standard normal.
	for i := range features {
		features[i] = rng.NormFloat64()
	}

	// Generate ground truth coefficients for the informative features.
	coefs := make([]float64, cfg.nInformative)
	for i := range coefs {
		coefs[i] = rng.NormFloat64()
	}

	// Compute targets as a linear combination of informative features + noise + bias.
	for i := range cfg.nSamples {
		y := cfg.bias
		for j := range cfg.nInformative {
			y += features[i*cfg.nFeatures+j] * coefs[j]
		}
		if cfg.noise > 0 {
			y += rng.NormFloat64() * cfg.noise
		}
		targets[i] = y
	}

	X := mat.NewDense(cfg.nSamples, cfg.nFeatures, features)
	return X, targets, nil
}
