package cluster

// KMeansOption configures KMeansConfig.
type KMeansOption func(*KMeansConfig)

// WithNClusters sets the number of clusters. Default is 8.
func WithNClusters(k int) KMeansOption {
	return func(cfg *KMeansConfig) {
		cfg.NClusters = k
	}
}

// WithMaxIter sets the maximum number of Lloyd's algorithm iterations. Default is 300.
func WithMaxIter(maxIter int) KMeansOption {
	return func(cfg *KMeansConfig) {
		cfg.MaxIter = maxIter
	}
}

// WithTolerance sets the convergence tolerance based on centroid movement. Default is 1e-4.
func WithTolerance(tol float64) KMeansOption {
	return func(cfg *KMeansConfig) {
		cfg.Tolerance = tol
	}
}

// WithSeed sets the random seed for centroid initialization. Default is 0.
func WithSeed(seed int64) KMeansOption {
	return func(cfg *KMeansConfig) {
		cfg.Seed = seed
	}
}

// WithNInit sets the number of times the algorithm is run with different
// centroid seeds. The best result (lowest inertia) is kept. Default is 10.
func WithNInit(n int) KMeansOption {
	return func(cfg *KMeansConfig) {
		cfg.NInit = n
	}
}

// DBSCANOption configures DBSCANConfig.
type DBSCANOption func(*DBSCANConfig)

// WithEps sets the maximum distance between two samples for one to be
// considered as in the neighborhood of the other. Default is 0.5.
func WithEps(eps float64) DBSCANOption {
	return func(cfg *DBSCANConfig) {
		cfg.Eps = eps
	}
}

// WithMinSamples sets the number of samples in a neighborhood for a point to
// be considered a core point. Default is 5.
func WithMinSamples(minSamples int) DBSCANOption {
	return func(cfg *DBSCANConfig) {
		cfg.MinSamples = minSamples
	}
}
