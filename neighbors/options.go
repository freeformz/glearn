package neighbors

// Weights specifies the weight function used in prediction.
type Weights string

const (
	// WeightsUniform gives equal weight to all neighbors.
	WeightsUniform Weights = "uniform"
	// WeightsDistance weights neighbors by the inverse of their distance.
	WeightsDistance Weights = "distance"
)

// KNeighborsClassifierOption configures KNeighborsClassifierConfig.
type KNeighborsClassifierOption func(*KNeighborsClassifierConfig)

// WithClassifierK sets the number of neighbors. Default is 5.
func WithClassifierK(k int) KNeighborsClassifierOption {
	return func(cfg *KNeighborsClassifierConfig) {
		cfg.K = k
	}
}

// WithClassifierWeights sets the weight function. Default is "uniform".
func WithClassifierWeights(w Weights) KNeighborsClassifierOption {
	return func(cfg *KNeighborsClassifierConfig) {
		cfg.Weights = w
	}
}

// KNeighborsRegressorOption configures KNeighborsRegressorConfig.
type KNeighborsRegressorOption func(*KNeighborsRegressorConfig)

// WithRegressorK sets the number of neighbors. Default is 5.
func WithRegressorK(k int) KNeighborsRegressorOption {
	return func(cfg *KNeighborsRegressorConfig) {
		cfg.K = k
	}
}

// WithRegressorWeights sets the weight function. Default is "uniform".
func WithRegressorWeights(w Weights) KNeighborsRegressorOption {
	return func(cfg *KNeighborsRegressorConfig) {
		cfg.Weights = w
	}
}
