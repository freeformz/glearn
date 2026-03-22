package decomposition

// PCAOption configures PCAConfig.
type PCAOption func(*PCAConfig)

// WithNComponents sets the number of principal components to keep.
// Default is min(nSamples, nFeatures).
func WithNComponents(n int) PCAOption {
	return func(cfg *PCAConfig) {
		cfg.NComponents = n
	}
}
