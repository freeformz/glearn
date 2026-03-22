package naivebayes

// GaussianNBOption configures GaussianNBConfig.
type GaussianNBOption func(*GaussianNBConfig)

// WithVarSmoothing sets the portion of the largest variance of all features
// that is added to variances for calculation stability. Default is 1e-9.
func WithVarSmoothing(smoothing float64) GaussianNBOption {
	return func(cfg *GaussianNBConfig) {
		cfg.VarSmoothing = smoothing
	}
}
