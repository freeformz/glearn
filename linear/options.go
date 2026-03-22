package linear

// LinearRegressionOption configures LinearRegressionConfig.
type LinearRegressionOption func(*LinearRegressionConfig)

// WithFitIntercept sets whether to calculate the intercept for the model.
// Default is true.
func WithFitIntercept(fit bool) LinearRegressionOption {
	return func(cfg *LinearRegressionConfig) {
		cfg.FitIntercept = fit
	}
}

// RidgeOption configures RidgeConfig.
type RidgeOption func(*RidgeConfig)

// WithRidgeAlpha sets the L2 regularization strength. Default is 1.0.
func WithRidgeAlpha(alpha float64) RidgeOption {
	return func(cfg *RidgeConfig) {
		cfg.Alpha = alpha
	}
}

// WithRidgeFitIntercept sets whether to calculate the intercept. Default is true.
func WithRidgeFitIntercept(fit bool) RidgeOption {
	return func(cfg *RidgeConfig) {
		cfg.FitIntercept = fit
	}
}

// LassoOption configures LassoConfig.
type LassoOption func(*LassoConfig)

// WithLassoAlpha sets the L1 regularization strength. Default is 1.0.
func WithLassoAlpha(alpha float64) LassoOption {
	return func(cfg *LassoConfig) {
		cfg.Alpha = alpha
	}
}

// WithLassoFitIntercept sets whether to calculate the intercept. Default is true.
func WithLassoFitIntercept(fit bool) LassoOption {
	return func(cfg *LassoConfig) {
		cfg.FitIntercept = fit
	}
}

// WithLassoMaxIter sets the maximum number of coordinate descent iterations.
// Default is 1000.
func WithLassoMaxIter(maxIter int) LassoOption {
	return func(cfg *LassoConfig) {
		cfg.MaxIter = maxIter
	}
}

// WithLassoTolerance sets the convergence tolerance. Default is 1e-4.
func WithLassoTolerance(tol float64) LassoOption {
	return func(cfg *LassoConfig) {
		cfg.Tolerance = tol
	}
}

// ElasticNetOption configures ElasticNetConfig.
type ElasticNetOption func(*ElasticNetConfig)

// WithElasticNetAlpha sets the overall regularization strength. Default is 1.0.
func WithElasticNetAlpha(alpha float64) ElasticNetOption {
	return func(cfg *ElasticNetConfig) {
		cfg.Alpha = alpha
	}
}

// WithElasticNetL1Ratio sets the mix between L1 and L2 penalty.
// 0 = pure L2, 1 = pure L1. Default is 0.5.
func WithElasticNetL1Ratio(ratio float64) ElasticNetOption {
	return func(cfg *ElasticNetConfig) {
		cfg.L1Ratio = ratio
	}
}

// WithElasticNetFitIntercept sets whether to calculate the intercept. Default is true.
func WithElasticNetFitIntercept(fit bool) ElasticNetOption {
	return func(cfg *ElasticNetConfig) {
		cfg.FitIntercept = fit
	}
}

// WithElasticNetMaxIter sets the maximum number of coordinate descent iterations.
// Default is 1000.
func WithElasticNetMaxIter(maxIter int) ElasticNetOption {
	return func(cfg *ElasticNetConfig) {
		cfg.MaxIter = maxIter
	}
}

// WithElasticNetTolerance sets the convergence tolerance. Default is 1e-4.
func WithElasticNetTolerance(tol float64) ElasticNetOption {
	return func(cfg *ElasticNetConfig) {
		cfg.Tolerance = tol
	}
}

// LogisticRegressionOption configures LogisticRegressionConfig.
type LogisticRegressionOption func(*LogisticRegressionConfig)

// WithLogisticC sets the inverse regularization strength. Default is 1.0.
// Smaller values mean stronger regularization.
func WithLogisticC(c float64) LogisticRegressionOption {
	return func(cfg *LogisticRegressionConfig) {
		cfg.C = c
	}
}

// WithLogisticFitIntercept sets whether to calculate the intercept. Default is true.
func WithLogisticFitIntercept(fit bool) LogisticRegressionOption {
	return func(cfg *LogisticRegressionConfig) {
		cfg.FitIntercept = fit
	}
}

// WithLogisticMaxIter sets the maximum number of L-BFGS iterations.
// Default is 100.
func WithLogisticMaxIter(maxIter int) LogisticRegressionOption {
	return func(cfg *LogisticRegressionConfig) {
		cfg.MaxIter = maxIter
	}
}

// WithLogisticTolerance sets the convergence tolerance for L-BFGS. Default is 1e-4.
func WithLogisticTolerance(tol float64) LogisticRegressionOption {
	return func(cfg *LogisticRegressionConfig) {
		cfg.Tolerance = tol
	}
}
