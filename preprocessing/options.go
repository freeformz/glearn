package preprocessing

// Strategy is an imputation strategy for SimpleImputer.
type Strategy int

const (
	// StrategyMean replaces missing values with the column mean.
	StrategyMean Strategy = iota
	// StrategyMedian replaces missing values with the column median.
	StrategyMedian
	// StrategyConstant replaces missing values with a constant fill value.
	StrategyConstant
)

// StandardScalerOption configures a StandardScalerConfig.
type StandardScalerOption func(*StandardScalerConfig)

// WithMean controls whether to center data to zero mean. Default is true.
func WithMean(center bool) StandardScalerOption {
	return func(cfg *StandardScalerConfig) {
		cfg.WithMean = center
	}
}

// WithStd controls whether to scale data to unit variance. Default is true.
func WithStd(scale bool) StandardScalerOption {
	return func(cfg *StandardScalerConfig) {
		cfg.WithStd = scale
	}
}

// MinMaxScalerOption configures a MinMaxScalerConfig.
type MinMaxScalerOption func(*MinMaxScalerConfig)

// WithFeatureRange sets the desired range of transformed data. Default is [0, 1].
func WithFeatureRange(min, max float64) MinMaxScalerOption {
	return func(cfg *MinMaxScalerConfig) {
		cfg.FeatureMin = min
		cfg.FeatureMax = max
	}
}

// OneHotEncoderOption configures a OneHotEncoderConfig.
type OneHotEncoderOption func(*OneHotEncoderConfig)

// WithDropFirst drops the first category per feature to avoid multicollinearity.
func WithDropFirst(drop bool) OneHotEncoderOption {
	return func(cfg *OneHotEncoderConfig) {
		cfg.DropFirst = drop
	}
}

// LabelEncoderOption configures a LabelEncoderConfig.
type LabelEncoderOption func(*LabelEncoderConfig)

// SimpleImputerOption configures a SimpleImputerConfig.
type SimpleImputerOption func(*SimpleImputerConfig)

// WithStrategy sets the imputation strategy. Default is StrategyMean.
func WithStrategy(s Strategy) SimpleImputerOption {
	return func(cfg *SimpleImputerConfig) {
		cfg.Strategy = s
	}
}

// WithFillValue sets the constant fill value when Strategy is StrategyConstant.
func WithFillValue(v float64) SimpleImputerOption {
	return func(cfg *SimpleImputerConfig) {
		cfg.FillValue = v
	}
}
