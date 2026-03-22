package tree

// ClassifierOption configures DecisionTreeClassifierConfig.
type ClassifierOption func(*DecisionTreeClassifierConfig)

// WithClassifierMaxDepth sets the maximum depth of the tree.
// -1 means unlimited. Default is -1.
func WithClassifierMaxDepth(depth int) ClassifierOption {
	return func(cfg *DecisionTreeClassifierConfig) {
		cfg.MaxDepth = depth
	}
}

// WithClassifierMinSamplesSplit sets the minimum number of samples required
// to split an internal node. Default is 2.
func WithClassifierMinSamplesSplit(n int) ClassifierOption {
	return func(cfg *DecisionTreeClassifierConfig) {
		cfg.MinSamplesSplit = n
	}
}

// WithClassifierMinSamplesLeaf sets the minimum number of samples required
// to be at a leaf node. Default is 1.
func WithClassifierMinSamplesLeaf(n int) ClassifierOption {
	return func(cfg *DecisionTreeClassifierConfig) {
		cfg.MinSamplesLeaf = n
	}
}

// WithClassifierCriterion sets the impurity criterion for splitting.
// Supported values: "gini" (default), "entropy".
func WithClassifierCriterion(criterion string) ClassifierOption {
	return func(cfg *DecisionTreeClassifierConfig) {
		cfg.Criterion = criterion
	}
}

// WithClassifierSeed sets the random seed for reproducibility.
// Default is 0.
func WithClassifierSeed(seed int64) ClassifierOption {
	return func(cfg *DecisionTreeClassifierConfig) {
		cfg.Seed = seed
	}
}

// RegressorOption configures DecisionTreeRegressorConfig.
type RegressorOption func(*DecisionTreeRegressorConfig)

// WithRegressorMaxDepth sets the maximum depth of the tree.
// -1 means unlimited. Default is -1.
func WithRegressorMaxDepth(depth int) RegressorOption {
	return func(cfg *DecisionTreeRegressorConfig) {
		cfg.MaxDepth = depth
	}
}

// WithRegressorMinSamplesSplit sets the minimum number of samples required
// to split an internal node. Default is 2.
func WithRegressorMinSamplesSplit(n int) RegressorOption {
	return func(cfg *DecisionTreeRegressorConfig) {
		cfg.MinSamplesSplit = n
	}
}

// WithRegressorMinSamplesLeaf sets the minimum number of samples required
// to be at a leaf node. Default is 1.
func WithRegressorMinSamplesLeaf(n int) RegressorOption {
	return func(cfg *DecisionTreeRegressorConfig) {
		cfg.MinSamplesLeaf = n
	}
}

// WithRegressorCriterion sets the impurity criterion for splitting.
// Supported values: "mse" (default).
func WithRegressorCriterion(criterion string) RegressorOption {
	return func(cfg *DecisionTreeRegressorConfig) {
		cfg.Criterion = criterion
	}
}

// WithRegressorSeed sets the random seed for reproducibility.
// Default is 0.
func WithRegressorSeed(seed int64) RegressorOption {
	return func(cfg *DecisionTreeRegressorConfig) {
		cfg.Seed = seed
	}
}
