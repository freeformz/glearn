package ensemble

// --- RandomForest options ---

// RFOption configures RandomForestClassifierConfig or RandomForestRegressorConfig.
type RFOption func(*rfOptions)

type rfOptions struct {
	nTrees          int
	maxDepth        int
	minSamplesSplit int
	minSamplesLeaf  int
	seed            int64
	nJobs           int
}

func defaultRFOptions() rfOptions {
	return rfOptions{
		nTrees:          100,
		maxDepth:        -1,
		minSamplesSplit: 2,
		minSamplesLeaf:  1,
		seed:            0,
		nJobs:           1,
	}
}

// WithNTrees sets the number of trees in the forest. Default is 100.
func WithNTrees(n int) RFOption {
	return func(o *rfOptions) {
		o.nTrees = n
	}
}

// WithMaxDepth sets the maximum depth of each tree. -1 means unlimited.
// Default is -1.
func WithMaxDepth(depth int) RFOption {
	return func(o *rfOptions) {
		o.maxDepth = depth
	}
}

// WithMinSamplesSplit sets the minimum number of samples required to split
// an internal node. Default is 2.
func WithMinSamplesSplit(n int) RFOption {
	return func(o *rfOptions) {
		o.minSamplesSplit = n
	}
}

// WithMinSamplesLeaf sets the minimum number of samples required to be at
// a leaf node. Default is 1.
func WithMinSamplesLeaf(n int) RFOption {
	return func(o *rfOptions) {
		o.minSamplesLeaf = n
	}
}

// WithSeed sets the random seed for reproducibility. Default is 0.
func WithSeed(seed int64) RFOption {
	return func(o *rfOptions) {
		o.seed = seed
	}
}

// WithNJobs sets the number of parallel workers for tree training.
// Default is 1 (sequential). Use -1 for number of available CPUs.
func WithNJobs(n int) RFOption {
	return func(o *rfOptions) {
		o.nJobs = n
	}
}

// --- GradientBoosting options ---

// GBOption configures GradientBoostingClassifierConfig or GradientBoostingRegressorConfig.
type GBOption func(*gbOptions)

type gbOptions struct {
	nTrees         int
	learningRate   float64
	maxDepth       int
	minSamplesLeaf int
	seed           int64
	subsample      float64
}

func defaultGBOptions() gbOptions {
	return gbOptions{
		nTrees:         100,
		learningRate:   0.1,
		maxDepth:       3,
		minSamplesLeaf: 1,
		seed:           0,
		subsample:      1.0,
	}
}

// WithGBNTrees sets the number of boosting iterations (trees). Default is 100.
func WithGBNTrees(n int) GBOption {
	return func(o *gbOptions) {
		o.nTrees = n
	}
}

// WithGBLearningRate sets the shrinkage applied to each tree's contribution.
// Default is 0.1.
func WithGBLearningRate(rate float64) GBOption {
	return func(o *gbOptions) {
		o.learningRate = rate
	}
}

// WithGBMaxDepth sets the maximum depth of each tree. Default is 3.
func WithGBMaxDepth(depth int) GBOption {
	return func(o *gbOptions) {
		o.maxDepth = depth
	}
}

// WithGBMinSamplesLeaf sets the minimum number of samples at a leaf node.
// Default is 1.
func WithGBMinSamplesLeaf(n int) GBOption {
	return func(o *gbOptions) {
		o.minSamplesLeaf = n
	}
}

// WithGBSeed sets the random seed for reproducibility. Default is 0.
func WithGBSeed(seed int64) GBOption {
	return func(o *gbOptions) {
		o.seed = seed
	}
}

// WithGBSubsample sets the fraction of samples used for fitting each tree.
// Default is 1.0 (use all samples). Values in (0, 1] enable stochastic
// gradient boosting.
func WithGBSubsample(frac float64) GBOption {
	return func(o *gbOptions) {
		o.subsample = frac
	}
}
