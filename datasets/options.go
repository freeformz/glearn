package datasets

// BlobsOption configures MakeBlobs.
type BlobsOption func(*blobsConfig)

type blobsConfig struct {
	nSamples   int
	nFeatures  int
	nClusters  int
	clusterStd float64
	seed       uint64
}

func defaultBlobsConfig() blobsConfig {
	return blobsConfig{
		nSamples:   100,
		nFeatures:  2,
		nClusters:  3,
		clusterStd: 1.0,
		seed:       0,
	}
}

// WithBlobsNSamples sets the total number of samples.
func WithBlobsNSamples(n int) BlobsOption {
	return func(c *blobsConfig) { c.nSamples = n }
}

// WithBlobsNFeatures sets the number of features per sample.
func WithBlobsNFeatures(n int) BlobsOption {
	return func(c *blobsConfig) { c.nFeatures = n }
}

// WithBlobsNClusters sets the number of Gaussian blob centers.
func WithBlobsNClusters(n int) BlobsOption {
	return func(c *blobsConfig) { c.nClusters = n }
}

// WithBlobsClusterStd sets the standard deviation of each cluster.
func WithBlobsClusterStd(std float64) BlobsOption {
	return func(c *blobsConfig) { c.clusterStd = std }
}

// WithBlobsSeed sets the random seed for reproducibility.
func WithBlobsSeed(seed uint64) BlobsOption {
	return func(c *blobsConfig) { c.seed = seed }
}

// ClassificationOption configures MakeClassification.
type ClassificationOption func(*classificationConfig)

type classificationConfig struct {
	nSamples           int
	nFeatures          int
	nInformative       int
	nRedundant         int
	nClasses           int
	flipY              float64
	classSep           float64
	seed               uint64
}

func defaultClassificationConfig() classificationConfig {
	return classificationConfig{
		nSamples:     100,
		nFeatures:    20,
		nInformative: 2,
		nRedundant:   2,
		nClasses:     2,
		flipY:        0.01,
		classSep:     1.0,
		seed:         0,
	}
}

// WithClassificationNSamples sets the total number of samples.
func WithClassificationNSamples(n int) ClassificationOption {
	return func(c *classificationConfig) { c.nSamples = n }
}

// WithClassificationNFeatures sets the total number of features.
func WithClassificationNFeatures(n int) ClassificationOption {
	return func(c *classificationConfig) { c.nFeatures = n }
}

// WithClassificationNInformative sets the number of informative features.
func WithClassificationNInformative(n int) ClassificationOption {
	return func(c *classificationConfig) { c.nInformative = n }
}

// WithClassificationNRedundant sets the number of redundant features.
func WithClassificationNRedundant(n int) ClassificationOption {
	return func(c *classificationConfig) { c.nRedundant = n }
}

// WithClassificationNClasses sets the number of classes.
func WithClassificationNClasses(n int) ClassificationOption {
	return func(c *classificationConfig) { c.nClasses = n }
}

// WithClassificationFlipY sets the fraction of samples whose class is randomly flipped.
func WithClassificationFlipY(f float64) ClassificationOption {
	return func(c *classificationConfig) { c.flipY = f }
}

// WithClassificationClassSep sets the factor multiplying the hypercube size.
func WithClassificationClassSep(s float64) ClassificationOption {
	return func(c *classificationConfig) { c.classSep = s }
}

// WithClassificationSeed sets the random seed for reproducibility.
func WithClassificationSeed(seed uint64) ClassificationOption {
	return func(c *classificationConfig) { c.seed = seed }
}

// RegressionOption configures MakeRegression.
type RegressionOption func(*regressionConfig)

type regressionConfig struct {
	nSamples     int
	nFeatures    int
	nInformative int
	noise        float64
	bias         float64
	seed         uint64
}

func defaultRegressionConfig() regressionConfig {
	return regressionConfig{
		nSamples:     100,
		nFeatures:    100,
		nInformative: 10,
		noise:        0.0,
		bias:         0.0,
		seed:         0,
	}
}

// WithRegressionNSamples sets the total number of samples.
func WithRegressionNSamples(n int) RegressionOption {
	return func(c *regressionConfig) { c.nSamples = n }
}

// WithRegressionNFeatures sets the total number of features.
func WithRegressionNFeatures(n int) RegressionOption {
	return func(c *regressionConfig) { c.nFeatures = n }
}

// WithRegressionNInformative sets the number of informative features.
func WithRegressionNInformative(n int) RegressionOption {
	return func(c *regressionConfig) { c.nInformative = n }
}

// WithRegressionNoise sets the standard deviation of Gaussian noise applied to the output.
func WithRegressionNoise(noise float64) RegressionOption {
	return func(c *regressionConfig) { c.noise = noise }
}

// WithRegressionBias sets the bias term in the underlying linear model.
func WithRegressionBias(bias float64) RegressionOption {
	return func(c *regressionConfig) { c.bias = bias }
}

// WithRegressionSeed sets the random seed for reproducibility.
func WithRegressionSeed(seed uint64) RegressionOption {
	return func(c *regressionConfig) { c.seed = seed }
}
