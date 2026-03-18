package metrics

// Average specifies the averaging strategy for multi-class metrics.
type Average string

const (
	// AverageBinary computes the metric for the positive class only (label 1).
	// This is the default for binary classification.
	AverageBinary Average = "binary"

	// AverageMicro computes the metric globally by counting total true positives,
	// false negatives, and false positives.
	AverageMicro Average = "micro"

	// AverageMacro computes the metric for each class and returns the unweighted mean.
	AverageMacro Average = "macro"

	// AverageWeighted computes the metric for each class and returns the
	// weighted mean by support (number of true instances per class).
	AverageWeighted Average = "weighted"
)

// options holds configuration for metrics that accept optional parameters.
type options struct {
	average Average
}

// Option configures a metric function.
type Option func(*options)

// WithAverage sets the averaging strategy for multi-class metrics.
// Valid values are AverageBinary (default), AverageMicro, AverageMacro, and AverageWeighted.
func WithAverage(avg Average) Option {
	return func(o *options) {
		o.average = avg
	}
}

func defaultOptions() options {
	return options{
		average: AverageBinary,
	}
}

func applyOptions(opts []Option) options {
	o := defaultOptions()
	for _, fn := range opts {
		fn(&o)
	}
	return o
}
