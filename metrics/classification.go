package metrics

// Accuracy returns the fraction of predictions that exactly match the true labels.
// It panics if the inputs are empty or have different lengths.
func Accuracy(yTrue, yPred []float64) float64 {
	validateInputs(yTrue, yPred)

	correct := 0
	for i := range yTrue {
		if yTrue[i] == yPred[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(yTrue))
}

// Precision computes the precision score. For binary classification (the default),
// it returns TP / (TP + FP) for the positive class (label 1). For multi-class,
// use WithAverage to select the averaging strategy.
//
// It panics if the inputs are empty or have different lengths.
func Precision(yTrue, yPred []float64, opts ...Option) float64 {
	validateInputs(yTrue, yPred)
	o := applyOptions(opts)

	classes := uniqueClasses(yTrue, yPred)
	tp, fp, _, support := perClassCounts(yTrue, yPred, classes)

	return computeAverage(o.average, classes, tp, fp, nil, support, func(tp, denom float64) float64 {
		if denom == 0 {
			return 0
		}
		return tp / denom
	}, func(classTP, classFP, _ float64) float64 {
		return classTP + classFP
	})
}

// Recall computes the recall (sensitivity) score. For binary classification
// (the default), it returns TP / (TP + FN) for the positive class (label 1).
// For multi-class, use WithAverage to select the averaging strategy.
//
// It panics if the inputs are empty or have different lengths.
func Recall(yTrue, yPred []float64, opts ...Option) float64 {
	validateInputs(yTrue, yPred)
	o := applyOptions(opts)

	classes := uniqueClasses(yTrue, yPred)
	tp, _, fn, support := perClassCounts(yTrue, yPred, classes)

	return computeAverage(o.average, classes, tp, nil, fn, support, func(tp, denom float64) float64 {
		if denom == 0 {
			return 0
		}
		return tp / denom
	}, func(classTP, _, classFN float64) float64 {
		return classTP + classFN
	})
}

// F1 computes the F1 score, the harmonic mean of precision and recall.
// For binary classification (the default), it operates on the positive class
// (label 1). For multi-class, use WithAverage to select the averaging strategy.
//
// It panics if the inputs are empty or have different lengths.
func F1(yTrue, yPred []float64, opts ...Option) float64 {
	validateInputs(yTrue, yPred)
	o := applyOptions(opts)

	classes := uniqueClasses(yTrue, yPred)
	tp, fp, fn, support := perClassCounts(yTrue, yPred, classes)

	switch o.average {
	case AverageMicro:
		totalTP := 0.0
		totalFP := 0.0
		totalFN := 0.0
		for _, c := range classes {
			totalTP += tp[c]
			totalFP += fp[c]
			totalFN += fn[c]
		}
		p := 0.0
		if totalTP+totalFP > 0 {
			p = totalTP / (totalTP + totalFP)
		}
		r := 0.0
		if totalTP+totalFN > 0 {
			r = totalTP / (totalTP + totalFN)
		}
		if p+r == 0 {
			return 0
		}
		return 2 * p * r / (p + r)

	case AverageMacro:
		sum := 0.0
		for _, c := range classes {
			sum += f1ForClass(tp[c], fp[c], fn[c])
		}
		return sum / float64(len(classes))

	case AverageWeighted:
		totalSupport := 0.0
		for _, c := range classes {
			totalSupport += support[c]
		}
		if totalSupport == 0 {
			return 0
		}
		sum := 0.0
		for _, c := range classes {
			sum += f1ForClass(tp[c], fp[c], fn[c]) * support[c]
		}
		return sum / totalSupport

	default: // binary
		return f1ForClass(tp[1.0], fp[1.0], fn[1.0])
	}
}

// ConfusionMatrix computes the confusion matrix for classification results.
// The returned matrix C has shape [n_classes][n_classes] where C[i][j] is the
// number of observations known to be in class i but predicted to be in class j.
// Classes are sorted in ascending order.
//
// It panics if the inputs are empty or have different lengths.
func ConfusionMatrix(yTrue, yPred []float64) [][]int {
	validateInputs(yTrue, yPred)

	classes := uniqueClasses(yTrue, yPred)
	classIdx := make(map[float64]int, len(classes))
	for i, c := range classes {
		classIdx[c] = i
	}

	n := len(classes)
	cm := make([][]int, n)
	for i := range cm {
		cm[i] = make([]int, n)
	}

	for i := range yTrue {
		row := classIdx[yTrue[i]]
		col := classIdx[yPred[i]]
		cm[row][col]++
	}

	return cm
}

// validateInputs panics if yTrue or yPred are empty or have different lengths.
func validateInputs(yTrue, yPred []float64) {
	if len(yTrue) == 0 || len(yPred) == 0 {
		panic("metrics: input slices must not be empty")
	}
	if len(yTrue) != len(yPred) {
		panic("metrics: input slices must have the same length")
	}
}

// uniqueClasses returns the sorted unique class labels from both slices.
func uniqueClasses(yTrue, yPred []float64) []float64 {
	set := make(map[float64]struct{})
	for _, v := range yTrue {
		set[v] = struct{}{}
	}
	for _, v := range yPred {
		set[v] = struct{}{}
	}

	classes := make([]float64, 0, len(set))
	for c := range set {
		classes = append(classes, c)
	}

	// Sort using insertion sort (small number of classes typically).
	for i := 1; i < len(classes); i++ {
		key := classes[i]
		j := i - 1
		for j >= 0 && classes[j] > key {
			classes[j+1] = classes[j]
			j--
		}
		classes[j+1] = key
	}

	return classes
}

// perClassCounts computes per-class true positives, false positives,
// false negatives, and support (number of true instances).
func perClassCounts(yTrue, yPred []float64, classes []float64) (
	tp, fp, fn, support map[float64]float64,
) {
	tp = make(map[float64]float64, len(classes))
	fp = make(map[float64]float64, len(classes))
	fn = make(map[float64]float64, len(classes))
	support = make(map[float64]float64, len(classes))

	for _, c := range classes {
		tp[c] = 0
		fp[c] = 0
		fn[c] = 0
		support[c] = 0
	}

	for i := range yTrue {
		actual := yTrue[i]
		predicted := yPred[i]
		support[actual]++

		if actual == predicted {
			tp[actual]++
		} else {
			fn[actual]++
			fp[predicted]++
		}
	}

	return tp, fp, fn, support
}

// computeAverage applies the selected averaging strategy to per-class metrics.
// metricFn computes the metric from (numerator, denominator).
// denomFn computes the denominator from (tp, fp, fn) for a class.
func computeAverage(
	avg Average,
	classes []float64,
	tp, fp, fn, support map[float64]float64,
	metricFn func(tp, denom float64) float64,
	denomFn func(classTP, classFP, classFN float64) float64,
) float64 {
	switch avg {
	case AverageMicro:
		totalTP := 0.0
		totalDenomParts := [3]float64{} // tp, fp, fn totals
		for _, c := range classes {
			totalTP += tp[c]
			if fp != nil {
				totalDenomParts[1] += fp[c]
			}
			if fn != nil {
				totalDenomParts[2] += fn[c]
			}
		}
		denom := denomFn(totalTP, totalDenomParts[1], totalDenomParts[2])
		return metricFn(totalTP, denom)

	case AverageMacro:
		sum := 0.0
		for _, c := range classes {
			fpVal := 0.0
			if fp != nil {
				fpVal = fp[c]
			}
			fnVal := 0.0
			if fn != nil {
				fnVal = fn[c]
			}
			denom := denomFn(tp[c], fpVal, fnVal)
			sum += metricFn(tp[c], denom)
		}
		return sum / float64(len(classes))

	case AverageWeighted:
		totalSupport := 0.0
		for _, c := range classes {
			totalSupport += support[c]
		}
		if totalSupport == 0 {
			return 0
		}
		sum := 0.0
		for _, c := range classes {
			fpVal := 0.0
			if fp != nil {
				fpVal = fp[c]
			}
			fnVal := 0.0
			if fn != nil {
				fnVal = fn[c]
			}
			denom := denomFn(tp[c], fpVal, fnVal)
			sum += metricFn(tp[c], denom) * support[c]
		}
		return sum / totalSupport

	default: // binary — compute for positive class (1.0)
		fpVal := 0.0
		if fp != nil {
			fpVal = fp[1.0]
		}
		fnVal := 0.0
		if fn != nil {
			fnVal = fn[1.0]
		}
		denom := denomFn(tp[1.0], fpVal, fnVal)
		return metricFn(tp[1.0], denom)
	}
}

// f1ForClass computes the F1 score for a single class from its TP, FP, FN counts.
func f1ForClass(tp, fp, fn float64) float64 {
	p := 0.0
	if tp+fp > 0 {
		p = tp / (tp + fp)
	}
	r := 0.0
	if tp+fn > 0 {
		r = tp / (tp + fn)
	}
	if p+r == 0 {
		return 0
	}
	return 2 * p * r / (p + r)
}
