package metrics

import "math"

// MAE returns the mean absolute error between yTrue and yPred.
// It panics if the inputs are empty or have different lengths.
func MAE(yTrue, yPred []float64) float64 {
	validateInputs(yTrue, yPred)

	sum := 0.0
	for i := range yTrue {
		sum += math.Abs(yTrue[i] - yPred[i])
	}
	return sum / float64(len(yTrue))
}

// MSE returns the mean squared error between yTrue and yPred.
// It panics if the inputs are empty or have different lengths.
func MSE(yTrue, yPred []float64) float64 {
	validateInputs(yTrue, yPred)

	sum := 0.0
	for i := range yTrue {
		diff := yTrue[i] - yPred[i]
		sum += diff * diff
	}
	return sum / float64(len(yTrue))
}

// RMSE returns the root mean squared error between yTrue and yPred.
// It panics if the inputs are empty or have different lengths.
func RMSE(yTrue, yPred []float64) float64 {
	return math.Sqrt(MSE(yTrue, yPred))
}

// R2 returns the coefficient of determination (R-squared) between yTrue and yPred.
// It measures how well predictions approximate the true values, with 1.0 being
// perfect prediction. It can be negative if predictions are worse than predicting
// the mean.
//
// It panics if the inputs are empty or have different lengths.
func R2(yTrue, yPred []float64) float64 {
	validateInputs(yTrue, yPred)

	// Compute mean of yTrue.
	mean := 0.0
	for _, v := range yTrue {
		mean += v
	}
	mean /= float64(len(yTrue))

	ssRes := 0.0
	ssTot := 0.0
	for i := range yTrue {
		diff := yTrue[i] - yPred[i]
		ssRes += diff * diff

		diffMean := yTrue[i] - mean
		ssTot += diffMean * diffMean
	}

	if ssTot == 0 {
		// All true values are identical. If predictions are also identical
		// and equal to the true values, R2 is 1; otherwise it's 0.
		if ssRes == 0 {
			return 1.0
		}
		return 0.0
	}

	return 1.0 - ssRes/ssTot
}
