// Package mathutil provides shared numerical utility functions for glearn algorithms.
package mathutil

import "math"

// Sum returns the sum of all elements in s.
func Sum(s []float64) float64 {
	var total float64
	for _, v := range s {
		total += v
	}
	return total
}

// Mean returns the arithmetic mean of s. Panics if s is empty.
func Mean(s []float64) float64 {
	return Sum(s) / float64(len(s))
}

// Variance returns the population variance of s. Panics if s is empty.
func Variance(s []float64) float64 {
	m := Mean(s)
	var ss float64
	for _, v := range s {
		d := v - m
		ss += d * d
	}
	return ss / float64(len(s))
}

// Std returns the population standard deviation of s. Panics if s is empty.
func Std(s []float64) float64 {
	return math.Sqrt(Variance(s))
}

// ArgMax returns the index of the maximum value in s. Panics if s is empty.
func ArgMax(s []float64) int {
	maxIdx := 0
	maxVal := s[0]
	for i, v := range s[1:] {
		if v > maxVal {
			maxVal = v
			maxIdx = i + 1
		}
	}
	return maxIdx
}

// Unique returns the unique elements of s in the order they first appear.
func Unique(s []float64) []float64 {
	seen := make(map[float64]struct{})
	var result []float64
	for _, v := range s {
		if _, ok := seen[v]; !ok {
			seen[v] = struct{}{}
			result = append(result, v)
		}
	}
	return result
}
