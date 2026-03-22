package cluster

import "math"

// euclideanDistance computes the Euclidean distance between two points.
func euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// euclideanDistanceSquared computes the squared Euclidean distance between two points.
// This avoids the sqrt when only relative distances are needed.
func euclideanDistanceSquared(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}
