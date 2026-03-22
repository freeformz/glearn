package neighbors

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// neighbor holds the index and distance of a training sample to a query point.
type neighbor struct {
	index    int
	distance float64
}

// euclideanDistance computes the Euclidean distance between two row vectors.
func euclideanDistance(a, b []float64) float64 {
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// findKNearest returns the K nearest neighbors for each row in query against
// the training data. Returns a slice of slices of neighbors (one per query row).
func findKNearest(trainX *mat.Dense, query *mat.Dense, k int) [][]neighbor {
	nTrain, nFeatures := trainX.Dims()
	nQuery, _ := query.Dims()
	rawTrain := trainX.RawMatrix()
	rawQuery := query.RawMatrix()

	results := make([][]neighbor, nQuery)
	for i := range nQuery {
		qRow := rawQuery.Data[i*rawQuery.Stride : i*rawQuery.Stride+nFeatures]

		neighbors := make([]neighbor, nTrain)
		for j := range nTrain {
			tRow := rawTrain.Data[j*rawTrain.Stride : j*rawTrain.Stride+nFeatures]
			neighbors[j] = neighbor{
				index:    j,
				distance: euclideanDistance(qRow, tRow),
			}
		}

		// Partial sort: only need K smallest.
		sort.Slice(neighbors, func(a, b int) bool {
			return neighbors[a].distance < neighbors[b].distance
		})

		results[i] = neighbors[:k]
	}
	return results
}
