package glearn

import "errors"

// Sentinel errors for common failure modes across glearn packages.
// Algorithm-specific errors wrap these with context via fmt.Errorf.
var (
	// ErrDimensionMismatch indicates that input dimensions do not match
	// what the model expects (e.g., predicting with a different number
	// of features than were used during fitting).
	ErrDimensionMismatch = errors.New("glearn: dimension mismatch")

	// ErrSingularMatrix indicates that a matrix operation failed because
	// the matrix is singular or nearly singular.
	ErrSingularMatrix = errors.New("glearn: singular matrix")

	// ErrConvergence indicates that an iterative algorithm failed to
	// converge within the maximum number of iterations.
	ErrConvergence = errors.New("glearn: failed to converge")

	// ErrEmptyInput indicates that the input data is empty (zero samples
	// or zero features).
	ErrEmptyInput = errors.New("glearn: empty input")

	// ErrInvalidParameter indicates that a hyperparameter value is invalid
	// (e.g., negative number of clusters, learning rate of zero).
	ErrInvalidParameter = errors.New("glearn: invalid parameter")
)
