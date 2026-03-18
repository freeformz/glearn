# glearn Development Guide

## Project Overview

glearn is a comprehensive classical ML library for Go, inspired by scikit-learn and ferrolearn.
See `docs/prd.md` for the full product requirements document.

## Build & Test

```bash
go build ./...          # Build all packages
go test ./...           # Run all tests
go test -race ./...     # Run tests with race detector
go test -bench=. ./...  # Run benchmarks
```

### Optional: Link OpenBLAS for production performance
```bash
CGO_ENABLED=1 go test -tags netlib ./...
```

## Core Design Pattern

Every algorithm uses the **hybrid builder pattern** with two types:

```go
// Config type — unfitted, has Fit(), NO Predict()
type LinearRegressionConfig struct { ... }

// Fitted type — has Predict(), NO Fit()
type LinearRegression struct { ... }
```

### Rules
- Config types hold hyperparameters only
- Fitted types hold learned parameters as exported fields
- `Fit()` returns `(Predictor, error)` to satisfy the `Estimator` interface
- Fitted models are immutable after construction — safe for concurrent `Predict()`
- All `Fit()` methods accept `context.Context` as first parameter
- Never use the global rand source — accept a seed or `*rand.Rand`

## Interfaces

Core interfaces are in the root `glearn` package (`glearn.go`):
- `Estimator` — `Fit(ctx, X, y) -> (Predictor, error)`
- `Predictor` — `Predict(X) -> ([]float64, error)`
- `Transformer` — `Fit(ctx, X) -> (FittedTransformer, error)`
- `FittedTransformer` — `Transform(X) -> (*mat.Dense, error)`
- `Scorer` — `Score(X, y) -> (float64, error)`

## Error Handling

- Use sentinel errors from the root package: `ErrDimensionMismatch`, `ErrSingularMatrix`, etc.
- Wrap with context: `fmt.Errorf("glearn/linear: ridge fit failed: %w", err)`
- Never panic for data-dependent failures
- Validate inputs at Fit/Predict time, not at construction time

## Package Dependencies

- Root `glearn` package has no sub-package dependencies
- Algorithm packages depend on root + `internal/` only
- `pipeline/` depends on root interfaces only
- `modelselection/` depends on root + `metrics/`
- No circular imports allowed

## Testing

- Unit tests for every public function
- Oracle tests compare against scikit-learn fixtures in `testdata/` directories
- Generate fixtures: `python scripts/generate_fixtures.py`
- Use relative tolerance for float comparisons
- Compile-time verification: `var _ glearn.Estimator = SomeConfig{}`

## Dependencies

- `gonum.org/v1/gonum` — linear algebra, statistics, optimization
- `github.com/vmihailenco/msgpack/v5` — MessagePack serialization
- Minimize dependencies: prefer stdlib where possible
