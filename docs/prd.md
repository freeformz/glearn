# glearn - Product Requirements Document

A comprehensive classical machine learning library for Go, inspired by [ferrolearn](https://github.com/dollspace-gay/ferrolearn) (Rust) and [scikit-learn](https://scikit-learn.org/) (Python).

**Module path:** `github.com/freeformz/glearn`
**Minimum Go version:** Go 1.26
**License:** MIT + Apache-2.0 (dual-licensed)

---

## 1. Vision

glearn aims to be the definitive classical ML library for Go — comprehensive, idiomatic, and production-ready. It brings the breadth of scikit-learn's algorithms to the Go ecosystem while leveraging Go's strengths: compile-time safety, built-in concurrency, and simple deployment.

### Goals

- Full parity with ferrolearn's algorithm coverage (~60+ algorithms)
- Compile-time safety: calling `Predict()` on an unfitted model does not compile
- Idiomatic Go API using interfaces, functional options, and error returns
- Built on gonum for linear algebra
- Production-ready: serialization, concurrency-safe fitted models, context-aware long operations
- Oracle-tested against scikit-learn reference outputs

### Non-Goals

- Deep learning / neural networks (use Gorgonia or call out to PyTorch/TF)
- GPU acceleration (initial release)
- Python bindings
- Real-time / streaming inference server (glearn is a library, not a service)

---

## 2. Core Design: The Hybrid Builder Pattern

glearn uses a **hybrid interface narrowing + builder pattern** for compile-time safety. Every model has two types: a config (unfitted) and a fitted model.

### Type Pattern

```go
// Config type — has Fit(), no Predict(). Holds hyperparameters only.
type LinearRegressionConfig struct {
    FitIntercept bool
}

// Fitted type — has Predict(), no Fit(). Holds learned parameters.
// The "natural" name is the fitted model — the thing you actually use.
type LinearRegression struct {
    Coefficients []float64
    Intercept    float64
}

func (cfg LinearRegressionConfig) Fit(ctx context.Context, X, y *mat.Dense) (*LinearRegression, error) {
    // ... train ...
    return &LinearRegression{Coefficients: coefs, Intercept: intercept}, nil
}

func (lr *LinearRegression) Predict(X *mat.Dense) ([]float64, error) {
    // ... predict ...
}
```

### Core Interfaces

```go
// Estimator is an unfitted model that can be trained.
type Estimator interface {
    Fit(ctx context.Context, X *mat.Dense, y []float64) (Predictor, error)
}

// Predictor is a fitted model that can make predictions.
type Predictor interface {
    Predict(X *mat.Dense) ([]float64, error)
}

// Transformer is an unfitted data transformer.
type Transformer interface {
    Fit(ctx context.Context, X *mat.Dense) (FittedTransformer, error)
}

// FittedTransformer can transform new data.
type FittedTransformer interface {
    Transform(X *mat.Dense) (*mat.Dense, error)
}

// Scorer evaluates a fitted model on test data.
type Scorer interface {
    Score(X *mat.Dense, y []float64) (float64, error)
}
```

### Key Design Properties

- **Compile-time safety:** `LinearRegressionConfig` has no `Predict()` method — calling it won't compile
- **Immutable fitted models:** fitted models don't mutate after construction — safe for concurrent reads
- **Interface composability:** `Estimator.Fit()` returns `Predictor` (interface), enabling Pipeline, GridSearchCV, etc. to work generically
- **Direct field access:** learned parameters are exported fields on the fitted type (`model.Coefficients`)
- **Easy serialization:** fitted models are concrete structs with exported fields — JSON/gob/msgpack just works
- **Context-aware:** all `Fit()` methods accept `context.Context` for cancellation of long-running training

### Configuration via Functional Options

```go
cfg := linear.NewLinearRegression(
    linear.WithFitIntercept(true),
    linear.WithNormalize(false),
)
// Or struct literal:
cfg := linear.LinearRegressionConfig{FitIntercept: true}
```

### Concurrency Model

- Fitted models are immutable after construction — safe for concurrent `Predict()` calls
- Models are NOT safe for concurrent `Fit()` — use separate config instances
- For serving: use `atomic.Pointer` to swap fitted models on re-train
- Parallel training (random forest trees, cross-validation folds) uses `errgroup` with `context.Context`

---

## 3. Numerical Foundation

### Primary: gonum

- `gonum.org/v1/gonum/mat` — dense matrix operations (float64)
- `gonum.org/v1/gonum/stat` — statistical functions, distributions
- `gonum.org/v1/gonum/optimize` — L-BFGS, gradient descent, Nelder-Mead
- `gonum.org/v1/gonum/graph` — graph algorithms (if needed for spectral methods)
- Pure Go by default; can link to OpenBLAS/MKL via `netlib` build tag for production performance

### float64 Only

All internal computation uses `float64`. This matches gonum's `mat.Dense` (hardcoded to float64) and avoids fighting the ecosystem with generics. Conversion helpers at boundaries for float32 input data.

### Sparse Matrices

Evaluate `github.com/james-bowman/sparse` for CSR/CSC/COO/DOK support with gonum `mat.Matrix` compatibility. If insufficient, implement a minimal sparse package internally. Critical for:
- Text/NLP feature matrices (TF-IDF)
- Large-scale linear models (sparse SGD)
- Sparse datasets

### Go 1.26 Advantages

- **Green Tea GC** (default) — 10-40% lower GC overhead for ML workloads
- **Experimental SIMD** (`simd/archsimd`) — potential for hand-optimized distance calculations, dot products
- **30% lower cgo overhead** — better OpenBLAS integration performance
- **Better stack allocation for slices** — reduces GC pressure for temporary buffers

---

## 4. Algorithm Coverage

Full parity with ferrolearn. Organized by package.

### 4.1 Supervised Learning

#### `linear/` — Linear Models

| Algorithm | Config Type | Fitted Type | Priority |
|-----------|-----------|-------------|----------|
| Ordinary Least Squares | `LinearRegressionConfig` | `LinearRegression` | P0 |
| Ridge Regression | `RidgeConfig` | `Ridge` | P0 |
| Lasso | `LassoConfig` | `Lasso` | P0 |
| Elastic Net | `ElasticNetConfig` | `ElasticNet` | P0 |
| Logistic Regression (L-BFGS) | `LogisticRegressionConfig` | `LogisticRegression` | P0 |
| Bayesian Ridge | `BayesianRidgeConfig` | `BayesianRidge` | P1 |
| Huber Regressor | `HuberRegressorConfig` | `HuberRegressor` | P1 |
| SGD Classifier | `SGDClassifierConfig` | `SGDClassifier` | P1 |
| SGD Regressor | `SGDRegressorConfig` | `SGDRegressor` | P1 |
| Linear Discriminant Analysis | `LDAClassifierConfig` | `LDAClassifier` | P1 |
| Isotonic Regression | `IsotonicRegressionConfig` | `IsotonicRegression` | P2 |
| RANSAC | `RANSACConfig` | `RANSAC` | P2 |
| Linear SVC | `LinearSVCConfig` | `LinearSVC` | P1 |
| Linear SVR | `LinearSVRConfig` | `LinearSVR` | P1 |

#### `tree/` — Decision Trees

| Algorithm | Config Type | Fitted Type | Priority |
|-----------|-----------|-------------|----------|
| Decision Tree Classifier | `DecisionTreeClassifierConfig` | `DecisionTreeClassifier` | P0 |
| Decision Tree Regressor | `DecisionTreeRegressorConfig` | `DecisionTreeRegressor` | P0 |

#### `ensemble/` — Ensemble Methods

| Algorithm | Config Type | Fitted Type | Priority |
|-----------|-----------|-------------|----------|
| Random Forest Classifier | `RandomForestClassifierConfig` | `RandomForestClassifier` | P0 |
| Random Forest Regressor | `RandomForestRegressorConfig` | `RandomForestRegressor` | P0 |
| Gradient Boosting Classifier | `GradientBoostingClassifierConfig` | `GradientBoostingClassifier` | P0 |
| Gradient Boosting Regressor | `GradientBoostingRegressorConfig` | `GradientBoostingRegressor` | P0 |
| Hist Gradient Boosting | `HistGradientBoostingConfig` | `HistGradientBoosting` | P1 |
| AdaBoost Classifier (SAMME/SAMME.R) | `AdaBoostClassifierConfig` | `AdaBoostClassifier` | P1 |

#### `neighbors/` — Nearest Neighbors

| Algorithm | Config Type | Fitted Type | Priority |
|-----------|-----------|-------------|----------|
| KNN Classifier | `KNeighborsClassifierConfig` | `KNeighborsClassifier` | P0 |
| KNN Regressor | `KNeighborsRegressorConfig` | `KNeighborsRegressor` | P0 |

KD-tree acceleration for dims <= 20, brute-force fallback for higher dimensions.

#### `naivebayes/` — Naive Bayes

| Algorithm | Config Type | Fitted Type | Priority |
|-----------|-----------|-------------|----------|
| Gaussian NB | `GaussianNBConfig` | `GaussianNB` | P0 |
| Multinomial NB | `MultinomialNBConfig` | `MultinomialNB` | P1 |
| Bernoulli NB | `BernoulliNBConfig` | `BernoulliNB` | P1 |
| Complement NB | `ComplementNBConfig` | `ComplementNB` | P2 |

### 4.2 Unsupervised Learning

#### `cluster/` — Clustering

| Algorithm | Config Type | Fitted Type | Priority |
|-----------|-----------|-------------|----------|
| KMeans | `KMeansConfig` | `KMeans` | P0 |
| Mini-Batch KMeans | `MiniBatchKMeansConfig` | `MiniBatchKMeans` | P1 |
| DBSCAN | `DBSCANConfig` | `DBSCAN` | P0 |
| Agglomerative Clustering | `AgglomerativeConfig` | `Agglomerative` | P1 |
| Gaussian Mixture (EM) | `GaussianMixtureConfig` | `GaussianMixture` | P1 |
| Mean Shift | `MeanShiftConfig` | `MeanShift` | P2 |
| Spectral Clustering | `SpectralClusteringConfig` | `SpectralClustering` | P2 |
| OPTICS | `OPTICSConfig` | `OPTICS` | P2 |
| Birch | `BirchConfig` | `Birch` | P2 |
| HDBSCAN | `HDBSCANConfig` | `HDBSCAN` | P2 |
| Label Propagation | `LabelPropagationConfig` | `LabelPropagation` | P2 |
| Label Spreading | `LabelSpreadingConfig` | `LabelSpreading` | P2 |

#### `decomposition/` — Dimensionality Reduction

| Algorithm | Config Type | Fitted Type | Priority |
|-----------|-----------|-------------|----------|
| PCA | `PCAConfig` | `PCA` | P0 |
| Incremental PCA | `IncrementalPCAConfig` | `IncrementalPCA` | P1 |
| Truncated SVD | `TruncatedSVDConfig` | `TruncatedSVD` | P1 |
| NMF | `NMFConfig` | `NMF` | P1 |
| Kernel PCA | `KernelPCAConfig` | `KernelPCA` | P2 |
| Factor Analysis | `FactorAnalysisConfig` | `FactorAnalysis` | P2 |
| Fast ICA | `FastICAConfig` | `FastICA` | P2 |
| Dictionary Learning | `DictionaryLearningConfig` | `DictionaryLearning` | P2 |
| LDA (Topic Modeling) | `LatentDirichletConfig` | `LatentDirichlet` | P2 |

#### `manifold/` — Manifold Learning

| Algorithm | Config Type | Fitted Type | Priority |
|-----------|-----------|-------------|----------|
| Isomap | `IsomapConfig` | `Isomap` | P2 |
| MDS | `MDSConfig` | `MDS` | P2 |
| Spectral Embedding | `SpectralEmbeddingConfig` | `SpectralEmbedding` | P2 |
| LLE | `LLEConfig` | `LLE` | P2 |
| t-SNE | `TSNEConfig` | `TSNE` | P1 |
| UMAP | `UMAPConfig` | `UMAP` | P1 |

### 4.3 Preprocessing

#### `preprocessing/` — Data Preprocessing

| Algorithm | Config Type | Fitted Type | Priority |
|-----------|-----------|-------------|----------|
| Standard Scaler | `StandardScalerConfig` | `StandardScaler` | P0 |
| MinMax Scaler | `MinMaxScalerConfig` | `MinMaxScaler` | P0 |
| Robust Scaler | `RobustScalerConfig` | `RobustScaler` | P1 |
| MaxAbs Scaler | `MaxAbsScalerConfig` | `MaxAbsScaler` | P1 |
| Normalizer | `NormalizerConfig` | `Normalizer` | P1 |
| One-Hot Encoder | `OneHotEncoderConfig` | `OneHotEncoder` | P0 |
| Label Encoder | `LabelEncoderConfig` | `LabelEncoder` | P0 |
| Ordinal Encoder | `OrdinalEncoderConfig` | `OrdinalEncoder` | P1 |
| Polynomial Features | `PolynomialFeaturesConfig` | `PolynomialFeatures` | P1 |
| Binarizer | `BinarizerConfig` | `Binarizer` | P1 |
| Power Transformer | `PowerTransformerConfig` | `PowerTransformer` | P2 |
| Function Transformer | `FunctionTransformerConfig` | `FunctionTransformer` | P1 |
| Column Transformer | `ColumnTransformerConfig` | `ColumnTransformer` | P1 |
| Simple Imputer | `SimpleImputerConfig` | `SimpleImputer` | P0 |
| Variance Threshold | `VarianceThresholdConfig` | `VarianceThreshold` | P1 |
| SelectKBest | `SelectKBestConfig` | `SelectKBest` | P1 |

### 4.4 Model Selection

#### `modelselection/` — Model Selection & Validation

| Feature | Type | Priority |
|---------|------|----------|
| KFold | `KFold` | P0 |
| Stratified KFold | `StratifiedKFold` | P0 |
| Time Series Split | `TimeSeriesSplit` | P1 |
| cross_val_score | `CrossValScore()` | P0 |
| Train/Test Split | `TrainTestSplit()` | P0 |
| Grid Search CV | `GridSearchCVConfig` / `GridSearchCV` | P0 |
| Randomized Search CV | `RandomizedSearchCVConfig` / `RandomizedSearchCV` | P1 |
| Halving Grid Search CV | `HalvingGridSearchCVConfig` / `HalvingGridSearchCV` | P2 |
| Calibrated Classifier CV | `CalibratedClassifierCVConfig` / `CalibratedClassifierCV` | P2 |
| Self-Training Classifier | `SelfTrainingConfig` / `SelfTraining` | P2 |

### 4.5 Metrics

#### `metrics/` — Evaluation Metrics

| Metric | Function | Priority |
|--------|----------|----------|
| Accuracy | `Accuracy(yTrue, yPred []float64) float64` | P0 |
| Precision | `Precision(yTrue, yPred []float64) float64` | P0 |
| Recall | `Recall(yTrue, yPred []float64) float64` | P0 |
| F1 Score | `F1(yTrue, yPred []float64) float64` | P0 |
| Confusion Matrix | `ConfusionMatrix(yTrue, yPred []float64) [][]int` | P0 |
| ROC AUC | `ROCAUC(yTrue, yScore []float64) float64` | P0 |
| Log Loss | `LogLoss(yTrue, yProb []float64) float64` | P1 |
| MAE | `MAE(yTrue, yPred []float64) float64` | P0 |
| MSE | `MSE(yTrue, yPred []float64) float64` | P0 |
| RMSE | `RMSE(yTrue, yPred []float64) float64` | P0 |
| R-squared | `R2(yTrue, yPred []float64) float64` | P0 |
| MAPE | `MAPE(yTrue, yPred []float64) float64` | P1 |
| Silhouette Score | `Silhouette(X *mat.Dense, labels []int) float64` | P1 |
| Adjusted Rand Score | `AdjustedRand(labelsTrue, labelsPred []int) float64` | P1 |
| Adjusted Mutual Info | `AdjustedMutualInfo(labelsTrue, labelsPred []int) float64` | P2 |
| Calinski-Harabasz | `CalinskiHarabasz(X *mat.Dense, labels []int) float64` | P2 |
| Davies-Bouldin | `DaviesBouldin(X *mat.Dense, labels []int) float64` | P2 |

### 4.6 Datasets

#### `datasets/` — Toy Datasets & Generators

| Feature | Function | Priority |
|---------|----------|----------|
| Iris | `LoadIris() (*mat.Dense, []float64, error)` | P0 |
| Diabetes | `LoadDiabetes() (*mat.Dense, []float64, error)` | P0 |
| Wine | `LoadWine() (*mat.Dense, []float64, error)` | P0 |
| make_blobs | `MakeBlobs(opts ...BlobsOption) (*mat.Dense, []float64, error)` | P0 |
| make_classification | `MakeClassification(opts ...ClassificationOption) (*mat.Dense, []float64, error)` | P0 |
| make_regression | `MakeRegression(opts ...RegressionOption) (*mat.Dense, []float64, error)` | P0 |
| make_moons | `MakeMoons(opts ...MoonsOption) (*mat.Dense, []float64, error)` | P1 |
| make_circles | `MakeCircles(opts ...CirclesOption) (*mat.Dense, []float64, error)` | P1 |
| make_sparse_uncorrelated | `MakeSparseUncorrelated(opts ...Option) (*mat.Dense, []float64, error)` | P2 |

### 4.7 Pipeline

#### `pipeline/` — Pipeline & Feature Union

```go
// Unfitted pipeline — has Fit(), no Predict()
pipe := pipeline.New(
    pipeline.TransformStep("scaler", preprocessing.NewStandardScaler()),
    pipeline.TransformStep("pca", decomposition.NewPCA(decomposition.WithNComponents(5))),
    pipeline.EstimatorStep("clf", linear.NewLogisticRegression()),
)

// Fitted pipeline — has Predict(), no Fit()
fitted, err := pipe.Fit(ctx, XTrain, yTrain)
preds, err := fitted.Predict(XTest)
```

| Feature | Priority |
|---------|----------|
| Pipeline (transform chain + estimator) | P0 |
| Feature Union (parallel transform + concat) | P1 |

### 4.8 Serialization

#### `io/` — Model Persistence

| Format | Priority | Notes |
|--------|----------|-------|
| JSON | P0 | Human-readable, debugging, interop |
| MessagePack | P0 | Compact binary, fast |
| gob | P0 | Go-native, fastest for Go-to-Go |
| ONNX export | P1 | Interop with other ML runtimes; via protobuf codegen from official schema |
| PMML export | Future | XML-based; stretch goal |

All serialized models include integrity checksums (CRC32).

### 4.9 Sparse Matrices

#### `sparse/` — Sparse Matrix Support

| Format | Priority |
|--------|----------|
| CSR (Compressed Sparse Row) | P0 |
| CSC (Compressed Sparse Column) | P1 |
| COO (Coordinate) | P1 |
| DOK (Dictionary of Keys) | P2 |

Must implement `gonum/mat.Matrix` interface for interop with dense algorithms.

**Evaluation of `james-bowman/sparse`:** The library provides CSR/CSC/COO/DOK/DIA formats with full `gonum/mat.Matrix` compatibility and assembly-optimized BLAS kernels. MIT licensed. However, it lacks sparse SVD, sparse eigendecomposition, iterative solvers, and scalar multiplication — critical gaps for ML. **Decision:** Use as the storage and basic arithmetic foundation. Build iterative methods (Lanczos/Arnoldi) on top of its `MulVecTo` primitive for sparse SVD/eigen. Consider `cpmech/gosl` as a cgo-optional complement for heavy sparse numerical work.

### 4.10 Kernels

#### `kernel/` — Kernel Functions

| Kernel | Priority |
|--------|----------|
| Linear | P1 |
| RBF (Gaussian) | P1 |
| Polynomial | P1 |
| Sigmoid | P2 |

### 4.11 Introspection

Fitted models expose learned parameters via exported fields and optional interfaces:

```go
type HasCoefficients interface {
    GetCoefficients() []float64
}

type HasFeatureImportances interface {
    GetFeatureImportances() []float64
}

type HasClasses interface {
    GetClasses() []float64
}
```

---

## 5. Package Structure

Single Go module (`github.com/freeformz/glearn`) with multiple packages:

```
glearn/
  go.mod
  glearn.go              # Core interfaces (Estimator, Predictor, Transformer, etc.)
  errors.go              # Sentinel errors
  linear/                # Linear models
  tree/                  # Decision trees
  ensemble/              # Random forests, gradient boosting, AdaBoost
  neighbors/             # KNN with KD-tree
  naivebayes/            # Naive Bayes classifiers
  cluster/               # Clustering algorithms
  decomposition/         # PCA, SVD, NMF, etc.
  manifold/              # t-SNE, UMAP, Isomap, etc.
  preprocessing/         # Scalers, encoders, imputers
  metrics/               # Evaluation metrics
  modelselection/        # Cross-validation, grid search, train/test split
  pipeline/              # Pipeline, FeatureUnion
  datasets/              # Toy datasets, synthetic generators
  io/                    # Serialization (JSON, MessagePack, gob, ONNX)
  sparse/                # Sparse matrix formats
  kernel/                # Kernel functions
  internal/              # Shared utilities (not public API)
    mathutil/            # Numerical helpers
    validate/            # Input validation
```

### Dependency Graph Principles

- Core interfaces live in the root `glearn` package (no dependencies on sub-packages)
- Algorithm packages depend on root + `internal/` only
- `pipeline/` depends on root interfaces only (not concrete algorithm packages)
- `modelselection/` depends on root interfaces + `metrics/`
- `io/` depends on root interfaces only
- No circular imports

---

## 6. Testing Strategy

### Unit Tests

Every algorithm has unit tests covering:
- Basic fit/predict correctness
- Edge cases (empty input, single sample, single feature)
- Hyperparameter variations
- Compile-time verification: `var _ glearn.Estimator = SomeConfig{}`

### Oracle Tests (scikit-learn Reference)

A Python script (`scripts/generate_fixtures.py`) generates reference fixtures from scikit-learn 1.7.x:
- Input data, fitted parameters (coefficients, centroids, etc.), predictions
- Stored as JSON in `testdata/` directories (Go convention)
- Go tests load fixtures and compare within relative tolerance
- Target: oracle tests for every P0 algorithm

### Property-Based Tests

Use `testing/quick` or `gopter` for invariant checks:
- Predicted probabilities sum to 1
- PCA components are orthogonal
- Scaler inverse_transform(transform(X)) == X
- Train accuracy >= test accuracy (on non-trivial data)

### Compile-Fail Tests

Verify that unfitted config types cannot call `Predict()`:
- Use `go vet` / `go build` on test fixtures that should fail to compile
- Stored in `testdata/` as `.go` files with expected errors

### Benchmarks

`testing.B` benchmarks from day one:
- Fit and Predict separately, with realistic data sizes
- Compare pure-Go vs OpenBLAS-linked performance
- Track regressions across versions

### Target

- 90%+ test coverage for P0 algorithms
- Oracle tests for all P0 algorithms
- Benchmarks for all P0 algorithms

---

## 7. Randomness & Reproducibility

All algorithms that use randomness accept a seed or `*rand.Rand` (from `math/rand/v2`):

```go
cfg := ensemble.NewRandomForestClassifier(
    ensemble.WithNTrees(100),
    ensemble.WithSeed(42),
)
```

- Never use the global rand source — tests must be deterministic
- Document that results are reproducible for a given seed + Go version + GOARCH

---

## 8. Error Handling

- All fallible operations return `error` as the last return value
- Sentinel errors for common cases: `glearn.ErrDimensionMismatch`, `glearn.ErrSingularMatrix`, `glearn.ErrConvergence`, `glearn.ErrEmptyInput`
- Errors wrap with context: `fmt.Errorf("glearn/linear: ridge fit failed: %w", err)`
- Configuration validation happens at `Fit()` time, not construction time (functional options don't return errors)
- Panic only for true programmer errors (e.g., nil receiver on internal types), never for data-dependent failures

---

## 9. Implementation Phases

### Phase 1: Foundation (P0)

Core infrastructure and most essential algorithms.

- Root package: interfaces, errors
- `internal/mathutil`, `internal/validate`
- `linear/`: LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
- `tree/`: DecisionTreeClassifier, DecisionTreeRegressor
- `ensemble/`: RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
- `neighbors/`: KNeighborsClassifier, KNeighborsRegressor
- `naivebayes/`: GaussianNB
- `cluster/`: KMeans, DBSCAN
- `decomposition/`: PCA
- `preprocessing/`: StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, SimpleImputer
- `metrics/`: Accuracy, Precision, Recall, F1, ConfusionMatrix, ROCAUC, MAE, MSE, RMSE, R2
- `modelselection/`: TrainTestSplit, KFold, StratifiedKFold, CrossValScore, GridSearchCV
- `pipeline/`: Pipeline
- `datasets/`: LoadIris, LoadDiabetes, LoadWine, MakeBlobs, MakeClassification, MakeRegression
- `io/`: JSON, MessagePack, gob serialization with CRC32
- `sparse/`: CSR (evaluate james-bowman/sparse)
- Oracle tests + benchmarks for all Phase 1 algorithms

### Phase 2: Breadth (P1)

Expand algorithm coverage and infrastructure.

- `linear/`: BayesianRidge, HuberRegressor, SGDClassifier, SGDRegressor, LDA, LinearSVC, LinearSVR
- `ensemble/`: HistGradientBoosting, AdaBoostClassifier
- `naivebayes/`: MultinomialNB, BernoulliNB
- `cluster/`: MiniBatchKMeans, AgglomerativeClustering, GaussianMixture
- `decomposition/`: IncrementalPCA, TruncatedSVD, NMF
- `manifold/`: t-SNE, UMAP
- `preprocessing/`: RobustScaler, MaxAbsScaler, Normalizer, OrdinalEncoder, PolynomialFeatures, Binarizer, FunctionTransformer, ColumnTransformer, VarianceThreshold, SelectKBest
- `metrics/`: LogLoss, MAPE, Silhouette, AdjustedRand
- `modelselection/`: TimeSeriesSplit, RandomizedSearchCV
- `pipeline/`: FeatureUnion
- `io/`: ONNX export (protobuf codegen from official schema)
- `sparse/`: CSC, COO
- `kernel/`: Linear, RBF, Polynomial

### Phase 3: Completeness (P2)

Full ferrolearn parity.

- `linear/`: IsotonicRegression, RANSAC
- `naivebayes/`: ComplementNB
- `cluster/`: MeanShift, SpectralClustering, OPTICS, Birch, HDBSCAN, LabelPropagation, LabelSpreading
- `decomposition/`: KernelPCA, FactorAnalysis, FastICA, DictionaryLearning, LatentDirichlet (topic modeling)
- `manifold/`: Isomap, MDS, SpectralEmbedding, LLE
- `preprocessing/`: PowerTransformer
- `metrics/`: AdjustedMutualInfo, CalinskiHarabasz, DaviesBouldin
- `modelselection/`: HalvingGridSearchCV, CalibratedClassifierCV, SelfTraining
- `sparse/`: DOK
- `kernel/`: Sigmoid

### Future

- PMML export
- Experimental SIMD optimizations for critical paths (distance computations, dot products)
- Online learning / streaming (`PartialFit` interface)
- Model explanation (feature importance, SHAP-like)

---

## 10. Versioning

- Stay at **v0.x.y** until API stabilizes — Go's module system is unforgiving post-v1
- Semantic versioning: breaking API changes bump minor version in v0.x
- Tag releases per phase milestone: `v0.1.0` (Phase 1 complete), `v0.2.0` (Phase 2), etc.

---

## 11. Dependencies

### Required

| Dependency | Purpose |
|------------|---------|
| `gonum.org/v1/gonum` | Linear algebra, statistics, optimization |
| `github.com/vmihailenco/msgpack/v5` | MessagePack serialization |

### Sparse Matrix Support

| Dependency | Purpose | Notes |
|------------|---------|-------|
| `github.com/james-bowman/sparse` | Sparse storage formats (CSR/CSC/COO/DOK), basic arithmetic | MIT, gonum compatible. Use as foundation. |

### Evaluate

| Dependency | Purpose | Decision Criteria |
|------------|---------|-------------------|
| `github.com/cpmech/gosl` | Sparse eigensolvers, UMFPACK, MUMPS (via cgo) | Performance vs. cgo build complexity tradeoff |

### Build-Tag Optional

| Dependency | Purpose | Build Tag |
|------------|---------|-----------|
| OpenBLAS / MKL (via gonum) | Accelerated BLAS/LAPACK | `netlib` |

### Test-Only

| Dependency | Purpose |
|------------|---------|
| `github.com/stretchr/testify` | Test assertions (evaluate) |

### Minimize Dependencies

- Prefer stdlib where possible
- No dependencies for core algorithm packages beyond gonum
- Serialization dependencies only in `io/` package

---

## 12. Documentation

- **godoc** for all public types, functions, interfaces
- **Testable examples** (`func ExampleLinearRegressionConfig_Fit()`) for every P0 algorithm
- **README.md** with quick-start guide, installation, feature matrix
- **CHANGELOG.md** tracking releases
- **CONTRIBUTING.md** with development setup, testing instructions, oracle test generation

---

## 13. Performance Targets

Not absolute numbers, but relative:
- Within 2-5x of scikit-learn (NumPy/C) for BLAS-heavy algorithms when linked to OpenBLAS
- Within 10x of scikit-learn for pure-Go builds
- Parallel algorithms (random forest, cross-validation) should achieve near-linear speedup up to GOMAXPROCS
- Memory usage comparable to equivalent gonum dense operations

---

## 14. Open Design Questions

1. **Sparse matrix strategy**: Adopt `james-bowman/sparse`, fork it, or build from scratch? (Pending evaluation)
2. **Partial fit / online learning**: How does `PartialFit` work with the builder pattern? Options:
   - `PartialFit` on config returns fitted type (first call), subsequent `PartialFit` on fitted type returns new fitted type
   - Separate `OnlineLearner` interface with different semantics
3. **Column/feature names**: Should matrices carry metadata (feature names, dtypes) or stay pure numerical?
4. **DataFrame-like abstraction**: Should glearn include a lightweight DataFrame, or leave that to external packages and work only with `mat.Dense`?
5. **SIMD utilization**: When/whether to use the experimental `simd/archsimd` package for distance calculations and other hot paths
