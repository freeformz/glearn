// Package linear provides linear model algorithms for regression and classification.
//
// Algorithms: LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression,
// BayesianRidge, HuberRegressor, SGDClassifier, SGDRegressor, LDA,
// IsotonicRegression, RANSAC, LinearSVC, LinearSVR.
//
// Each algorithm follows the config/fitted pattern:
//
//	cfg := linear.NewLinearRegression(linear.WithFitIntercept(true))
//	model, err := cfg.Fit(ctx, X, y) // returns *LinearRegression
//	preds, err := model.Predict(X)
package linear
