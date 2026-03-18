// Package pipeline provides composable ML pipelines that chain transformers and estimators.
//
// A Pipeline follows the same config/fitted pattern as individual models:
// the unfitted Pipeline has Fit() but no Predict(), and the fitted
// FittedPipeline has Predict() but no Fit().
package pipeline
