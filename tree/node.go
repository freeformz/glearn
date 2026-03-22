package tree

// TreeNode represents a single node in a decision tree.
// Internal nodes have Feature >= 0 and non-nil Left/Right children.
// Leaf nodes have Feature == -1 and a prediction Value.
type TreeNode struct {
	// Feature is the index of the feature used for splitting.
	// -1 indicates a leaf node.
	Feature int

	// Threshold is the split threshold. Samples with feature value <= Threshold
	// go to the Left child; samples with feature value > Threshold go to the Right child.
	Threshold float64

	// Left is the left child subtree (feature value <= Threshold).
	Left *TreeNode

	// Right is the right child subtree (feature value > Threshold).
	Right *TreeNode

	// Value is the prediction value at this node.
	// For classifiers, this is the majority class.
	// For regressors, this is the mean target value.
	Value float64

	// Proba holds class probabilities for classifier nodes.
	// Index i corresponds to the probability of Classes[i].
	// nil for regressor nodes.
	Proba []float64

	// NSamples is the number of training samples reaching this node.
	NSamples int
}

// IsLeaf returns true if this node is a leaf (has no children).
func (n *TreeNode) IsLeaf() bool {
	return n.Feature == -1
}

// predict traverses the tree to find the leaf node for a single sample
// and returns the prediction value.
func (n *TreeNode) predict(sample []float64) float64 {
	node := n
	for !node.IsLeaf() {
		if sample[node.Feature] <= node.Threshold {
			node = node.Left
		} else {
			node = node.Right
		}
	}
	return node.Value
}

// predictProba traverses the tree to find the leaf node for a single sample
// and returns the class probability distribution.
func (n *TreeNode) predictProba(sample []float64) []float64 {
	node := n
	for !node.IsLeaf() {
		if sample[node.Feature] <= node.Threshold {
			node = node.Left
		} else {
			node = node.Right
		}
	}
	return node.Proba
}
