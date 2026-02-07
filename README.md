# ML Classification Methods
Implementation and comparison of supervised classification algorithms: Bayes-optimal classifiers, SVMs, and neural networks.
Projects

## 1. Bayes-Optimal Classification and ROC Analysis
Comparison of classifiers on 3D Gaussian mixture data.
The Bayes-optimal classifier sets the performance ceiling. Naive Bayes suffers from model mismatch. Fisher LDA underperforms due to its linear projection constraint.

## 2. MLP Training Dynamics
An investigation on how neural networks approach Bayes-optimal performance as training data increases.
It was discovered that the gap narrows with more data but never fully closes.

## 3. SVM vs MLP Comparison
Binary classification on concentric circles data using Support Vector Machines and Multi-Layer Perceptrons.

Both classifiers achieve similar performance. The SVM is slightly better and more stable during training. The MLP struggles with optimization as model complexity increases.

## Datasets
All experiments use synthetic Gaussian mixture data with known ground truth, allowing direct comparison against theoretical optimal performance.

## Future Work
Python implementations with scikit-learn
Additional activation function comparisons for MLP
