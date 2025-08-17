# Chapter 23: Classification - Mathematical Explanations

## Overview
Classification is the problem of predicting a discrete variable Y from another random variable X. This chapter covers fundamental classification methods, their theoretical foundations, and practical applications in supervised learning.

## 23.1 Introduction to Classification

### Problem Setup
Consider IID data (X₁, Y₁), ..., (Xₙ, Yₙ) where:
- Xᵢ = (Xᵢ₁, ..., Xᵢₓ) ∈ 𝒳 ⊂ ℝᵈ is a d-dimensional feature vector
- Yᵢ ∈ 𝒴 = {1, 2, ..., k} is the class label

**Goal:** Find a classification rule h: 𝒳 → 𝒴 that predicts Y from X.

### Vocabulary Translation
| Statistics | Computer Science | Meaning |
|------------|------------------|---------|
| Classification | Supervised learning | Predicting discrete Y from X |
| Data | Training sample | (X₁, Y₁), ..., (Xₙ, Yₙ) |
| Covariates | Features | The Xᵢ's |
| Classifier | Hypothesis | Map h: 𝒳 → 𝒴 |
| Estimation | Learning | Finding good classifier |

## 23.2 Risk and the Bayes Classifier

### Risk Function
The **risk** (or misclassification error) of classifier h is:
```
R(h) = P(h(X) ≠ Y) = E[I(h(X) ≠ Y)]
```

### Bayes Classifier
The **Bayes classifier** minimizes the risk:
```
h*(x) = argmax P(Y = j|X = x)
```

**Bayes risk:** R* = R(h*) (minimum possible risk)

### Posterior Probabilities
Let ηⱼ(x) = P(Y = j|X = x) be the posterior probability.

**Properties:**
- ∑ⱼ ηⱼ(x) = 1
- ηⱼ(x) ≥ 0
- h*(x) = argmax ηⱼ(x)

### Binary Classification
For k = 2 classes:
```
h*(x) = {
  1  if η₁(x) > 1/2
  0  if η₁(x) < 1/2
}
```

**Decision boundary:** {x : η₁(x) = 1/2}

## 23.3 Linear and Quadratic Classifiers

### Linear Discriminant Analysis (LDA)

**Assumptions:**
- X|Y = j ~ N(μⱼ, Σ) (same covariance)
- P(Y = j) = πⱼ

**Discriminant functions:**
```
δⱼ(x) = log πⱼ - (1/2)μⱼᵀΣ⁻¹μⱼ + xᵀΣ⁻¹μⱼ
```

**Classification rule:** h(x) = argmax δⱼ(x)

**Decision boundary between classes i and j:**
```
{x : δᵢ(x) = δⱼ(x)}
```

This is linear in x, hence "linear" discriminant analysis.

### Sample-based LDA
Replace population parameters with sample estimates:
- μ̂ⱼ = (1/nⱼ) ∑ᵢ:Yᵢ=j Xᵢ
- Σ̂ = (1/(n-k)) ∑ⱼ ∑ᵢ:Yᵢ=j (Xᵢ - μ̂ⱼ)(Xᵢ - μ̂ⱼ)ᵀ
- π̂ⱼ = nⱼ/n

### Quadratic Discriminant Analysis (QDA)

**Assumption:** X|Y = j ~ N(μⱼ, Σⱼ) (different covariances)

**Discriminant functions:**
```
δⱼ(x) = log πⱼ - (1/2)log|Σⱼ| - (1/2)(x - μⱼ)ᵀΣⱼ⁻¹(x - μⱼ)
```

**Decision boundaries:** Quadratic in x

### Bias-Variance Trade-off
- **LDA:** Lower variance, higher bias (assumes equal covariances)
- **QDA:** Higher variance, lower bias (more flexible)

## 23.4 Logistic Regression

### Model
For binary classification:
```
log(P(Y = 1|X = x)/(P(Y = 0|X = x))) = β₀ + βᵀx
```

**Probability:**
```
P(Y = 1|X = x) = exp(β₀ + βᵀx)/(1 + exp(β₀ + βᵀx))
```

### Multinomial Logistic Regression
For k classes, use k-1 log-odds:
```
log(P(Y = j|X = x)/P(Y = k|X = x)) = β₀ⱼ + βⱼᵀx, j = 1, ..., k-1
```

**Probabilities:**
```
P(Y = j|X = x) = exp(β₀ⱼ + βⱼᵀx)/(1 + ∑ₗ₌₁ᵏ⁻¹ exp(β₀ₗ + βₗᵀx))
```

### Maximum Likelihood Estimation
**Log-likelihood:**
```
ℓ(β) = ∑ᵢ₌₁ⁿ ∑ⱼ₌₁ᵏ Yᵢⱼ log P(Y = j|Xᵢ, β)
```

where Yᵢⱼ = I(Yᵢ = j).

**No closed form:** Use iterative methods (Newton-Raphson, IRLS)

## 23.5 Classification Trees

### Tree Structure
- **Internal nodes:** Tests on features
- **Leaves:** Class predictions
- **Path:** Sequence of tests leading to prediction

### Growing Trees

**Recursive binary splitting:**
1. Find best split (feature + threshold)
2. Split data into two subsets
3. Repeat recursively on each subset

**Splitting criteria:**
- **Misclassification rate:** ∑ⱼ I(ĵ ≠ j)p̂ⱼ
- **Gini index:** ∑ⱼ p̂ⱼ(1 - p̂ⱼ) = 1 - ∑ⱼ p̂ⱼ²
- **Entropy:** -∑ⱼ p̂ⱼ log p̂ⱼ

where p̂ⱼ = proportion of class j in node.

### Pruning
Large trees overfit. Use **cost-complexity pruning:**

**Cost-complexity criterion:**
```
Cₐ(T) = ∑ₜ Nₜ Qₜ + α|T|
```

where:
- Nₜ = number of observations in leaf t
- Qₜ = misclassification rate in leaf t  
- |T| = number of leaves
- α = complexity parameter

**Procedure:**
1. Grow large tree T₀
2. For each α, find optimal subtree Tₐ
3. Use cross-validation to choose α

### Advantages and Disadvantages
**Advantages:**
- Easy to interpret
- Handles mixed data types
- Automatic feature selection
- Captures interactions

**Disadvantages:**
- High variance
- Biased toward features with more levels
- Difficulty capturing linear relationships

## 23.6 Nearest Neighbors

### k-Nearest Neighbors (k-NN)

**Algorithm:**
1. Find k nearest neighbors to x in training set
2. Classify x as majority class among k neighbors

**Distance metrics:**
- **Euclidean:** ||x - xᵢ||₂ = √∑ⱼ(xⱼ - xᵢⱼ)²
- **Manhattan:** ||x - xᵢ||₁ = ∑ⱼ|xⱼ - xᵢⱼ|
- **Minkowski:** ||x - xᵢ||ₚ = (∑ⱼ|xⱼ - xᵢⱼ|ᵖ)^(1/p)

### Theoretical Properties

**Consistency:** As n → ∞ and k → ∞ with k/n → 0:
```
R(ĥₖ) → R* (Bayes risk)
```

**Curse of dimensionality:** Performance degrades in high dimensions
- All points become equidistant
- Need exponentially more data

### Choosing k
- **Small k:** Low bias, high variance
- **Large k:** High bias, low variance
- **Typical choice:** k = √n

**Cross-validation:** Common method for selecting k

## 23.7 Support Vector Machines

### Linear SVM (Separable Case)

**Goal:** Find hyperplane that separates classes with maximum margin

### Linear SVM (Separable Case)

**Goal:** Find hyperplane that separates classes with maximum margin

**Hyperplane:** wᵀx + b = 0

**Margin:** Distance from hyperplane to nearest point = 1/||w||

**Optimization problem:**
```
minimize (1/2)||w||²
subject to yᵢ(wᵀxᵢ + b) ≥ 1, i = 1, ..., n
```

**Solution:** Only depends on **support vectors** (points on margin boundary)

### Linear SVM (Non-separable Case)

**Soft margin SVM:** Allow some misclassification
```
minimize (1/2)||w||² + C∑ᵢξᵢ
subject to yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

where ξᵢ are **slack variables** and C controls trade-off between margin and violations.

### Kernel SVM

**Key insight:** Transform data to higher-dimensional space where linear separation possible

**Kernel trick:** Instead of computing φ(x) explicitly, use kernel function:
```
K(x, x') = φ(x)ᵀφ(x')
```

**Decision function:**
```
f(x) = ∑ᵢ αᵢyᵢK(xᵢ, x) + b
```

### Common Kernels

**Linear:** K(x, x') = xᵀx'

**Polynomial:** K(x, x') = (γxᵀx' + r)ᵈ

**RBF (Gaussian):** K(x, x') = exp(-γ||x - x'||²)

**Sigmoid:** K(x, x') = tanh(γxᵀx' + r)

### Properties
- **Maximum margin:** Unique solution (if exists)
- **Sparse:** Only support vectors matter
- **Kernel trick:** Handles nonlinear boundaries
- **Regularization:** C parameter controls overfitting

## 23.8 Neural Networks

### Single Layer Perceptron
**Model:**
```
f(x) = σ(w₀ + ∑ⱼwⱼxⱼ)
```

where σ is activation function (e.g., sigmoid, tanh, ReLU).

**Limitation:** Can only learn linearly separable functions

### Multi-layer Perceptron (MLP)

**Architecture:**
- **Input layer:** Features x₁, ..., xₓ
- **Hidden layers:** Transformed features
- **Output layer:** Class probabilities

**Forward propagation:**
```
z₁ = W₁x + b₁
a₁ = σ(z₁)
⋮
zₗ = Wₗaₗ₋₁ + bₗ
ŷ = softmax(zₗ)
```

### Backpropagation
**Algorithm for computing gradients:**
1. Forward pass: Compute predictions
2. Backward pass: Compute gradients using chain rule
3. Update weights: wᵢⱼ ← wᵢⱼ - η ∂L/∂wᵢⱼ

**Gradient computation:**
```
∂L/∂wᵢⱼ = ∂L/∂zⱼ · ∂zⱼ/∂wᵢⱼ = δⱼ · aᵢ
```

### Universal Approximation
**Theorem:** MLP with one hidden layer can approximate any continuous function on compact set arbitrarily well (with enough hidden units).

**Practical limitation:** May need exponentially many hidden units

## 23.9 Ensemble Methods

### Bootstrap Aggregating (Bagging)

**Algorithm:**
1. Generate B bootstrap samples
2. Train classifier on each sample
3. Combine predictions by voting

**Variance reduction:** Averaging reduces variance without increasing bias

**Random Forests:** Bagging + random feature selection
- At each split, consider only subset of features
- Further reduces correlation between trees

### Boosting

**Key idea:** Sequentially build weak learners, focusing on previously misclassified examples

**AdaBoost algorithm:**
1. Initialize weights: wᵢ = 1/n
2. For m = 1, ..., M:
   - Train classifier hₘ on weighted data
   - Compute error: εₘ = ∑ᵢ wᵢI(hₘ(xᵢ) ≠ yᵢ)
   - Compute αₘ = (1/2)log((1-εₘ)/εₘ)
   - Update weights: wᵢ ← wᵢexp(αₘI(hₘ(xᵢ) ≠ yᵢ))
   - Normalize weights

**Final classifier:**
```
H(x) = sign(∑ₘ αₘhₘ(x))
```

### Gradient Boosting
**Functional gradient descent:** Fit new learner to residuals

**Algorithm:**
1. Initialize f₀(x) = argmin ∑ᵢ L(yᵢ, γ)
2. For m = 1, ..., M:
   - Compute residuals: rᵢ = -∂L(yᵢ, f(xᵢ))/∂f(xᵢ)
   - Fit hₘ(x) to residuals
   - Find γₘ = argmin ∑ᵢ L(yᵢ, fₘ₋₁(xᵢ) + γhₘ(xᵢ))
   - Update: fₘ(x) = fₘ₋₁(x) + γₘhₘ(x)

## 23.10 Model Assessment and Selection

### Training vs Test Error
**Training error:** Error on data used to fit model
**Test error:** Error on independent test data

**Key insight:** Training error underestimates true error (overfitting)

### Cross-Validation

**k-fold CV:**
1. Divide data into k folds
2. For each fold:
   - Train on k-1 folds
   - Test on remaining fold
3. Average test errors

**Leave-one-out CV:** Special case with k = n

### Model Selection Criteria

**AIC:** 2k - 2ln(L̂)
**BIC:** k ln(n) - 2ln(L̂)

where k = number of parameters, L̂ = maximum likelihood.

### Performance Metrics

**Confusion Matrix:**
|           | Predicted |          |
|-----------|-----------|----------|
| **Actual**| Positive  | Negative |
| Positive  | TP        | FN       |
| Negative  | FP        | TN       |

**Metrics:**
- **Accuracy:** (TP + TN)/(TP + TN + FP + FN)
- **Precision:** TP/(TP + FP)
- **Recall (Sensitivity):** TP/(TP + FN)
- **Specificity:** TN/(TN + FP)
- **F1-score:** 2 × (Precision × Recall)/(Precision + Recall)

### ROC Curves
**ROC curve:** Plot of True Positive Rate vs False Positive Rate

**AUC:** Area Under the ROC Curve
- AUC = 0.5: Random classifier
- AUC = 1.0: Perfect classifier

## 23.11 Statistical Learning Theory

### PAC Learning Framework
**Probably Approximately Correct (PAC) learning:**

A concept class is PAC-learnable if there exists algorithm that, for any ε > 0, δ > 0:
- Uses polynomial number of samples and computation time
- With probability ≥ 1-δ, outputs hypothesis with error ≤ ε

### VC Dimension

**Definition:** VC dimension of hypothesis class ℋ is largest set size that ℋ can shatter.

**Shattering:** ℋ shatters set S if for every labeling of S, some h ∈ ℋ achieves zero error.

**Examples:**
- Linear classifiers in ℝᵈ: VC dimension = d + 1
- k-NN: VC dimension = ∞

### Generalization Bounds

**VC inequality:** With probability 1-δ:
```
R(h) ≤ R̂(h) + √[(2/n)(VC(ℋ) + log(2/δ))]
```

where R̂(h) is empirical risk.

**Rademacher complexity:** Sharper bounds using data-dependent measures

### Structural Risk Minimization
Balance empirical risk and model complexity:
```
minimize R̂(h) + √[complexity penalty]
```

## 23.12 Feature Selection and Dimensionality Reduction

### Filter Methods
**Univariate selection:** Score each feature independently
- Chi-square test for categorical features
- Mutual information
- Correlation with target

### Wrapper Methods
**Forward selection:**
1. Start with empty set
2. Add feature that most improves performance
3. Stop when no improvement

**Backward elimination:** Start with all features, remove worst

### Embedded Methods
**L1 regularization (Lasso):** Automatically selects features
**Tree-based:** Features used in tree splits are selected

### Principal Component Analysis (PCA)
**Goal:** Find low-dimensional representation preserving variance

**Steps:**
1. Center data: X̃ = X - X̄
2. Compute covariance: C = X̃ᵀX̃/(n-1)
3. Find eigenvectors of C
4. Project onto top k eigenvectors

**Linear Discriminant Analysis (LDA):** Finds projection maximizing between-class separation

## 23.13 Imbalanced Data

### Problem
When classes have very different frequencies:
- Algorithms biased toward majority class
- Minority class often more important

### Solutions

**Resampling:**
- **Undersampling:** Remove majority class examples
- **Oversampling:** Duplicate minority class examples
- **SMOTE:** Generate synthetic minority examples

**Cost-sensitive learning:** Assign different costs to misclassification types

**Threshold adjustment:** Adjust decision threshold based on class frequencies

### Evaluation Metrics
Standard accuracy misleading for imbalanced data:
- Use precision, recall, F1-score
- Area under precision-recall curve
- Matthews correlation coefficient

## 23.14 Multi-class Classification

### One-vs-Rest (OvR)
Train k binary classifiers:
- Classifier j: class j vs all others
- Prediction: argmax fⱼ(x)

### One-vs-One (OvO)
Train k(k-1)/2 binary classifiers:
- One for each pair of classes
- Prediction: majority vote

### Direct Multi-class
Some algorithms naturally handle multiple classes:
- Multinomial logistic regression
- Decision trees
- Neural networks

## 23.15 Online Learning

### Setting
Data arrives sequentially: (x₁, y₁), (x₂, y₂), ...

**Goal:** Make prediction ŷₜ before seeing yₜ

### Perceptron Algorithm
**Update rule:** If mistake at time t:
```
wₜ₊₁ = wₜ + yₜxₜ
```

**Mistake bound:** Number of mistakes ≤ R²/γ²
where R = max ||xₜ||, γ = margin

### Gradient Descent
**Update:** wₜ₊₁ = wₜ - ηₜ∇L(yₜ, wₜᵀxₜ)

**Regret bound:** O(√T) for convex losses

## 23.16 Semi-supervised Learning

### Setting
Small amount of labeled data + large amount of unlabeled data

### Self-training
1. Train classifier on labeled data
2. Predict labels for unlabeled data
3. Add high-confidence predictions to training set
4. Repeat

### Co-training
**Requirements:** Features can be split into two views
1. Train two classifiers on different feature views
2. Each classifier labels examples for the other
3. Add high-confidence predictions to training set

### Graph-based Methods
**Assumption:** Nearby points likely to have same label
- Build graph connecting similar points
- Propagate labels through graph

## 23.17 Deep Learning for Classification

### Convolutional Neural Networks (CNNs)
**Architecture:**
- **Convolutional layers:** Local feature detection
- **Pooling layers:** Dimensionality reduction
- **Fully connected layers:** Final classification

**Advantages:** 
- Translation invariance
- Parameter sharing
- Hierarchical feature learning

### Transfer Learning
**Idea:** Use pre-trained network as feature extractor
1. Take network trained on large dataset (e.g., ImageNet)
2. Remove final layer
3. Add new classifier for target task
4. Fine-tune on target data

### Regularization Techniques
**Dropout:** Randomly set fraction of neurons to zero during training
**Batch normalization:** Normalize inputs to each layer
**Data augmentation:** Generate variations of training data

## 23.18 Practical Considerations

### Data Preprocessing
**Standardization:** Scale features to have mean 0, variance 1
**Normalization:** Scale to [0,1] range
**Categorical encoding:** One-hot, ordinal, target encoding

### Hyperparameter Tuning
**Grid search:** Try all combinations of parameter values
**Random search:** Sample parameter values randomly
**Bayesian optimization:** Use probabilistic models to guide search

### Model Interpretation
**Feature importance:** Which features matter most?
**Partial dependence plots:** How does each feature affect predictions?
**SHAP values:** Unified approach to feature attribution

### Computational Considerations
**Scalability:** How does algorithm scale with n, d?
**Memory requirements:** Can algorithm handle large datasets?
**Parallelization:** Can computation be distributed?

## 23.19 Modern Developments

### AutoML
**Goal:** Automate machine learning pipeline
- Feature engineering
- Algorithm selection
- Hyperparameter tuning

### Federated Learning
**Setting:** Training across multiple devices/organizations
- Data remains local
- Only model updates shared
- Privacy preservation

### Adversarial Examples
**Problem:** Small perturbations can fool classifiers
**Defense:** Adversarial training, certified defenses

## Key Insights

1. **No Free Lunch:** No single classifier dominates all problems
2. **Bias-Variance Trade-off:** Central theme in model selection
3. **Overfitting:** Major concern, especially with complex models
4. **Feature Engineering:** Often more important than algorithm choice
5. **Evaluation:** Proper assessment crucial for real-world deployment

## Common Pitfalls

1. **Data leakage:** Using future information to predict past
2. **Selection bias:** Non-representative training data
3. **Overfitting to validation set:** Repeated model tuning
4. **Ignoring class imbalance:** Misleading accuracy metrics
5. **Correlation vs causation:** Classification ≠ causal inference

## Comparison of Methods

| Method | Interpretability | Flexibility | Scalability | Assumptions |
|--------|------------------|-------------|-------------|-------------|
| Logistic Regression | High | Low | High | Linear boundaries |
| Decision Trees | High | Medium | Medium | None |
| Random Forest | Medium | High | High | None |
| SVM | Low | High | Medium | None |
| Neural Networks | Low | Very High | High | Large data |
| k-NN | Medium | High | Poor | Local smoothness |

## Conclusion

Classification is a rich field connecting statistics, computer science, and machine learning. The choice of method depends on:
- Data size and dimensionality
- Interpretability requirements
- Computational constraints
- Domain knowledge

Modern practice often involves ensemble methods combining multiple approaches for robust performance.