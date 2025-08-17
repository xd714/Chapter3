# Chapter 25 - High-Dimensional Statistics

*Note: This chapter extends beyond the original 24 chapters of "All of Statistics" to cover modern high-dimensional statistical methods.*

## 25.1 Introduction

Modern data analysis frequently involves **high-dimensional** settings where the number of parameters $p$ is large, potentially much larger than the sample size $n$. Traditional statistical methods often fail in these settings due to:

- **Curse of dimensionality**: Statistical procedures that work well in low dimensions break down as dimension increases
- **Overfitting**: With $p > n$, classical methods can fit the data perfectly but generalize poorly
- **Computational challenges**: Standard algorithms become intractable

Examples of high-dimensional data include:
- **Genomics**: Expression levels of thousands of genes
- **Text analysis**: Document-term matrices with vast vocabularies
- **Image analysis**: Pixel intensities in high-resolution images
- **Finance**: Returns of many assets in portfolio optimization

This chapter covers fundamental concepts and methods for high-dimensional statistics.

## 25.2 The High-Dimensional Setting

### 25.2.1 Notation and Setup

We consider the standard setup where we observe $n$ samples and $p$ features:

$$\mathbf{X} = \begin{pmatrix} 
X_{11} & X_{12} & \cdots & X_{1p} \\
X_{21} & X_{22} & \cdots & X_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
X_{n1} & X_{n2} & \cdots & X_{np}
\end{pmatrix}, \quad \mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}$$

The **high-dimensional regime** typically refers to cases where $p$ is large relative to $n$, including:
- **Fixed $p$, large $n$**: Classical asymptotics
- **$p$ fixed, $n \to \infty$**: Traditional statistics
- **$p \to \infty, n \to \infty$ with $p/n \to c$**: Modern high-dimensional asymptotics
- **$p \gg n$**: Ultra-high-dimensional setting

### 25.2.2 Challenges in High Dimensions

**Theorem 25.1** (Curse of Dimensionality). In high dimensions:
1. **Distance concentration**: All pairwise distances become approximately equal
2. **Volume concentration**: Most volume of a sphere concentrates near its surface
3. **Sample coverage**: To maintain the same density, sample size must grow exponentially with dimension

**Example 25.1** (Distance Concentration). For $\mathbf{X}_1, \mathbf{X}_2 \stackrel{\text{iid}}{\sim} N(0, I_p)$:

$$\frac{\|\mathbf{X}_1 - \mathbf{X}_2\|^2}{p} \to 4 \quad \text{in probability as } p \to \infty$$

This means all points look equidistant in high dimensions!

## 25.3 Regularization and Penalized Methods

### 25.3.1 Ridge Regression

When $p > n$, ordinary least squares is not well-defined. **Ridge regression** adds a penalty term:

$$\hat{\boldsymbol{\beta}}_{\text{ridge}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|^2 \right\}$$

**Solution**: $\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda I)^{-1}\mathbf{X}^T\mathbf{y}$

The penalty parameter $\lambda \geq 0$ controls the trade-off between fit and complexity.

**Properties of Ridge Regression**:
- Always has a unique solution even when $p > n$
- Shrinks coefficients toward zero
- Doesn't perform variable selection (all coefficients non-zero)

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

def ridge_regression_path(X, y, lambdas):
    """Compute ridge regression for different lambda values"""
    coefficients = []
    
    for lam in lambdas:
        ridge = Ridge(alpha=lam)
        ridge.fit(X, y)
        coefficients.append(ridge.coef_)
    
    return np.array(coefficients)

# Example: Ridge path
n, p = 50, 100
X = np.random.normal(0, 1, (n, p))
beta_true = np.zeros(p)
beta_true[:5] = [2, -1.5, 1, -0.5, 0.8]  # Only first 5 coefficients non-zero
y = X @ beta_true + 0.1 * np.random.normal(0, 1, n)

lambdas = np.logspace(-3, 2, 50)
ridge_coefs = ridge_regression_path(X, y, lambdas)
```

### 25.3.2 Lasso Regression

The **Lasso** (Least Absolute Shrinkage and Selection Operator) uses an $\ell_1$ penalty:

$$\hat{\boldsymbol{\beta}}_{\text{lasso}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_1 \right\}$$

where $\|\boldsymbol{\beta}\|_1 = \sum_{j=1}^p |\beta_j|$.

**Key Properties**:
- Performs automatic variable selection (sets some coefficients exactly to zero)
- Solution depends on $\lambda$ in a piecewise-linear fashion
- No closed-form solution; requires optimization algorithms

**Theorem 25.2** (Lasso Sparsity). As $\lambda$ increases, the Lasso solution becomes sparser, eventually setting all coefficients to zero.

```python
from sklearn.linear_model import Lasso, LassoCV

def lasso_path_demo(X, y):
    """Demonstrate Lasso regularization path"""
    # Use cross-validation to select lambda
    lasso_cv = LassoCV(cv=5, random_state=42)
    lasso_cv.fit(X, y)
    
    print(f"Optimal lambda: {lasso_cv.alpha_:.4f}")
    print(f"Number of selected features: {np.sum(lasso_cv.coef_ != 0)}")
    
    return lasso_cv.coef_, lasso_cv.alpha_

# Example
lasso_coefs, optimal_lambda = lasso_path_demo(X, y)
```

### 25.3.3 Elastic Net

The **Elastic Net** combines Ridge and Lasso penalties:

$$\hat{\boldsymbol{\beta}}_{\text{elastic}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2 \|\boldsymbol{\beta}\|^2 \right\}$$

**Advantages**:
- Combines variable selection (Lasso) with coefficient shrinkage (Ridge)
- Better handles correlated predictors than Lasso alone
- Can select more than $n$ variables

## 25.4 Sparsity and Variable Selection

### 25.4.1 Sparse Models

A model is **sparse** if only a small subset of the $p$ predictors have non-zero coefficients. This is formalized as:

$$\|\boldsymbol{\beta}\|_0 = \sum_{j=1}^p I(\beta_j \neq 0) \ll p$$

where $\|\cdot\|_0$ is the $\ell_0$ "norm" (actually a pseudo-norm).

**Motivation for Sparsity**:
- **Interpretability**: Easier to understand which variables matter
- **Prediction**: Reduces overfitting by focusing on important variables
- **Computational efficiency**: Faster predictions with fewer variables

### 25.4.2 Best Subset Selection

The **best subset selection** problem is:

$$\hat{\boldsymbol{\beta}}_{\text{best}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_0 \right\}$$

This is **NP-hard** in general, requiring evaluation of $\binom{p}{k}$ subsets of size $k$.

**Modern Algorithms**:
- **Branch and bound**: Intelligently search the space of subsets
- **Forward/backward stepwise**: Greedy algorithms
- **Mixed integer optimization**: Formulate as integer programming problem

### 25.4.3 Information Criteria

**Information criteria** balance fit and model complexity:

**Akaike Information Criterion (AIC)**:
$$\text{AIC} = -2\log L(\hat{\boldsymbol{\beta}}) + 2k$$

**Bayesian Information Criterion (BIC)**:
$$\text{BIC} = -2\log L(\hat{\boldsymbol{\beta}}) + k \log n$$

where $k$ is the number of parameters and $L(\hat{\boldsymbol{\beta}})$ is the likelihood.

BIC tends to select sparser models than AIC as $n$ grows.

## 25.5 Random Matrix Theory

### 25.5.1 Eigenvalue Distributions

**Random Matrix Theory** studies the behavior of eigenvalues of random matrices, crucial for understanding high-dimensional covariance matrices.

**Theorem 25.3** (Marchenko-Pastur Law). Let $\mathbf{S} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$ where $\mathbf{X}$ is $n \times p$ with iid entries having mean 0 and variance 1. As $n, p \to \infty$ with $p/n \to \gamma \in (0, \infty)$, the empirical distribution of eigenvalues of $\mathbf{S}$ converges to the Marchenko-Pastur distribution with density:

$$f(x) = \frac{\sqrt{(b-x)(x-a)}}{2\pi \gamma x} \quad \text{for } x \in [a, b]$$

where $a = (1-\sqrt{\gamma})^2$ and $b = (1+\sqrt{\gamma})^2$.

**Implications**:
- When $p/n$ is large, many eigenvalues are close to zero
- Standard PCA may not work well in high dimensions
- Need for regularized covariance estimation

### 25.5.2 Regularized Covariance Estimation

**Sample Covariance**: $\hat{\boldsymbol{\Sigma}} = \frac{1}{n-1}\sum_{i=1}^n (\mathbf{X}_i - \overline{\mathbf{X}})(\mathbf{X}_i - \overline{\mathbf{X}})^T$

**Problems in High Dimensions**:
- Singular when $p > n$
- Many eigenvalues close to zero even when $p < n$
- Poor condition number

**Ledoit-Wolf Shrinkage**:
$$\hat{\boldsymbol{\Sigma}}_{\text{LW}} = (1-\rho)\hat{\boldsymbol{\Sigma}} + \rho \mu I$$

where $\rho$ is the shrinkage intensity and $\mu = \text{tr}(\hat{\boldsymbol{\Sigma}})/p$.

```python
from sklearn.covariance import LedoitWolf

def regularized_covariance_demo(X):
    """Compare sample and regularized covariance estimation"""
    n, p = X.shape
    
    # Sample covariance
    sample_cov = np.cov(X.T)
    
    # Ledoit-Wolf shrinkage
    lw = LedoitWolf()
    lw_cov = lw.fit(X).covariance_
    
    # Compare condition numbers
    sample_cond = np.linalg.cond(sample_cov)
    lw_cond = np.linalg.cond(lw_cov)
    
    print(f"Sample covariance condition number: {sample_cond:.2e}")
    print(f"Ledoit-Wolf condition number: {lw_cond:.2e}")
    print(f"Shrinkage intensity: {lw.shrinkage_:.3f}")
    
    return sample_cov, lw_cov

# Example
X_highdim = np.random.multivariate_normal(np.zeros(100), np.eye(100), 50)
sample_cov, lw_cov = regularized_covariance_demo(X_highdim)
```

## 25.6 Multiple Testing

### 25.6.1 The Multiple Testing Problem

When testing $m$ hypotheses simultaneously, the probability of at least one Type I error increases dramatically:

$$\mathbb{P}(\text{at least one Type I error}) = 1 - (1-\alpha)^m \approx m\alpha$$

for small $\alpha$.

**Example**: Testing $m = 1000$ hypotheses at level $\alpha = 0.05$ gives:
$$\mathbb{P}(\text{at least one false positive}) \approx 1 - 0.95^{1000} \approx 1$$

### 25.6.2 Family-Wise Error Rate (FWER)

**FWER** = Probability of making at least one Type I error.

**Bonferroni Correction**: Test each hypothesis at level $\alpha/m$ to control FWER at level $\alpha$.

**Holm's Method** (Step-down procedure):
1. Order p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. Find $k = \min\{i : p_{(i)} > \alpha/(m+1-i)\}$
3. Reject hypotheses $1, \ldots, k-1$

### 25.6.3 False Discovery Rate (FDR)

**False Discovery Rate** = Expected proportion of false discoveries among all discoveries.

$$\text{FDR} = \mathbb{E}\left[\frac{V}{V + S}\right]$$

where $V$ = number of false discoveries, $S$ = number of true discoveries.

**Benjamini-Hochberg Procedure**:
1. Order p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. Find $k = \max\{i : p_{(i)} \leq \frac{i}{m}\alpha\}$
3. Reject hypotheses $1, \ldots, k$

```python
def benjamini_hochberg(pvalues, alpha=0.05):
    """Benjamini-Hochberg procedure for FDR control"""
    m = len(pvalues)
    sorted_indices = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_indices]
    
    # Find largest k such that p_(k) <= k/m * alpha
    thresholds = np.arange(1, m+1) / m * alpha
    significant = sorted_pvals <= thresholds
    
    if np.any(significant):
        k = np.where(significant)[0][-1]
        rejected = sorted_indices[:k+1]
    else:
        rejected = []
    
    return rejected

# Example
m = 1000
pvalues = np.random.uniform(0, 1, m)
pvalues[:50] = np.random.uniform(0, 0.01, 50)  # 50 true signals

rejected = benjamini_hochberg(pvalues, alpha=0.1)
print(f"Number of rejections: {len(rejected)}")
```

## 25.7 High-Dimensional Inference

### 25.7.1 Debiased/Desparsified Lasso

The Lasso estimate is biased due to the $\ell_1$ penalty. The **debiased Lasso** corrects this bias:

$$\hat{\boldsymbol{\beta}}^{\text{debiased}} = \hat{\boldsymbol{\beta}}^{\text{lasso}} + \frac{1}{n}\hat{\boldsymbol{\Theta}}\mathbf{X}^T(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}^{\text{lasso}})$$

where $\hat{\boldsymbol{\Theta}}$ approximates $(\mathbf{X}^T\mathbf{X}/n)^{-1}$.

**Theorem 25.4**. Under regularity conditions:
$$\sqrt{n}(\hat{\beta}_j^{\text{debiased}} - \beta_j) \leadsto N(0, \sigma^2 \Theta_{jj})$$

This enables hypothesis testing and confidence intervals.

### 25.7.2 Post-Selection Inference

After variable selection, naive inference ignores the selection step. **Post-selection inference** accounts for this.

**Selective Inference**: Condition on the selection event $\{S = \hat{S}\}$ where $\hat{S}$ is the selected set.

The conditional distribution can be computed exactly for some selection procedures.

## 25.8 Compressed Sensing

### 25.8.1 Sparse Recovery

**Compressed Sensing** studies recovery of sparse signals from few measurements:

Given $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$ where $\boldsymbol{\beta}$ is $s$-sparse, can we recover $\boldsymbol{\beta}$ when $n \ll p$?

**Restricted Isometry Property (RIP)**: A matrix $\mathbf{X}$ satisfies RIP of order $s$ with constant $\delta_s$ if:

$$(1-\delta_s)\|\boldsymbol{\beta}\|^2 \leq \|\mathbf{X}\boldsymbol{\beta}\|^2 \leq (1+\delta_s)\|\boldsymbol{\beta}\|^2$$

for all $s$-sparse vectors $\boldsymbol{\beta}$.

**Theorem 25.5**. If $\mathbf{X}$ satisfies RIP with $\delta_{2s} < \sqrt{2} - 1$, then the Lasso exactly recovers $s$-sparse signals with high probability.

### 25.8.2 Recovery Guarantees

**Theorem 25.6** (Compressed Sensing). For Gaussian random matrices $\mathbf{X}$, exact recovery via $\ell_1$ minimization succeeds with high probability if:

$$n \geq C s \log(p/s)$$

for some constant $C$.

This shows that we need far fewer measurements than variables!

## 25.9 Machine Learning Connections

### 25.9.1 Bias-Variance Decomposition in High Dimensions

For prediction error:
$$\mathbb{E}[(\hat{f}(\mathbf{x}) - f(\mathbf{x}))^2] = \text{Bias}^2[\hat{f}(\mathbf{x})] + \text{Var}[\hat{f}(\mathbf{x})] + \sigma^2$$

In high dimensions:
- **Variance** often dominates due to overfitting
- **Regularization** trades bias for variance reduction
- **Sparsity assumptions** can dramatically improve performance

### 25.9.2 Double Descent

Recent work shows that prediction error can exhibit **double descent**:
1. Classical bias-variance trade-off for $p < n$
2. Peak at $p = n$ (interpolation threshold)
3. Decreasing error for $p > n$ (overparameterized regime)

This challenges traditional statistical wisdom about overfitting.

## 25.10 Computational Considerations

### 25.10.1 Coordinate Descent

For Lasso and Elastic Net, **coordinate descent** is highly efficient:

**Algorithm 25.7** (Coordinate Descent for Lasso):
1. Initialize $\hat{\boldsymbol{\beta}}^{(0)}$
2. For $t = 1, 2, \ldots$ until convergence:
   - For $j = 1, \ldots, p$:
     $$\hat{\beta}_j^{(t)} = S\left(\frac{1}{n}\mathbf{X}_j^T(\mathbf{y} - \mathbf{X}_{-j}\hat{\boldsymbol{\beta}}_{-j}^{(t)})}{\|\mathbf{X}_j\|^2/n}, \frac{\lambda}{2\|\mathbf{X}_j\|^2/n}\right)$$

where $S(z, \gamma) = \text{sign}(z)(|z| - \gamma)_+$ is the soft-thresholding operator.

```python
def soft_threshold(z, gamma):
    """Soft thresholding operator"""
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)

def coordinate_descent_lasso(X, y, lam, max_iter=1000, tol=1e-6):
    """Coordinate descent for Lasso regression"""
    n, p = X.shape
    beta = np.zeros(p)
    
    # Precompute column norms
    col_norms_sq = np.sum(X**2, axis=0)
    
    for iteration in range(max_iter):
        beta_old = beta.copy()
        
        for j in range(p):
            # Compute residual without j-th variable
            r_j = y - X @ beta + X[:, j] * beta[j]
            
            # Update j-th coefficient
            beta[j] = soft_threshold(
                (X[:, j] @ r_j) / n,
                lam / (2 * col_norms_sq[j] / n)
            )
        
        # Check convergence
        if np.linalg.norm(beta - beta_old) < tol:
            break
    
    return beta

# Example
beta_cd = coordinate_descent_lasso(X, y, 0.1)
```

### 25.10.2 Screening Rules

**Screening rules** identify variables that will be zero in the optimal solution, allowing us to remove them before optimization:

**Basic Screening**: If $|X_j^T \mathbf{r}| < \lambda$ where $\mathbf{r}$ is the current residual, then $\hat{\beta}_j = 0$.

**Safe Screening**: Provides certificates that certain variables will be zero in the optimal solution.

## 25.11 Recent Developments

### 25.11.1 High-Dimensional Probability

**Concentration inequalities** are crucial for understanding high-dimensional phenomena:

**Theorem 25.8** (Sub-Gaussian Concentration). If $X$ is sub-Gaussian with parameter $\sigma$, then:
$$\mathbb{P}(|X - \mathbb{E}[X]| \geq t) \leq 2\exp(-t^2/(2\sigma^2))$$

**Matrix concentration** extends these ideas to random matrices.

### 25.11.2 Deep Learning Connections

Modern deep learning exhibits many high-dimensional phenomena:
- **Overparameterization**: More parameters than training samples
- **Implicit regularization**: SGD acts as implicit regularizer
- **Double descent**: Test error can decrease with more parameters

## 25.12 Exercises

**Exercise 25.1**. Show that in $p$ dimensions, the volume of a sphere concentrates in a thin shell near the surface. Specifically, show that for a unit sphere in $\mathbb{R}^p$, the fraction of volume in the outer shell of thickness $\epsilon$ approaches 1 as $p \to \infty$.

**Exercise 25.2**. Implement coordinate descent for the Elastic Net. Compare the convergence rate with the Lasso.

**Exercise 25.3**. Generate high-dimensional data where only the first 5 out of 100 variables are relevant. Compare Ridge, Lasso, and Elastic Net in terms of variable selection and prediction accuracy.

**Exercise 25.4**. Implement the Benjamini-Hochberg procedure and compare it with Bonferroni correction on simulated multiple testing data.

**Exercise 25.5**. Study the eigenvalue distribution of random covariance matrices. Generate data according to the Marchenko-Pastur setting and verify the theoretical predictions.

**Exercise 25.6**. Implement best subset selection using branch-and-bound for small problems ($p \leq 20$). Compare with Lasso solutions.

**Exercise 25.7**. Study the double descent phenomenon by fitting polynomial regression with varying degrees on a fixed dataset.

**Exercise 25.8**. Implement compressed sensing recovery using Lasso. Generate sparse signals and study the phase transition between successful and failed recovery.

## 25.13 Bibliographic Remarks

High-dimensional statistics is a rapidly evolving field. Key references include:

**Books**:
- **Hastie, Tibshirani, and Wainwright (2015)**: Statistical Learning with Sparsity
- **Wainwright (2019)**: High-Dimensional Statistics: A Non-Asymptotic Viewpoint
- **Bühlmann and van de Geer (2011)**: Statistics for High-Dimensional Data

**Foundational Papers**:
- **Tibshirani (1996)**: Regression shrinkage and selection via the Lasso
- **Candès and Tao (2006)**: Near-optimal signal recovery from random projections
- **Benjamini and Hochberg (1995)**: Controlling the false discovery rate

**Recent Developments**:
- **Zhang et al. (2014)**: Confidence intervals for low dimensional parameters in high dimensional linear models
- **Belkin et al. (2019)**: Reconciling modern machine-learning practice and the classical bias–variance trade-off

The field continues to evolve rapidly with connections to machine learning, optimization, and random matrix theory.
