# Chapter 17: Nonparametric Smoothing - Mathematical Explanations

## Overview
Nonparametric smoothing methods estimate relationships between variables without assuming specific functional forms. This chapter covers kernel smoothing, spline methods, local regression, and modern approaches for flexible function estimation.

## 17.1 Introduction to Nonparametric Regression

### The Regression Problem
Given data (X₁, Y₁), ..., (Xₙ, Yₙ), estimate the regression function:
```
m(x) = E[Y|X = x]
```

without assuming a specific parametric form.

### Parametric vs Nonparametric
**Parametric:** m(x) = β₀ + β₁x + β₂x² (finite parameters)
**Nonparametric:** m(x) ∈ ℱ (infinite-dimensional function space)

### Bias-Variance Trade-off
**Flexible models:** Low bias, high variance
**Rigid models:** High bias, low variance
**Goal:** Find optimal balance

### Local vs Global Methods
**Local:** Use nearby points to estimate m(x)
**Global:** Use all data to estimate m(x)

## 17.2 Kernel Smoothing

### Nadaraya-Watson Estimator
```
m̂(x) = ∑ᵢ₌₁ⁿ Yᵢ K((x - Xᵢ)/h) / ∑ᵢ₌₁ⁿ K((x - Xᵢ)/h)
```

where:
- K(·) is kernel function
- h > 0 is bandwidth

**Weighted average:** Points closer to x get higher weights.

### Kernel Functions
**Properties:**
1. ∫ K(u) du = 1 (integrates to 1)
2. K(u) ≥ 0 (non-negative)
3. K(u) = K(-u) (symmetric)
4. ∫ u K(u) du = 0 (zero mean)

**Common kernels:**

**Gaussian:** K(u) = (1/√(2π))e^(-u²/2)
**Epanechnikov:** K(u) = (3/4)(1-u²)I(|u| ≤ 1)
**Uniform:** K(u) = (1/2)I(|u| ≤ 1)
**Triangular:** K(u) = (1-|u|)I(|u| ≤ 1)

### Bandwidth Selection
**Key parameter:** Controls smoothness
- Small h: Undersmoothing (high variance)
- Large h: Oversmoothing (high bias)

**Cross-validation:**
```
CV(h) = (1/n) ∑ᵢ₌₁ⁿ (Yᵢ - m̂₋ᵢ(Xᵢ))²
```

where m̂₋ᵢ(Xᵢ) is estimate without observation i.

**Plug-in methods:** Estimate unknown quantities in optimal bandwidth formula.

### Asymptotic Properties
**Bias:**
```
E[m̂(x)] - m(x) ≈ (h²/2)m''(x) ∫ u²K(u)du
```

**Variance:**
```
Var(m̂(x)) ≈ σ²(x)/(nhf(x)) ∫ K²(u)du
```

**Mean Squared Error:**
```
MSE(x) ≈ (h⁴/4)(m''(x))²(∫u²K(u)du)² + σ²(x)/(nhf(x))∫K²(u)du
```

**Optimal bandwidth:**
```
h_opt ∝ n^(-1/5)
```

## 17.3 Local Polynomial Regression

### Local Linear Regression
At each point x, fit weighted linear regression:
```
min ∑ᵢ₌₁ⁿ (Yᵢ - α - β(Xᵢ - x))² K((Xᵢ - x)/h)
```

**Solution:**
```
[α̂] = (XᵀWX)⁻¹XᵀWY
[β̂]
```

where W = diag(K((X₁-x)/h), ..., K((Xₙ-x)/h))

**Estimate:** m̂(x) = α̂

### Local Polynomial of Degree p
```
min ∑ᵢ₌₁ⁿ (Yᵢ - ∑ⱼ₌₀ᵖ βⱼ(Xᵢ - x)ʲ)² K((Xᵢ - x)/h)
```

**Common choices:**
- p = 0: Nadaraya-Watson (local constant)
- p = 1: Local linear
- p = 3: Local cubic

### Advantages of Local Linear
1. **Bias reduction:** Better bias properties than kernel smoothing
2. **Boundary effects:** Automatic adjustment at boundaries
3. **Asymmetric designs:** Handles unequal spacing well

### LOESS (LOcally WEighted Scatterplot Smoothing)
**Algorithm:**
1. For each x, find k nearest neighbors
2. Fit weighted polynomial (usually linear or quadratic)
3. Use tricube weights: w = (1 - |u|³)³ for |u| ≤ 1
4. Robust version: Downweight outliers iteratively

## 17.4 Spline Methods

### Piecewise Polynomials
**Definition:** Function that is polynomial on each interval between knots.

**Knots:** ξ₁ < ξ₂ < ... < ξₖ partition domain

**Piecewise linear:** Different slopes on each interval

### Splines
**s(x) is spline of degree p** if:
1. s(x) is polynomial of degree p on each interval
2. s(x) has p-1 continuous derivatives

**Linear splines:** Continuous piecewise linear functions
**Cubic splines:** Continuous with continuous first and second derivatives

### B-splines
**Basis functions:** {B₁(x), ..., Bₘ(x)}
```
s(x) = ∑ⱼ₌₁ᵐ cⱼBⱼ(x)
```

**Properties:**
- Local support: Bⱼ(x) = 0 outside small interval
- Partition of unity: ∑ⱼ Bⱼ(x) = 1
- Non-negative: Bⱼ(x) ≥ 0

### Regression Splines
**Model:** Y = s(X) + ε where s is spline

**Estimation:** Minimize
```
∑ᵢ₌₁ⁿ (Yᵢ - s(Xᵢ))²
```

**Linear in parameters:**
```
s(x) = ∑ⱼ₌₁ᵐ βⱼBⱼ(x)
```

Reduces to linear regression with basis functions.

### Smoothing Splines
**Penalized least squares:**
```
min_g ∑ᵢ₌₁ⁿ (Yᵢ - g(Xᵢ))² + λ ∫ (g''(x))² dx
```

**Solution:** Natural cubic spline with knots at data points.

**Smoothing parameter λ:** Controls smoothness
- λ = 0: Interpolating spline
- λ → ∞: Linear function

### Generalized Cross-Validation
**GCV criterion:**
```
GCV(λ) = n·RSS(λ) / (n - tr(S_λ))²
```

where S_λ is smoother matrix and tr(S_λ) is effective degrees of freedom.

## 17.5 Regression Trees

### Binary Trees
**Recursive partitioning:**
1. Split data based on predictor values
2. Fit constant in each region
3. Continue until stopping criterion met

**Prediction:** m̂(x) = average of Y in leaf containing x

### Tree Growing
**Greedy algorithm:**
1. Consider all possible splits on all variables
2. Choose split minimizing RSS:
   ```
   RSS = ∑_{i:xᵢ∈R₁} (yᵢ - ȳ₁)² + ∑_{i:xᵢ∈R₂} (yᵢ - ȳ₂)²
   ```
3. Repeat for each new region

### Tree Pruning
**Problem:** Large trees overfit

**Cost-complexity pruning:**
```
C_α(T) = ∑_{leaves} ∑_{i∈leaf} (yᵢ - ȳ_leaf)² + α|T|
```

where |T| is number of leaves.

**Cross-validation:** Choose α minimizing CV error.

### Advantages and Disadvantages
**Advantages:**
- Easy to interpret
- Handle interactions automatically
- Robust to outliers
- Handle mixed variable types

**Disadvantages:**
- High variance
- Biased toward variables with many levels
- Rectangular partitions only

## 17.6 Multivariate Smoothing

### Curse of Dimensionality
**Problem:** Performance degrades rapidly as dimension increases

**Local neighborhoods become sparse:**
- Volume of hypersphere: r^d
- Need exponentially more data

### Tensor Product Smoothing
**Separable model:**
```
m(x₁, x₂) = ∑ᵢ ∑ⱼ θᵢⱼ Bᵢ(x₁)Bⱼ(x₂)
```

**Assumption:** Smooth in each coordinate direction.

### Additive Models
```
m(x₁, ..., x_d) = α + ∑ⱼ₌₁ᵈ fⱼ(xⱼ)
```

**Backfitting algorithm:**
1. Initialize: f̂ⱼ(xⱼ) = 0 for all j
2. For j = 1, ..., d:
   - Compute partial residuals
   - Smooth residuals against xⱼ
   - Update f̂ⱼ
3. Repeat until convergence

### Generalized Additive Models (GAMs)
**Framework:** Link function connects additive predictor to response:
```
g(E[Y]) = α + ∑ⱼ₌₁ᵈ fⱼ(xⱼ)
```

**Examples:**
- Linear: g(μ) = μ
- Logistic: g(μ) = log(μ/(1-μ))
- Poisson: g(μ) = log(μ)

## 17.7 Wavelets

### Wavelet Basis
**Scaling function φ and wavelet ψ:**
```
f(x) = ∑ₖ αₖφₖ(x) + ∑ⱼ ∑ₖ βⱼₖψⱼₖ(x)
```

**Multi-resolution analysis:** Different levels of detail.

### Wavelet Regression
**Model:** Y = f(X) + ε where f has wavelet expansion

**Estimation:**
1. Transform data to wavelet domain
2. Threshold wavelet coefficients
3. Inverse transform to get estimate

### Thresholding
**Hard thresholding:**
```
δ_λ(x) = x · I(|x| > λ)
```

**Soft thresholding:**
```
δ_λ(x) = sign(x)(|x| - λ)₊
```

**Advantages:**
- Adaptive to local smoothness
- Efficient computation
- Good for functions with jumps

## 17.8 Modern Smoothing Methods

### Reproducing Kernel Hilbert Spaces
**RKHS framework:** Function space with inner product structure

**Representer theorem:** Solution has form:
```
f(x) = ∑ᵢ₌₁ⁿ αᵢ K(x, xᵢ)
```

### Gaussian Process Regression
**Prior:** f ~ GP(μ, K) where K is covariance function

**Posterior prediction:**
```
f(x)|data ~ N(μ*, σ²*)
```

**Mean and variance computed from kernel matrix.**

### Support Vector Regression
**ε-insensitive loss:**
```
L_ε(y, f(x)) = max(0, |y - f(x)| - ε)
```

**Optimization:**
```
min (1/2)||f||² + C ∑ᵢ L_ε(yᵢ, f(xᵢ))
```

### Neural Networks
**Universal approximation:** Can approximate any continuous function

**Single hidden layer:**
```
f(x) = ∑ⱼ₌₁ᵐ βⱼσ(αⱼᵀx + γⱼ)
```

where σ is activation function.

## 17.9 Bandwidth and Parameter Selection

### Cross-Validation Methods
**Leave-one-out CV:**
```
CV = (1/n) ∑ᵢ₌₁ⁿ (Yᵢ - m̂₋ᵢ(Xᵢ))²
```

**k-fold CV:** More computationally efficient

**Generalized CV:** Approximation using leverage values

### Information Criteria
**AIC for smoothing:**
```
AIC = n log(RSS/n) + 2·df
```

where df is effective degrees of freedom.

**BIC:** Replace 2 with log(n)

### Plug-in Methods
**Estimate unknown quantities in optimal bandwidth:**
- Pilot bandwidth to estimate derivatives
- Iterative refinement

**Rule-of-thumb:** Simple formulas based on normal reference

### Bootstrap Methods
**Bootstrap bandwidth selection:**
1. Resample data
2. Compute optimal bandwidth for each sample
3. Average across bootstrap samples

## 17.10 Confidence Bands

### Pointwise Confidence Intervals
**For kernel estimator:**
```
m̂(x) ± z_{α/2} · ŝe(m̂(x))
```

where ŝe(m̂(x)) = σ̂√(∫K²(u)du/(nhf̂(x)))

### Simultaneous Confidence Bands
**Goal:** P(m(x) ∈ [L(x), U(x)] for all x) = 1-α

**Methods:**
1. **Bonferroni:** Conservative but simple
2. **Bootstrap:** Resample to get critical values
3. **Gaussian process:** Exploit correlation structure

### Bootstrap Confidence Bands
**Algorithm:**
1. Generate bootstrap samples
2. Compute m̂*(x) - m̂(x) for each sample
3. Find critical values for simultaneous coverage

## 17.11 Density Estimation

### Kernel Density Estimation
```
f̂(x) = (1/nh) ∑ᵢ₌₁ⁿ K((x - Xᵢ)/h)
```

**Same bandwidth selection issues as regression.**

### Adaptive Methods
**Variable bandwidth:**
```
f̂(x) = (1/n) ∑ᵢ₌₁ⁿ (1/hᵢ) K((x - Xᵢ)/hᵢ)
```

where hᵢ depends on local density.

### Multivariate Density Estimation
**Product kernels:**
```
K(x) = ∏ⱼ₌₁ᵈ K₁((x - Xⱼ)/hⱼ)
```

**Curse of dimensionality:** Need hⱼ ∝ n^(-1/(4+d))

## 17.12 Computational Aspects

### Efficient Algorithms
**Binning:** Approximate kernel sums using discrete grid
**Fast Fourier Transform:** For convolution-type computations
**K-d trees:** Efficient nearest neighbor searches

### Large Datasets
**Subsampling:** Use subset for bandwidth selection
**Local fitting:** Fit only in neighborhood of query points
**Parallel processing:** Distribute computations

### Software
**R packages:** stats, KernSmooth, mgcv, np
**Python:** scipy, statsmodels, scikit-learn

## 17.13 Diagnostic Methods

### Residual Analysis
**Residuals:** êᵢ = Yᵢ - m̂(Xᵢ)

**Plots:**
- Residuals vs fitted values
- Residuals vs predictors
- Q-Q plots for normality

### Lack-of-Fit Tests
**Compare nonparametric vs parametric fits:**
- F-test for nested linear models
- Bootstrap tests for general alternatives

### Model Checking
**Visual assessment:**
- Compare parametric and nonparametric fits
- Examine confidence bands
- Check residual patterns

## Key Insights

1. **Flexibility vs Interpretability:** Nonparametric methods more flexible but less interpretable.

2. **Bandwidth crucial:** Controls bias-variance trade-off in smoothing methods.

3. **Local vs Global:** Different methods make different locality assumptions.

4. **Curse of Dimensionality:** Performance degrades in high dimensions without structure.

5. **Model Selection:** Cross-validation provides practical approach to tuning parameter selection.

## Common Pitfalls

1. **Oversmoothing:** Missing important features due to large bandwidth
2. **Undersmoothing:** Fitting noise due to small bandwidth  
3. **Boundary effects:** Poor performance at domain boundaries
4. **Extrapolation:** Dangerous to predict outside observed range
5. **High dimensions:** Naive application fails due to sparsity

## Practical Guidelines

### Method Selection
1. **Start simple:** Try linear/parametric model first
2. **Check assumptions:** Residual plots reveal nonlinearity
3. **Consider interpretability:** Trade-off with flexibility
4. **Validate thoroughly:** Cross-validation essential
5. **Visualize results:** Plots reveal model behavior

### Implementation
1. **Bandwidth selection:** Use cross-validation or GCV
2. **Check sensitivity:** Try different bandwidths
3. **Examine residuals:** Standard diagnostic plots
4. **Confidence bands:** Quantify uncertainty
5. **Compare methods:** Try multiple approaches

## Connections to Other Chapters

### To Chapter 14-15 (Linear Regression)
- Extension to nonlinear relationships
- Similar diagnostic procedures  
- Parametric vs nonparametric trade-offs

### To Chapter 8 (CDF Estimation)
- Kernel methods for density estimation
- Bandwidth selection strategies
- Nonparametric approach to inference

### To Chapter 23 (Classification)
- Nonparametric classification methods
- Local methods for complex boundaries
- Kernel methods in machine learning

This chapter provides comprehensive coverage of nonparametric smoothing methods, essential for flexible function estimation when parametric assumptions are questionable.