# Chapter 9: The Bootstrap - Mathematical Explanations

## Overview
The bootstrap is a powerful nonparametric method for estimating the distribution of statistics and constructing confidence intervals. It provides a way to assess uncertainty without making strong distributional assumptions.

## 9.1 Introduction to Bootstrap

### The Bootstrap Principle
The bootstrap is based on the **plug-in principle**: if we don't know the true distribution F, we use the empirical distribution F̂ₙ as an estimate.

**Key Idea:** The relationship between the sample and the population mirrors the relationship between bootstrap samples and the original sample.

### Notation
- True distribution: F
- Sample: X₁, ..., Xₙ ~ F
- Empirical distribution: F̂ₙ
- Statistic of interest: T = t(X₁, ..., Xₙ)
- Bootstrap samples: X₁*, ..., Xₙ* ~ F̂ₙ
- Bootstrap statistic: T* = t(X₁*, ..., Xₙ*)

## 9.2 The Empirical Distribution

### Definition
The **empirical distribution function** is:
```
F̂ₙ(x) = (1/n) ∑ᵢ₌₁ⁿ I(Xᵢ ≤ x)
```

where I(·) is the indicator function.

**Properties:**
- F̂ₙ is a step function with jumps of size 1/n at each data point
- F̂₍ₙ(x) → F(x) as n → ∞ (Glivenko-Cantelli theorem)
- √n(F̂ₙ(x) - F(x)) converges to a Gaussian process (Donsker's theorem)

### Sampling from F̂ₙ
Sampling from the empirical distribution F̂ₙ is equivalent to:
- Sampling uniformly from {X₁, ..., Xₙ} with replacement
- Each Xᵢ has probability 1/n of being selected

## 9.3 The Bootstrap Algorithm

### Nonparametric Bootstrap
1. **Draw bootstrap sample:** Sample X₁*, ..., Xₙ* with replacement from {X₁, ..., Xₙ}
2. **Compute bootstrap statistic:** T* = t(X₁*, ..., Xₙ*)
3. **Repeat:** Perform steps 1-2 B times to get T₁*, ..., T_B*
4. **Estimate distribution:** Use the empirical distribution of {T₁*, ..., T_B*} to approximate the distribution of T

### Mathematical Framework
The bootstrap approximates:
```
P_F(T ≤ x) ≈ P_F̂ₙ(T* ≤ x)
```

We estimate the right-hand side by:
```
P̂(T* ≤ x) = (1/B) ∑ᵦ₌₁ᴮ I(Tᵦ* ≤ x)
```

## 9.4 Bootstrap Bias and Variance

### Bias Estimation
The bootstrap estimate of bias is:
```
bias_boot(T̂) = E_F̂ₙ[T*] - T̂ = T̄* - T̂
```

where T̄* = (1/B) ∑ᵦ₌₁ᴮ Tᵦ* is the mean of bootstrap statistics.

**Bias-corrected estimator:**
```
T̂_bc = T̂ - bias_boot(T̂) = 2T̂ - T̄*
```

### Variance Estimation
The bootstrap estimate of variance is:
```
V̂ar_boot(T̂) = (1/(B-1)) ∑ᵦ₌₁ᴮ (Tᵦ* - T̄*)²
```

**Standard Error:** ŝe_boot(T̂) = √V̂ar_boot(T̂)

## 9.5 Bootstrap Confidence Intervals

### 1. Normal Approximation Method
Assumes T̂ is approximately normal:
```
T̂ ± z_{α/2} · ŝe_boot(T̂)
```

where z_{α/2} is the (1-α/2) quantile of the standard normal distribution.

### 2. Percentile Method
Use the empirical quantiles of the bootstrap distribution:
```
CI = [T*_{(α/2)}, T*_{(1-α/2)}]
```

where T*_{(p)} is the p-th quantile of {T₁*, ..., T_B*}.

**Advantages:**
- Transformation invariant
- Automatically accounts for skewness

### 3. Pivotal (Basic) Method
Based on the pivotal quantity T̂ - T:
```
CI = [2T̂ - T*_{(1-α/2)}, 2T̂ - T*_{(α/2)}]
```

**Rationale:** If G(x) = P(T̂ - T ≤ x), then:
```
P(T̂ - G⁻¹(1-α/2) ≤ T ≤ T̂ - G⁻¹(α/2)) = 1-α
```

### 4. Bias-Corrected and Accelerated (BCₐ) Method
More sophisticated method that corrects for bias and skewness:
```
CI = [T*_{(Φ(ẑ₀ + (ẑ₀+z_{α/2})/(1-â(ẑ₀+z_{α/2}))))}, T*_{(Φ(ẑ₀ + (ẑ₀+z_{1-α/2})/(1-â(ẑ₀+z_{1-α/2}))))}]
```

Where:
- ẑ₀ is the bias-correction constant
- â is the acceleration constant
- Φ is the standard normal CDF

**Bias-correction constant:**
```
ẑ₀ = Φ⁻¹(#{Tᵦ* < T̂}/B)
```

**Acceleration constant:**
```
â = skew(T̂)/6
```

## 9.6 Theoretical Properties

### Consistency
Under regularity conditions, the bootstrap is consistent:
```
sup_x |P_F̂ₙ(T* ≤ x) - P_F(T ≤ x)| → 0
```

as n → ∞.

### Rate of Convergence
The bootstrap typically provides:
- **First-order accuracy:** Error of order O(n⁻¹/²)
- **Second-order accuracy:** Error of order O(n⁻¹) for some statistics

### Bootstrap Failure
The bootstrap can fail when:
1. **Extreme order statistics:** Bootstrap of maximum/minimum
2. **Heavy-tailed distributions:** Infinite variance cases
3. **Boundary problems:** Parameters near boundaries
4. **Non-smooth statistics:** Non-differentiable functionals

## 9.7 Parametric Bootstrap

### Method
When the distribution family is known:
1. **Estimate parameters:** θ̂ = estimate from original sample
2. **Generate bootstrap sample:** X₁*, ..., Xₙ* ~ F_θ̂
3. **Compute bootstrap statistic:** T* = t(X₁*, ..., Xₙ*)
4. **Repeat:** B times

### Comparison with Nonparametric Bootstrap
**Advantages:**
- More efficient when model is correct
- Can handle smaller sample sizes
- Smoother bootstrap distribution

**Disadvantages:**
- Requires correct model specification
- Model misspecification can lead to poor coverage

## 9.8 Jackknife

### Definition
The **jackknife** is a precursor to the bootstrap:
- Leave-one-out samples: {X₁, ..., Xᵢ₋₁, Xᵢ₊₁, ..., Xₙ}
- Jackknife statistics: T₍ᵢ₎ = t(X₁, ..., Xᵢ₋₁, Xᵢ₊₁, ..., Xₙ)

### Jackknife Estimators
**Bias estimation:**
```
bias_jack = (n-1)(T̄₍·₎ - T̂)
```

where T̄₍·₎ = (1/n) ∑ᵢ₌₁ⁿ T₍ᵢ₎.

**Variance estimation:**
```
V̂ar_jack = ((n-1)/n) ∑ᵢ₌₁ⁿ (T₍ᵢ₎ - T̄₍·₎)²
```

### Relationship to Bootstrap
- Jackknife is less computationally intensive
- Bootstrap generally more accurate
- Jackknife variance estimate often used in practice

## 9.9 Smooth Bootstrap

### Motivation
The empirical distribution F̂ₙ is discrete, but true F might be continuous.

### Kernel Smoothing
Replace F̂ₙ with a smooth estimate:
```
F̃ₙ(x) = (1/n) ∑ᵢ₌₁ⁿ K((x - Xᵢ)/h)
```

where K is a kernel function and h is the bandwidth.

### Smooth Bootstrap Algorithm
1. **Draw bootstrap sample:** X₁*, ..., Xₙ* from F̃ₙ
2. **Equivalent method:** Xᵢ* = Xⱼ + hεᵢ where Xⱼ is sampled uniformly from data and ε ~ K

## 9.10 Bootstrap for Regression

### Residual Bootstrap
For regression model Y = m(X) + ε:
1. **Fit model:** Ŷᵢ = m̂(Xᵢ), residuals êᵢ = Yᵢ - Ŷᵢ
2. **Resample residuals:** e₁*, ..., eₙ* from {ê₁, ..., êₙ}
3. **Generate bootstrap response:** Yᵢ* = m̂(Xᵢ) + eᵢ*
4. **Refit model:** Use (Xᵢ, Yᵢ*) to get new estimates

### Paired Bootstrap
Resample (Xᵢ, Yᵢ) pairs directly:
1. **Resample pairs:** (X₁*, Y₁*), ..., (Xₙ*, Yₙ*) from {(X₁, Y₁), ..., (Xₙ, Yₙ)}
2. **Refit model:** Use bootstrap sample

### Wild Bootstrap
For heteroscedastic errors:
1. **Generate bootstrap errors:** eᵢ* = êᵢ · vᵢ where vᵢ are auxiliary random variables
2. **Common choice:** vᵢ takes values {(1-√5)/2, (1+√5)/2} with appropriate probabilities

## 9.11 Computational Considerations

### Choice of B (Number of Bootstrap Samples)
- **Standard errors:** B = 200-1000 usually sufficient
- **Confidence intervals:** B = 1000-2000 recommended
- **Hypothesis tests:** B = 2000+ for accurate p-values

### Computational Complexity
- Basic bootstrap: O(nB) for each bootstrap iteration
- Efficient implementations use vectorization
- Parallel computing naturally applicable

### Bootstrap Diagnostics
1. **Check convergence:** Monitor statistics as B increases
2. **Examine bootstrap distribution:** Look for outliers or multimodality
3. **Compare different methods:** Normal, percentile, pivotal intervals

## 9.12 Applications and Examples

### Example 1: Bootstrap for Sample Mean
For X̄ with unknown distribution:
```python
# Bootstrap algorithm
def bootstrap_mean(data, B=1000):
    n = len(data)
    bootstrap_means = []
    for b in range(B):
        bootstrap_sample = np.random.choice(data, n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    return np.array(bootstrap_means)
```

### Example 2: Bootstrap for Correlation
Correlation coefficient has complex distribution; bootstrap provides practical solution:
```python
def bootstrap_correlation(x, y, B=1000):
    n = len(x)
    bootstrap_corrs = []
    for b in range(B):
        indices = np.random.choice(n, n, replace=True)
        x_boot, y_boot = x[indices], y[indices]
        bootstrap_corrs.append(np.corrcoef(x_boot, y_boot)[0,1])
    return np.array(bootstrap_corrs)
```

### Example 3: Bootstrap for Complex Statistics
For statistics without known distributions (e.g., skewness, kurtosis):
```python
def skewness(x):
    n = len(x)
    mean_x = np.mean(x)
    var_x = np.var(x, ddof=0)
    return np.mean(((x - mean_x) / np.sqrt(var_x))**3)

def bootstrap_skewness(data, B=1000):
    bootstrap_stats = []
    for b in range(B):
        bootstrap_sample = np.random.choice(data, len(data), replace=True)
        bootstrap_stats.append(skewness(bootstrap_sample))
    return np.array(bootstrap_stats)
```

## 9.13 Bootstrap vs. Other Methods

### Bootstrap vs. Delta Method
- **Delta method:** Asymptotic, requires derivatives
- **Bootstrap:** Finite sample, automatic

### Bootstrap vs. Permutation Tests
- **Bootstrap:** Estimates sampling distribution
- **Permutation:** Tests null hypothesis of exchangeability

### Bootstrap vs. Cross-Validation
- **Bootstrap:** Estimates variability/uncertainty
- **Cross-validation:** Estimates prediction error

## 9.14 Limitations and Cautions

### When Bootstrap Fails
1. **Estimating extreme quantiles:** Bootstrap of maximum/minimum inconsistent
2. **Infinite variance:** Heavy-tailed distributions without finite second moment
3. **Boundary effects:** Parameters constrained to intervals
4. **Time series:** Dependent data requires modified bootstrap methods

### Practical Issues
1. **Computational cost:** Can be expensive for complex models
2. **Software implementation:** Need to ensure proper random sampling
3. **Interpretation:** Bootstrap intervals are not always exactly what they seem

## 9.15 Modern Extensions

### Block Bootstrap
For time series data:
- **Moving blocks:** Resample overlapping blocks
- **Circular blocks:** Handle boundary effects

### Bayesian Bootstrap
- Weight observations by Dirichlet(1,...,1) random variables
- Provides Bayesian interpretation

### m-out-of-n Bootstrap
- Resample m < n observations
- Better for some boundary problems

## Key Insights

1. **Plug-in Principle:** The bootstrap operationalizes the plug-in principle for distribution estimation.

2. **Resampling Revolution:** Provides solutions when analytical methods are intractable.

3. **Automatic Inference:** Requires minimal distributional assumptions.

4. **Computational Statistics:** Exemplifies how computational power enables new statistical methods.

## Common Pitfalls

1. **Assuming the bootstrap always works:** It can fail in specific situations
2. **Using too few bootstrap samples:** Leads to Monte Carlo error
3. **Misunderstanding confidence interval interpretation:** Bootstrap CIs are not always exact
4. **Ignoring dependence structure:** Standard bootstrap assumes independence

This chapter introduces one of the most important computational tools in modern statistics, providing a bridge between theoretical concepts and practical data analysis.