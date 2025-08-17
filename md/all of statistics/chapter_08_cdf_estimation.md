# Chapter 8: Estimating the CDF and Statistical Functionals - Mathematical Explanations

## Overview
This chapter focuses on nonparametric methods for estimating cumulative distribution functions (CDFs) and functionals of distributions. These methods make minimal assumptions about the underlying distribution and form the foundation for robust statistical inference.

## 8.1 The Empirical Distribution Function

### Definition
Given observations X₁, X₂, ..., Xₙ from distribution F, the **empirical distribution function** is:
```
F̂ₙ(x) = (1/n) ∑ᵢ₌₁ⁿ I(Xᵢ ≤ x)
```

where I(·) is the indicator function.

### Properties

**Step function:** F̂ₙ(x) is a right-continuous step function with jumps of size 1/n at each observation.

**Proper CDF:** F̂ₙ satisfies all properties of a cumulative distribution function:
- Non-decreasing
- Right-continuous
- lim_{x→-∞} F̂ₙ(x) = 0
- lim_{x→∞} F̂ₙ(x) = 1

**Discrete approximation:** F̂ₙ places mass 1/n at each observed value.

### Basic Properties

**Unbiasedness:** E[F̂ₙ(x)] = F(x) for all x

**Variance:** Var(F̂ₙ(x)) = F(x)(1-F(x))/n

**For fixed x:** nF̂ₙ(x) ~ Binomial(n, F(x))

## 8.2 Theoretical Properties of F̂ₙ

### Pointwise Convergence
For any fixed x:
```
F̂ₙ(x) →ᵖ F(x)   (convergence in probability)
F̂ₙ(x) →ᵃ·ˢ· F(x)  (almost sure convergence)
```

**Proof:** Follows from strong law of large numbers applied to I(X ≤ x).

### Asymptotic Normality
For fixed x with 0 < F(x) < 1:
```
√n(F̂ₙ(x) - F(x)) ⇝ N(0, F(x)(1-F(x)))
```

**Proof:** Follows from central limit theorem.

### Uniform Convergence: Glivenko-Cantelli Theorem
```
sup_x |F̂ₙ(x) - F(x)| →ᵃ·ˢ· 0
```

**Interpretation:** The empirical CDF converges uniformly to the true CDF.

**Proof sketch:** Uses maximal inequalities and approximation arguments.

### Rate of Uniform Convergence: DKW Inequality
**Dvoretzky-Kiefer-Wolfowitz inequality:**
```
P(sup_x |F̂ₙ(x) - F(x)| > ε) ≤ 2e^{-2nε²}
```

**Massart's refinement:**
```
P(sup_x |F̂ₙ(x) - F(x)| > ε) ≤ 2e^{-2nε²}(1 + O(ε))
```

### Confidence Bands
The DKW inequality provides **simultaneous confidence bands**:
```
P(F(x) ∈ [F̂ₙ(x) - εₙ, F̂ₙ(x) + εₙ] ∀x) ≥ 1 - α
```

where εₙ = √(log(2/α)/(2n)).

## 8.3 Statistical Functionals

### Definition
A **statistical functional** T(F) is a real-valued function of the distribution F.

**Examples:**
- Mean: T(F) = ∫ x dF(x)
- Variance: T(F) = ∫ x² dF(x) - (∫ x dF(x))²
- Median: T(F) = F⁻¹(1/2)
- α-quantile: T(F) = F⁻¹(α)

### Plug-in Estimator
**Plug-in principle:** Estimate T(F) by T(F̂ₙ).

**Notation:** T̂ₙ = T(F̂ₙ)

### Examples of Plug-in Estimators

**Sample mean:**
```
μ̂ = T(F̂ₙ) = ∫ x dF̂ₙ(x) = (1/n) ∑ᵢ₌₁ⁿ Xᵢ
```

**Sample variance:**
```
σ̂² = ∫ x² dF̂ₙ(x) - (∫ x dF̂ₙ(x))² = (1/n) ∑ᵢ₌₁ⁿ Xᵢ² - X̄²
```

**Sample quantiles:**
```
q̂ₐ = F̂ₙ⁻¹(α) = X₍⌈nα⌉₎
```

where X₍ₖ₎ is the k-th order statistic.

## 8.4 Influence Functions and Robustness

### Influence Function
The **influence function** measures the effect of an infinitesimal contamination at point x:
```
IF(x; T, F) = lim_{ε→0} [T((1-ε)F + εδₓ) - T(F)]/ε
```

where δₓ is the point mass at x.

### Gross Error Sensitivity
```
γ* = sup_x |IF(x; T, F)|
```

**Interpretation:** Maximum effect of a single outlier.

### Breakdown Point
The **breakdown point** is the smallest fraction of contamination that can make the estimator arbitrarily bad:
```
ε* = min{ε: sup_{G} ||T(εG + (1-ε)F)|| = ∞}
```

### Examples

**Sample mean:**
- IF(x; μ, F) = x - μ
- γ* = ∞ (unbounded)
- ε* = 0 (breakdown point = 0)

**Sample median:**
- IF(x; median, F) = sign(x - m)/(2f(m))
- γ* = 1/(2f(m)) (bounded if f(m) > 0)
- ε* = 1/2 (breakdown point = 50%)

## 8.5 Quantile Estimation

### Sample Quantiles
For α ∈ (0,1), the **sample α-quantile** is:
```
q̂ₐ = F̂ₙ⁻¹(α) = X₍k₎
```

where k = ⌈nα⌉ or using interpolation between order statistics.

### Asymptotic Distribution
Under regularity conditions:
```
√n(q̂ₐ - qₐ) ⇝ N(0, α(1-α)/f²(qₐ))
```

where f is the density at the true quantile qₐ.

### Confidence Intervals for Quantiles

**Distribution-free approach:** Using order statistics
```
P(X₍j₎ ≤ qₐ ≤ X₍k₎) = ∑ᵢ₌ⱼᵏ⁻¹ (n choose i) αⁱ(1-α)ⁿ⁻ⁱ
```

**Asymptotic approach:**
```
q̂ₐ ± z_{α/2} √(α(1-α)/(nf̂²(q̂ₐ)))
```

### Kernel Density Estimation for f(qₐ)
```
f̂(q̂ₐ) = (1/(nhₙ)) ∑ᵢ₌₁ⁿ K((q̂ₐ - Xᵢ)/hₙ)
```

## 8.6 Density Estimation

### Histogram
**Bin-based estimator:** Divide range into bins of width h:
```
f̂(x) = (1/nh) × (number of Xᵢ in bin containing x)
```

### Kernel Density Estimator
```
f̂(x) = (1/nh) ∑ᵢ₌₁ⁿ K((x - Xᵢ)/h)
```

where K is a kernel function and h is the bandwidth.

### Common Kernels

**Gaussian:** K(u) = (1/√(2π))e^{-u²/2}

**Epanechnikov:** K(u) = (3/4)(1-u²)I(|u| ≤ 1)

**Uniform:** K(u) = (1/2)I(|u| ≤ 1)

**Triangular:** K(u) = (1-|u|)I(|u| ≤ 1)

### Bias and Variance

**Bias:**
```
E[f̂(x)] - f(x) ≈ (h²/2)f''(x) ∫ u²K(u)du
```

**Variance:**
```
Var(f̂(x)) ≈ (f(x)/(nh)) ∫ K²(u)du
```

**Mean Squared Error:**
```
MSE(x) = Bias² + Variance ≈ (h⁴/4)(f''(x))²(∫u²K(u)du)² + (f(x)/(nh))∫K²(u)du
```

### Optimal Bandwidth
Minimizing integrated MSE leads to:
```
h_opt = C_K n^{-1/5}
```

where C_K depends on the kernel and unknown density characteristics.

**Silverman's rule of thumb:**
```
h = 1.06 σ̂ n^{-1/5}
```

## 8.7 Functional Delta Method

### Statement
If T is Hadamard differentiable at F with derivative T'_F, then:
```
√n(T(F̂ₙ) - T(F)) ⇝ N(0, σ²_T)
```

where σ²_T = ∫(T'_F(x))² dF(x).

### Applications

**Sample variance:**
```
√n(σ̂²ₙ - σ²) ⇝ N(0, μ₄ - σ⁴)
```

**Sample correlation:**
```
√n(ρ̂ₙ - ρ) ⇝ N(0, (1-ρ²)²)
```

for bivariate normal data.

## 8.8 Bootstrap for Statistical Functionals

### Bootstrap Approximation
Approximate the distribution of √n(T̂ₙ - T(F)) by the bootstrap distribution of √n(T* - T̂ₙ).

### Bootstrap Algorithm
1. Generate X₁*, ..., Xₙ* by sampling with replacement from {X₁, ..., Xₙ}
2. Compute T* = T(F̂ₙ*)
3. Repeat B times to get T₁*, ..., T_B*
4. Use empirical distribution of {√n(Tᵦ* - T̂ₙ)}

### Bootstrap Consistency
Under regularity conditions:
```
sup_x |P*(√n(T* - T̂ₙ) ≤ x) - P(√n(T̂ₙ - T(F)) ≤ x)| →ᵖ 0
```

## 8.9 Nonparametric Confidence Intervals

### Bootstrap Percentile Method
Use quantiles of bootstrap distribution:
```
CI = [T*_{(α/2)}, T*_{(1-α/2)}]
```

### Bootstrap-t Method
```
T*_i = (T*_i - T̂ₙ)/ŝe*(T*_i)
```

**Confidence interval:**
```
CI = [T̂ₙ - t*_{(1-α/2)}ŝe(T̂ₙ), T̂ₙ - t*_{(α/2)}ŝe(T̂ₙ)]
```

### Bias-Corrected and Accelerated (BCₐ)
More sophisticated bootstrap interval accounting for bias and skewness.

## 8.10 Goodness-of-Fit Tests

### Kolmogorov-Smirnov Test
**Test statistic:**
```
D_n = sup_x |F̂ₙ(x) - F₀(x)|
```

**Null distribution:** Under H₀: F = F₀, D_n has the Kolmogorov distribution.

**Critical values:** Reject H₀ if D_n > K_{α}/√n where K_α depends on significance level.

### Anderson-Darling Test
**Test statistic:**
```
A² = n ∫ (F̂ₙ(x) - F₀(x))² / (F₀(x)(1-F₀(x))) dF₀(x)
```

**Advantage:** More sensitive to tail differences than KS test.

### Cramér-von Mises Test
**Test statistic:**
```
W² = n ∫ (F̂ₙ(x) - F₀(x))² dF₀(x)
```

## 8.11 Two-Sample Tests

### Two-Sample Kolmogorov-Smirnov
**Test statistic:**
```
D_{m,n} = sup_x |F̂_m(x) - Ĝ_n(x)|
```

where F̂_m and Ĝ_n are empirical CDFs from two samples.

### Permutation Distribution
Under H₀: F = G, all (m+n choose m) ways of dividing the combined sample are equally likely.

### Asymptotic Distribution
```
√(mn/(m+n)) D_{m,n} ⇝ sup_t |B(t)|
```

where B(t) is a Brownian bridge.

## 8.12 Survival Analysis

### Kaplan-Meier Estimator
For censored survival data:
```
Ŝ(t) = ∏_{tᵢ≤t} (1 - dᵢ/nᵢ)
```

where dᵢ = deaths at time tᵢ, nᵢ = at risk at time tᵢ.

### Greenwood's Formula
**Variance estimator:**
```
Var(Ŝ(t)) ≈ (Ŝ(t))² ∑_{tᵢ≤t} dᵢ/(nᵢ(nᵢ - dᵢ))
```

### Log-rank Test
Compare survival curves between groups:
```
Z = (O₁ - E₁)/√V₁ ~ N(0,1)
```

under null hypothesis of equal survival.

## 8.13 Multivariate Extensions

### Multivariate Empirical Distribution
```
F̂ₙ(x) = (1/n) ∑ᵢ₌₁ⁿ I(X₁ᵢ ≤ x₁, ..., X_dᵢ ≤ x_d)
```

### Glivenko-Cantelli in Higher Dimensions
```
sup_x ||F̂ₙ(x) - F(x)|| →ᵃ·ˢ· 0
```

but convergence rate slower: O(n^{-1/(2+d)}) curse of dimensionality.

### Copula Estimation
**Empirical copula:**
```
Ĉₙ(u₁, ..., u_d) = F̂ₙ(F̂₁ₙ⁻¹(u₁), ..., F̂_dₙ⁻¹(u_d))
```

## 8.14 Functional Data Analysis

### Functional Empirical Process
For function-valued data X₁(t), ..., Xₙ(t):
```
F̂ₙ(x)(t) = (1/n) ∑ᵢ₌₁ⁿ I(Xᵢ(t) ≤ x)
```

### Principal Component Analysis
Decompose functional data using eigenfunctions of covariance operator.

### Functional Central Limit Theorem
Weak convergence in function spaces (e.g., C[0,1] or L²[0,1]).

## 8.15 High-Dimensional Considerations

### Curse of Dimensionality
In high dimensions:
- Sample size requirements grow exponentially
- All distances become similar
- Volume concentration phenomena

### Sparse Functionals
Focus on functionals that depend on few coordinates or have sparse representations.

### Random Matrix Theory
Eigenvalue distributions of sample covariance matrices when p ≈ n.

## 8.16 Computational Aspects

### Efficient Computation of F̂ₙ
- Sort data once: O(n log n)
- Evaluate at m points: O(m log n) using binary search

### Kernel Density Estimation
- Fast Fourier Transform methods
- Binning for large datasets
- Adaptive bandwidth selection

### Bootstrap Implementation
- Vectorized resampling
- Parallel computation
- Antithetic variables for variance reduction

## 8.17 Model Selection and Bandwidth Choice

### Cross-Validation for Density Estimation
**Leave-one-out likelihood:**
```
CV(h) = ∏ᵢ f̂₋ᵢ(Xᵢ)
```

where f̂₋ᵢ is density estimate without observation i.

### Plug-in Methods
Estimate unknown quantities in optimal bandwidth formula:
```
h_opt = C(∫f''(x)²dx)^{-1/5} n^{-1/5}
```

### Adaptive Methods
- Variable bandwidth kernels
- Local likelihood methods
- Wavelet-based approaches

## 8.18 Robustness and Trimming

### Trimmed Estimators
Remove extreme observations before computing functionals:
```
T_γ(F̂ₙ) = functional based on middle (1-2γ) fraction of data
```

### Winsorized Estimators
Replace extreme values with less extreme ones:
```
X_i^{(w)} = max(X_{(k)}, min(X_i, X_{(n-k+1)}))
```

### M-estimators
Solve:
```
∑ᵢ ψ((Xᵢ - θ)/σ) = 0
```

for robust location estimation.

## 8.19 Smoothing and Regularization

### Nadaraya-Watson Estimator
For regression function estimation:
```
m̂(x) = ∑ᵢ Yᵢ K((x-Xᵢ)/h) / ∑ᵢ K((x-Xᵢ)/h)
```

### Local Polynomial Regression
Fit polynomials locally using weighted least squares.

### Spline Smoothing
Minimize penalized sum of squares:
```
∑ᵢ (Yᵢ - g(Xᵢ))² + λ ∫ (g''(x))² dx
```

## 8.20 Empirical Likelihood

### Nonparametric Likelihood
Maximize:
```
L = ∏ᵢ pᵢ
```

subject to ∑pᵢ = 1, pᵢ ≥ 0, and moment constraints.

### Empirical Likelihood Ratio
```
R(θ) = max{∏ᵢ npᵢ : ∑pᵢg(Xᵢ,θ) = 0}
```

### Wilks' Theorem for Empirical Likelihood
```
-2 log R(θ₀) ⇝ χ²_k
```

under null hypothesis θ = θ₀.

## 8.21 Minimax Theory

### Minimax Rates for Density Estimation
Over Hölder class H(β, L):
```
inf_f̂ sup_{f∈H(β,L)} E[||f̂ - f||²] ≍ n^{-2β/(2β+1)}
```

### Adaptivity
Adaptive estimators achieve optimal rates without knowing smoothness β.

### Lower Bounds
**Fano's method:** Information-theoretic lower bounds
**Assouad's method:** Metric-based lower bounds

## 8.22 Bayesian Nonparametrics

### Dirichlet Process
Random probability measure with:
- **Concentration parameter:** α > 0
- **Base measure:** G₀
- **Stick-breaking construction**

### Gaussian Process Priors
For function estimation:
```
f ~ GP(μ, K)
```

where μ is mean function and K is covariance kernel.

### Posterior Consistency
Conditions for:
```
Π(d(f, f₀) > ε | X₁,...,Xₙ) → 0
```

almost surely.

## 8.23 Recent Developments

### Deep Learning for Density Estimation
- Normalizing flows
- Variational autoencoders
- Generative adversarial networks

### Optimal Transport
**Wasserstein distance:**
```
W_p(F,G) = (inf_{π∈Π(F,G)} ∫||x-y||^p dπ(x,y))^{1/p}
```

### Empirical Bayes
**Robbins' empirical Bayes:**
```
δ_n(X) = E[θ|X, F̂ₙ]
```

## 8.24 Applications

### Quality Control
Control charts based on empirical quantiles and robust estimators.

### Finance
- Value at Risk estimation using empirical quantiles
- Volatility modeling with kernel methods
- Risk measure estimation

### Bioinformatics
- Gene expression analysis
- Survival curve estimation
- Multiple testing with FDR control

### Environmental Statistics
- Extreme value analysis
- Spatial interpolation
- Change point detection

## 8.25 Software and Implementation

### R Packages
- **stats:** Basic empirical CDF functions
- **KernSmooth:** Kernel density estimation
- **survival:** Kaplan-Meier estimation
- **boot:** Bootstrap methods

### Python Libraries
- **scipy.stats:** Statistical functions
- **sklearn:** Kernel density estimation
- **lifelines:** Survival analysis

### Computational Considerations
- Memory efficiency for large datasets
- Streaming algorithms for online estimation
- GPU acceleration for kernel methods

## Key Insights

1. **Nonparametric Flexibility:** Makes minimal distributional assumptions while providing consistent estimation.

2. **Uniform Convergence:** Glivenko-Cantelli theorem provides foundation for distribution-free inference.

3. **Bootstrap Universality:** Works for general functionals without distributional knowledge.

4. **Bias-Variance Trade-off:** Central theme in bandwidth selection and smoothing.

5. **Robustness:** Empirical methods often more robust than parametric alternatives.

## Common Pitfalls

1. **Curse of dimensionality:** Performance degrades rapidly in high dimensions
2. **Boundary effects:** Kernel methods have issues near domain boundaries  
3. **Bandwidth selection:** Critical but difficult choice in practice
4. **Computational complexity:** Can be expensive for large datasets
5. **Interpretation:** Nonparametric estimates may be harder to interpret

## Theoretical Challenges

1. **Optimal rates:** Determining fundamental limits of estimation accuracy
2. **Adaptivity:** Achieving optimal rates without knowing regularity
3. **High dimensions:** Developing methods that scale well
4. **Dependence:** Extending to non-iid settings
5. **Model selection:** Choosing complexity in data-driven way

## Practical Guidelines

### When to Use Nonparametric Methods
- Unknown or complex distributional forms
- Robustness concerns
- Exploratory data analysis
- Small sample sizes where parametric assumptions questionable

### Implementation Tips
- Start with simple methods (empirical CDF, basic kernel density)
- Use cross-validation for tuning parameters
- Check robustness with different bandwidth choices
- Visualize results to check reasonableness

### Reporting Results
- Include confidence bands for CDFs
- Report bandwidth choices and sensitivity
- Compare with parametric alternatives when appropriate
- Discuss limitations due to sample size

## Connections to Other Chapters

### To Chapter 5 (Inequalities)
- Dvoretzky-Kiefer-Wolfowitz inequality
- Concentration bounds for empirical processes
- Uniform convergence rates

### To Chapter 6 (Convergence)
- Glivenko-Cantelli theorem
- Functional central limit theorem
- Weak convergence of empirical processes

### To Chapter 9 (Bootstrap)
- Bootstrap for functionals
- Bootstrap consistency theory
- Confidence interval construction

### To Chapter 11 (Hypothesis Testing)
- Kolmogorov-Smirnov tests
- Two-sample tests
- Goodness-of-fit procedures

### To Chapter 14 (Regression)
- Nonparametric regression
- Kernel regression methods
- Smoothing splines

This chapter provides the foundation for distribution-free statistical inference, bridging the gap between theoretical probability and practical data analysis without restrictive parametric assumptions.