# Chapter 7: Statistical Inference - Mathematical Explanations

## Overview
Statistical inference is the process of drawing conclusions about population parameters based on sample data. This chapter covers the fundamental concepts and methods of statistical inference, including point estimation, interval estimation, and the theoretical foundations that support these methods.

## 7.1 Introduction to Statistical Inference

### The Inference Problem
Given:
- Sample X₁, X₂, ..., Xₙ from population with distribution F
- Unknown parameter θ ∈ Θ

**Goals:**
1. **Point estimation:** Find θ̂, a single "best guess" for θ
2. **Interval estimation:** Find interval [L, U] likely to contain θ
3. **Hypothesis testing:** Decide between competing claims about θ

### Statistical Model
A **statistical model** is a set of probability distributions:
```
ℳ = {P_θ : θ ∈ Θ}
```

**Parametric model:** Θ has finite dimension (e.g., Θ ⊂ ℝᵈ)
**Nonparametric model:** Θ has infinite dimension (e.g., all continuous distributions)

### Well-posed Inference Problems
1. **Identifiability:** θ ≠ θ' ⟹ P_θ ≠ P_θ'
2. **Estimability:** Possible to learn about θ from data
3. **Regularity:** Smooth dependence on θ

## 7.2 Point Estimation

### Estimators and Estimates
An **estimator** θ̂ₙ = T(X₁, ..., Xₙ) is a function of the data.
An **estimate** is the numerical value θ̂ₙ(x₁, ..., xₙ) for observed data.

### Sampling Distribution
The **sampling distribution** of θ̂ₙ is its probability distribution across repeated samples.

### Mean Squared Error
```
MSE(θ̂) = E[(θ̂ - θ)²] = Var(θ̂) + (Bias(θ̂))²
```

where Bias(θ̂) = E[θ̂] - θ.

### Unbiasedness
θ̂ is **unbiased** if E[θ̂] = θ for all θ ∈ Θ.

**Examples:**
- Sample mean: E[X̄] = μ (unbiased for population mean)
- Sample variance: E[S²] = σ² where S² = (1/(n-1))∑(Xᵢ - X̄)²

### Consistency
θ̂ₙ is **consistent** if θ̂ₙ →ᵖ θ as n → ∞.

**Weak consistency:** θ̂ₙ →ᵖ θ
**Strong consistency:** θ̂ₙ →ᵃ·ˢ· θ

### Asymptotic Normality
θ̂ₙ is **asymptotically normal** if:
```
√n(θ̂ₙ - θ) ⇝ N(0, σ²)
```

for some σ² > 0.

## 7.3 Evaluating Estimators

### Efficiency
**Cramér-Rao Lower Bound:** For unbiased estimator θ̂:
```
Var(θ̂) ≥ 1/I(θ)
```

where I(θ) is Fisher information.

**Efficient estimator:** Achieves the Cramér-Rao bound.

### Relative Efficiency
For two unbiased estimators θ̂₁, θ̂₂:
```
eff(θ̂₁, θ̂₂) = Var(θ̂₂)/Var(θ̂₁)
```

### Asymptotic Efficiency
θ̂ₙ is **asymptotically efficient** if:
```
√n(θ̂ₙ - θ) ⇝ N(0, 1/I(θ))
```

### Robustness
**Breakdown point:** Smallest fraction of outliers that can make estimator arbitrarily bad.
**Influence function:** Effect of infinitesimal contamination.

**Examples:**
- Sample mean: breakdown point = 0
- Sample median: breakdown point = 50%

## 7.4 Methods of Point Estimation

### Method of Moments
Equate sample moments to population moments:
```
(1/n)∑Xᵢᵏ = E[Xᵏ], k = 1, ..., p
```

**Example (Normal distribution):**
```
μ̂ = X̄
σ̂² = (1/n)∑(Xᵢ - X̄)²
```

### Maximum Likelihood Estimation
**Likelihood function:**
```
L(θ) = ∏ᵢ₌₁ⁿ f(xᵢ; θ)
```

**MLE:** θ̂ₘₗₑ = argmax L(θ)

**Log-likelihood:**
```
ℓ(θ) = ∑ᵢ₌₁ⁿ log f(xᵢ; θ)
```

**Score function:**
```
S(θ) = ∂ℓ/∂θ
```

**Fisher information:**
```
I(θ) = E[S²(θ)] = -E[∂²ℓ/∂θ²]
```

### Properties of MLE
1. **Consistency:** θ̂ₘₗₑ →ᵖ θ₀
2. **Asymptotic normality:** √n(θ̂ₘₗₑ - θ₀) ⇝ N(0, I⁻¹(θ₀))
3. **Invariance:** If θ̂ is MLE of θ, then g(θ̂) is MLE of g(θ)
4. **Efficiency:** Asymptotically efficient in regular exponential families

### Least Squares Estimation
Minimize sum of squared residuals:
```
θ̂ₗₛ = argmin ∑ᵢ₌₁ⁿ (Yᵢ - g(Xᵢ; θ))²
```

**Linear regression:** g(x; θ) = θᵀx leads to:
```
θ̂ = (XᵀX)⁻¹XᵀY
```

## 7.5 Sufficient Statistics

### Definition
T(X) is **sufficient** for θ if the conditional distribution of X given T(X) doesn't depend on θ.

### Factorization Theorem
T is sufficient if and only if:
```
f(x; θ) = g(T(x); θ)h(x)
```

where g depends on θ and h doesn't.

### Examples
**Normal (μ unknown, σ² known):** T(X) = X̄
**Normal (both unknown):** T(X) = (X̄, ∑Xᵢ²)
**Exponential:** T(X) = ∑Xᵢ

### Minimal Sufficiency
T is **minimal sufficient** if it's a function of every other sufficient statistic.

### Rao-Blackwell Theorem
If T is sufficient and θ̂ is unbiased, then:
```
θ̂* = E[θ̂|T]
```

is unbiased with Var(θ̂*) ≤ Var(θ̂).

## 7.6 Completeness and Optimality

### Complete Statistics
T is **complete** if E[g(T)] = 0 for all θ implies g(T) = 0 almost surely.

### Lehmann-Scheffé Theorem
If T is complete and sufficient, and θ̂ is unbiased, then E[θ̂|T] is the unique UMVU estimator.

### Exponential Families
In canonical exponential families:
```
f(x; θ) = h(x)exp{θᵀT(x) - A(θ)}
```

- T(X) is sufficient
- T(X) is complete if parameter space contains open set
- ∇A(θ) is UMVU estimator of E[T(X)]

## 7.7 Interval Estimation

### Confidence Intervals
A **confidence interval** [L(X), U(X)] has **coverage probability**:
```
P_θ(L(X) ≤ θ ≤ U(X)) ≥ 1 - α
```

for all θ ∈ Θ.

**Interpretation:** In repeated sampling, (1-α)×100% of intervals contain true parameter.

### Constructing Confidence Intervals

**Pivotal method:** Find pivotal quantity Q(X,θ) with known distribution.

**Example (Normal mean, known variance):**
```
Q = (X̄ - μ)/(σ/√n) ~ N(0,1)
P(-z_{α/2} ≤ Q ≤ z_{α/2}) = 1 - α
```

**Confidence interval:**
```
[X̄ - z_{α/2}σ/√n, X̄ + z_{α/2}σ/√n]
```

### Asymptotic Confidence Intervals
For asymptotically normal estimator:
```
√n(θ̂ - θ) ⇝ N(0, σ²)
```

**Asymptotic CI:**
```
θ̂ ± z_{α/2} σ̂/√n
```

### Likelihood-based Intervals
**Likelihood ratio confidence region:**
```
{θ : 2[ℓ(θ̂) - ℓ(θ)] ≤ χ²_{p,α}}
```

where p = dim(θ).

## 7.8 Properties of Confidence Intervals

### Coverage Probability
Actual probability that interval contains parameter:
```
C(θ) = P_θ(θ ∈ [L(X), U(X)])
```

**Exact CI:** C(θ) = 1 - α for all θ
**Asymptotic CI:** C(θ) → 1 - α as n → ∞

### Length and Optimality
**Expected length:** E[U(X) - L(X)]
**Optimal CI:** Shortest expected length among all CIs with coverage ≥ 1 - α

### One-sided Intervals
**Upper confidence bound:** (-∞, U(X)] with P(θ ≤ U(X)) ≥ 1 - α
**Lower confidence bound:** [L(X), ∞) with P(θ ≥ L(X)) ≥ 1 - α

## 7.9 Prediction and Tolerance Intervals

### Prediction Intervals
Interval for future observation Y given past data X:
```
P(Y ∈ [L(X), U(X)]) ≥ 1 - α
```

**Example (Normal):** For Y ~ N(μ, σ²) with σ known:
```
[X̄ - z_{α/2}σ√(1 + 1/n), X̄ + z_{α/2}σ√(1 + 1/n)]
```

### Tolerance Intervals
Interval containing proportion p of population with confidence 1-α:
```
P(P_θ(X ∈ [L, U]) ≥ p) ≥ 1 - α
```

## 7.10 Large Sample Theory

### Consistency Results
**Weak Law of Large Numbers:** Sample mean converges in probability to population mean
**Strong Law of Large Numbers:** Sample mean converges almost surely

**Glivenko-Cantelli Theorem:** Empirical CDF converges uniformly to true CDF:
```
sup_x |F̂_n(x) - F(x)| →^{a.s.} 0
```

### Central Limit Theorem Applications
**Sample mean:**
```
√n(X̄ - μ) ⇝ N(0, σ²)
```

**Sample proportion:**
```
√n(p̂ - p) ⇝ N(0, p(1-p))
```

**Delta method:** For smooth function g:
```
√n(g(X̄) - g(μ)) ⇝ N(0, [g'(μ)]²σ²)
```

### Asymptotic Distributions of Estimators
**MLE asymptotic distribution:**
```
√n(θ̂_{MLE} - θ₀) ⇝ N(0, I⁻¹(θ₀))
```

**Method of moments:**
```
√n(θ̂_{MM} - θ₀) ⇝ N(0, Σ)
```

where Σ depends on moments and their gradients.

## 7.11 Multiparameter Inference

### Vector Parameters
For θ = (θ₁, ..., θₖ) ∈ ℝᵏ:

**Fisher Information Matrix:**
```
I(θ) = E[∇ℓ(θ) ∇ℓ(θ)ᵀ] = -E[∇²ℓ(θ)]
```

**Asymptotic distribution of MLE:**
```
√n(θ̂ - θ₀) ⇝ N(0, I⁻¹(θ₀))
```

### Confidence Regions
**Elliptical confidence region:**
```
{θ : n(θ̂ - θ)ᵀ Î(θ̂) (θ̂ - θ) ≤ χ²_{k,α}}
```

### Marginal vs Joint Inference
**Marginal confidence intervals:** For individual parameters
**Simultaneous confidence intervals:** Control family-wise error rate
**Bonferroni correction:** Use α/k for each of k intervals

## 7.12 Information and Efficiency

### Cramér-Rao Inequality
For unbiased estimator T of parameter θ:
```
Var(T) ≥ \frac{(\partial τ(θ)/\partial θ)²}{I(θ)}
```

where τ(θ) = E[T] and I(θ) is Fisher information.

### Information Processing Inequality
**Data processing reduces information:**
If Y = f(X), then I_Y(θ) ≤ I_X(θ)

### Fisher Information Properties
1. **Additivity:** For independent observations: I_n(θ) = nI(θ)
2. **Reparameterization:** I_φ(φ) = I_θ(θ)/[g'(θ)]² where φ = g(θ)
3. **Lower bound:** Provides fundamental limit on estimation accuracy

### Efficiency Comparisons
**Relative efficiency:**
```
e(T₁, T₂) = \frac{MSE(T₂)}{MSE(T₁)}
```

**Asymptotic relative efficiency:**
```
ARE(T₁, T₂) = \lim_{n→∞} \frac{nVar(T₂)}{nVar(T₁)}
```

## 7.13 Robust Inference

### Robust Statistics Motivation
Classical methods assume:
- Exact distributional form
- No outliers or contamination
- Model assumptions hold exactly

**Reality:** Models are approximations; data may be contaminated.

### M-Estimators
Generalize MLE by solving:
```
∑_{i=1}^n ψ(x_i, θ) = 0
```

where ψ is chosen for robustness.

**Huber M-estimator:**
```
ψ(x, θ) = \begin{cases}
x - θ & \text{if } |x - θ| ≤ k \\
k \cdot \text{sign}(x - θ) & \text{if } |x - θ| > k
\end{cases}
```

### Breakdown Point
**Definition:** Smallest fraction ε* of contamination that can make estimator arbitrarily bad.

**Examples:**
- Sample mean: ε* = 0
- Sample median: ε* = 50%
- Trimmed mean: ε* > 0

### Influence Function
**Definition:** Effect of infinitesimal contamination at point x:
```
IF(x; T, F) = \lim_{ε→0} \frac{T((1-ε)F + ε\delta_x) - T(F)}{ε}
```

**Gross error sensitivity:**
```
γ* = \sup_x |IF(x; T, F)|
```

## 7.14 Resampling Methods

### Bootstrap Principle
**Plug-in principle:** Replace unknown distribution F with empirical distribution F̂_n.

**Bootstrap sample:** X₁*, ..., X_n* sampled with replacement from {X₁, ..., X_n}

**Bootstrap distribution:** Distribution of T* = T(X₁*, ..., X_n*)

### Bootstrap Confidence Intervals

**Percentile method:**
```
[T*_{(α/2)}, T*_{(1-α/2)}]
```

**Bias-corrected percentile:**
```
[T*_{(α₁)}, T*_{(α₂)}]
```

where α₁, α₂ correct for bias and skewness.

### Jackknife
**Leave-one-out samples:** X₁^{(i)}, ..., X_{n-1}^{(i)} (remove i-th observation)

**Jackknife estimator:**
```
T_{jack} = nT_n - \frac{n-1}{n}\sum_{i=1}^n T_n^{(i)}
```

**Variance estimation:**
```
\hat{Var}(T_n) = \frac{n-1}{n}\sum_{i=1}^n (T_n^{(i)} - \bar{T}^{(·)})²
```

## 7.15 Nonparametric Inference

### Empirical CDF
```
F̂_n(x) = \frac{1}{n}\sum_{i=1}^n I(X_i ≤ x)
```

**Properties:**
- Unbiased: E[F̂_n(x)] = F(x)
- Consistent: F̂_n(x) →^{a.s.} F(x)
- Asymptotically normal: √n(F̂_n(x) - F(x)) ⇝ N(0, F(x)(1-F(x)))

### Quantile Estimation
**Sample quantile:**
```
q̂_α = F̂_n^{-1}(α) = X_{(⌈nα⌉)}
```

**Asymptotic distribution:**
```
√n(q̂_α - q_α) ⇝ N(0, α(1-α)/f²(q_α))
```

### Nonparametric Confidence Intervals
**Distribution-free intervals:** Based on order statistics, no distributional assumptions.

**Example (median):** For median m:
```
P(X_{(j)} ≤ m ≤ X_{(k)}) = \sum_{i=j}^{k-1} \binom{n}{i} 2^{-n}
```

## 7.16 Computational Inference

### Numerical Optimization
**Newton-Raphson method:**
```
θ^{(k+1)} = θ^{(k)} - H^{-1}(θ^{(k)})S(θ^{(k)})
```

where S is score function, H is Hessian.

**Fisher scoring:** Replace Hessian with Fisher information:
```
θ^{(k+1)} = θ^{(k)} + I^{-1}(θ^{(k)})S(θ^{(k)})
```

### EM Algorithm
For models with missing data or latent variables:

**E-step:** Compute Q(θ|θ^{(k)}) = E[log L(θ|X,Z)|X,θ^{(k)}]
**M-step:** θ^{(k+1)} = argmax_θ Q(θ|θ^{(k)})

### Markov Chain Monte Carlo (MCMC)
**Metropolis-Hastings algorithm:**
1. Propose θ* from q(·|θ^{(k)})
2. Accept with probability α = min{1, π(θ*)q(θ^{(k)}|θ*)/[π(θ^{(k)})q(θ*|θ^{(k)})]}
3. Set θ^{(k+1)} = θ* if accepted, θ^{(k)} otherwise

## 7.17 Model Selection and Validation

### Information Criteria
**Akaike Information Criterion:**
```
AIC = -2ℓ(θ̂) + 2k
```

**Bayesian Information Criterion:**
```
BIC = -2ℓ(θ̂) + k log n
```

**Interpretation:** Balance fit (likelihood) with complexity (number of parameters).

### Cross-Validation
**k-fold CV:**
1. Divide data into k folds
2. For each fold: train on k-1 folds, validate on remaining fold
3. Average validation errors

**Leave-one-out CV:**
```
CV = \frac{1}{n}\sum_{i=1}^n L(y_i, \hat{f}^{(-i)}(x_i))
```

### Model Assessment vs Selection
**Assessment:** How well does chosen model perform?
**Selection:** Which model to choose from candidates?

**Proper scoring rules:** Encourage honest probability assessments.

## 7.18 Minimax Theory

### Minimax Risk
**Risk function:** R(θ, δ) = E_θ[L(θ, δ(X))]
**Minimax risk:** inf_δ sup_θ R(θ, δ)

### Minimax Estimators
**Theorem:** Under squared error loss in exponential families, generalized Bayes estimators are often minimax.

### Lower Bounds
**Cramér-Rao bound:** Parametric case
**Fano's inequality:** Nonparametric case
**Le Cam's method:** Two-point testing problems

### Adaptive Estimation
**Goal:** Achieve optimal rate without knowing regularity (smoothness) of unknown function.

**Example:** Adaptive density estimation achieves optimal rate over range of smoothness classes.

## 7.19 High-Dimensional Inference

### Curse of Dimensionality
When dimension p is large relative to sample size n:
- Standard asymptotic theory may fail
- Sparsity assumptions often needed
- Regularization becomes essential

### Sparse Estimation
**LASSO:** ℓ₁ penalty encourages sparsity
**Ridge:** ℓ₂ penalty shrinks toward zero
**Elastic net:** Combination of ℓ₁ and ℓ₂ penalties

### Multiple Testing
When testing many hypotheses simultaneously:
**Family-wise error rate:** P(reject at least one true null)
**False discovery rate:** E[proportion of false discoveries]

**Bonferroni correction:** Control FWER at α/m
**Benjamini-Hochberg:** Control FDR at α

## 7.20 Causal Inference

### Correlation vs Causation
**Association:** X and Y are statistically related
**Causation:** X causes Y (intervention on X changes Y)

### Confounding
**Confounder:** Variable affecting both treatment and outcome
**Solution:** Randomization, stratification, matching, instrumental variables

### Potential Outcomes Framework
**Potential outcomes:** Y^{(1)}, Y^{(0)} (what would happen under treatment/control)
**Causal effect:** Y^{(1)} - Y^{(0)} for individual
**Average treatment effect:** E[Y^{(1)} - Y^{(0)}]

**Fundamental problem:** Can't observe both Y^{(1)} and Y^{(0)} for same individual.

## 7.21 Modern Developments

### Machine Learning Connections
**Bias-variance decomposition:**
```
E[(f̂(x) - f(x))²] = Bias²[f̂(x)] + Var[f̂(x)] + σ²
```

**Regularization:** Add penalty to prevent overfitting
**Cross-validation:** Estimate out-of-sample performance

### High-Dimensional Statistics
**Sparse models:** Most parameters are zero
**Variable selection:** Choose relevant predictors
**Compressed sensing:** Recover sparse signals from few measurements

### Computational Statistics
**Approximate inference:** Variational methods, ABC
**Scalable algorithms:** Stochastic gradient methods
**Distributed computing:** Parallel and distributed algorithms

## Key Insights

1. **Trade-offs:** Bias vs. variance, efficiency vs. robustness, simplicity vs. flexibility

2. **Large Sample Theory:** Provides foundation for practical inference methods

3. **Model Uncertainty:** Account for uncertainty in model choice, not just parameters

4. **Computational Revolution:** Modern inference relies heavily on computational methods

5. **Robustness:** Methods should work well under model misspecification

## Common Pitfalls

1. **Confusing confidence intervals with prediction intervals**
2. **Misinterpreting confidence level as probability**
3. **Ignoring model assumptions and their violations**
4. **Multiple testing without correction**
5. **Causal claims from observational data**

## Practical Guidelines

### Estimation Strategy
1. **Explore data** thoroughly before modeling
2. **Check assumptions** and assess their plausibility
3. **Compare multiple methods** and assess sensitivity
4. **Report uncertainty** honestly and completely
5. **Validate results** using independent data when possible

### Model Building
1. **Start simple** and add complexity gradually
2. **Use domain knowledge** to guide model specification
3. **Cross-validate** to assess out-of-sample performance
4. **Document assumptions** and their justification
5. **Consider robustness** to assumption violations

## Connections to Other Chapters

### To Chapter 6 (Convergence)
- Law of large numbers for consistency
- Central limit theorem for asymptotic normality
- Delta method for transformed estimators

### To Chapter 8 (CDF Estimation)
- Empirical distribution as nonparametric estimator
- Glivenko-Cantelli theorem
- Bootstrap as inference method

### To Chapter 10 (Parametric Inference)
- Maximum likelihood estimation
- Fisher information and efficiency
- Asymptotic theory

### To Chapter 11 (Hypothesis Testing)
- Relationship between confidence intervals and tests
- Power and sample size considerations
- Multiple testing corrections

This chapter provides the conceptual and theoretical foundation for all of statistical inference, bridging probability theory with practical data analysis methods.