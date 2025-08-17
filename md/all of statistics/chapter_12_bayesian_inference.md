# Chapter 12: Bayesian Inference - Mathematical Explanations

## Overview
Bayesian inference provides a framework for updating beliefs about parameters based on observed data. Unlike frequentist methods, Bayesian methods treat parameters as random variables and use probability to quantify uncertainty about them.

## 12.1 Bayesian Philosophy

### Frequentist vs Bayesian Paradigms

**Frequentist Postulates:**
- Probability refers to limiting relative frequencies
- Parameters are fixed, unknown constants
- Statistical procedures should have well-defined long-run frequency properties

**Bayesian Postulates:**
- Probability represents degrees of belief, not limiting frequency
- Parameters are random variables with probability distributions
- Inference is based on the posterior distribution of parameters given data

### Subjective vs Objective Probability
- **Subjective:** Personal degrees of belief
- **Objective:** Based on symmetry, indifference, or reference principles

## 12.2 The Bayesian Method

### Basic Framework
1. **Prior distribution:** π(θ) expresses beliefs about θ before seeing data
2. **Likelihood:** f(x|θ) describes how data depends on θ
3. **Posterior distribution:** π(θ|x) combines prior and likelihood via Bayes' theorem

### Bayes' Theorem
```
π(θ|x) = f(x|θ)π(θ) / m(x)
```

where m(x) = ∫ f(x|θ)π(θ) dθ is the **marginal likelihood** or **evidence**.

**Proportional form:**
```
π(θ|x) ∝ f(x|θ)π(θ)
```

### Sequential Updating
Today's posterior becomes tomorrow's prior:
```
π(θ|x₁, x₂) ∝ f(x₂|θ)π(θ|x₁)
```

## 12.3 Prior Distributions

### Types of Priors

**Informative priors:** Incorporate substantial prior knowledge
**Non-informative priors:** Minimal impact on posterior
**Conjugate priors:** Lead to closed-form posteriors

### Conjugate Families

**Beta-Binomial:**
- Prior: θ ~ Beta(α, β)
- Likelihood: X|θ ~ Binomial(n, θ)
- Posterior: θ|x ~ Beta(α + x, β + n - x)

**Gamma-Poisson:**
- Prior: λ ~ Gamma(α, β)
- Likelihood: X|λ ~ Poisson(λ)
- Posterior: λ|x ~ Gamma(α + ∑xᵢ, β + n)

**Normal-Normal (known variance):**
- Prior: μ ~ N(μ₀, τ²)
- Likelihood: X|μ ~ N(μ, σ²)
- Posterior: μ|x ~ N(μₙ, τₙ²)

where:
```
τₙ² = 1/(1/τ² + n/σ²)
μₙ = τₙ²(μ₀/τ² + nx̄/σ²)
```

### Non-informative Priors

**Uniform prior:** π(θ) ∝ 1 (often improper)

**Jeffreys' prior:** π(θ) ∝ √I(θ) where I(θ) is Fisher information
- **Invariant** under reparameterization
- **Example:** For normal mean with known variance: π(μ) ∝ 1

**Reference priors:** Maximize expected information gain

### Improper Priors
Priors that don't integrate to finite values:
- Often arise from limiting cases of proper priors
- Posterior may still be proper if likelihood is informative enough
- Require careful checking of posterior propriety

## 12.4 Posterior Inference

### Point Estimation

**Posterior mean:** θ̂ = E[θ|x] = ∫ θπ(θ|x) dθ
**Posterior median:** θ̃ such that P(θ ≤ θ̃|x) = 0.5
**Maximum a posteriori (MAP):** θ̂ₘₐₚ = argmax π(θ|x)

### Loss Functions and Decision Theory

**Squared error loss:** L(θ, a) = (θ - a)²
- **Optimal action:** Posterior mean

**Absolute error loss:** L(θ, a) = |θ - a|
- **Optimal action:** Posterior median

**0-1 loss:** L(θ, a) = I(θ ≠ a)
- **Optimal action:** Posterior mode

### Interval Estimation

**Credible intervals:** P(θ ∈ C|x) = 1 - α

**Equal-tailed interval:**
```
C = [θₐ/₂, θ₁₋ₐ/₂]
```

where θₚ is the p-th quantile of π(θ|x).

**Highest Posterior Density (HPD) interval:**
```
C = {θ : π(θ|x) ≥ k}
```

where k is chosen so that P(θ ∈ C|x) = 1 - α.

**Properties of HPD:**
- Shortest possible interval
- All points inside have higher density than points outside

## 12.5 Computational Methods

### When Posteriors Are Not Tractable
Most real problems don't have conjugate priors, requiring computational methods.

### Grid Approximation
1. Define grid of θ values
2. Compute π(θᵢ)f(x|θᵢ) for each grid point
3. Normalize to get approximate posterior

**Limitations:** Curse of dimensionality

### Monte Carlo Methods

**Basic idea:** Generate samples θ₁, ..., θₘ from π(θ|x)

**Posterior mean approximation:**
```
E[θ|x] ≈ (1/M) ∑ᵢ₌₁ᴹ θᵢ
```

**By Law of Large Numbers:** Approximation → true value as M → ∞

### Markov Chain Monte Carlo (MCMC)

**Goal:** Generate correlated samples that converge to posterior distribution

**Markov Chain:** Sequence θ₁, θ₂, ... where θₜ₊₁ depends only on θₜ

**Stationary distribution:** π(θ) such that if θₜ ~ π(θ), then θₜ₊₁ ~ π(θ)

### Metropolis-Hastings Algorithm

**Algorithm:**
1. Start with θ₀
2. For t = 1, 2, ...:
   - Propose θ* from proposal distribution q(θ*|θₜ₋₁)
   - Compute acceptance probability:
     ```
     α = min{1, [π(θ*)f(x|θ*)q(θₜ₋₁|θ*)] / [π(θₜ₋₁)f(x|θₜ₋₁)q(θ*|θₜ₋₁)]}
     ```
   - Accept θₜ = θ* with probability α, otherwise θₜ = θₜ₋₁

**Special cases:**
- **Random walk:** q(θ*|θ) = N(θ, σ²)
- **Independence sampler:** q(θ*|θ) = g(θ*)

### Gibbs Sampling
For multivariate θ = (θ₁, ..., θₖ):

**Algorithm:**
1. Initialize θ⁽⁰⁾ = (θ₁⁽⁰⁾, ..., θₖ⁽⁰⁾)
2. For t = 1, 2, ...:
   - θ₁⁽ᵗ⁾ ~ π(θ₁|θ₂⁽ᵗ⁻¹⁾, ..., θₖ⁽ᵗ⁻¹⁾, x)
   - θ₂⁽ᵗ⁾ ~ π(θ₂|θ₁⁽ᵗ⁾, θ₃⁽ᵗ⁻¹⁾, ..., θₖ⁽ᵗ⁻¹⁾, x)
   - ⋮
   - θₖ⁽ᵗ⁾ ~ π(θₖ|θ₁⁽ᵗ⁾, ..., θₖ₋₁⁽ᵗ⁾, x)

**Advantages:**
- No tuning of proposal distribution
- Always accepts proposals

**Requirements:** Must be able to sample from full conditional distributions

## 12.6 MCMC Diagnostics

### Convergence Assessment

**Trace plots:** Plot θₜ vs t
- Should look like "fuzzy caterpillar"
- No trends or patterns

**Running averages:** Plot (1/t)∑ᵢ₌₁ᵗ θᵢ vs t
- Should stabilize to posterior mean

**Multiple chains:** Run several chains from different starting points
- Should converge to same distribution

### Gelman-Rubin Diagnostic
```
R̂ = √[(n-1)/n + (1/n)(B/W)]
```

where:
- B = between-chain variance
- W = within-chain variance
- **Rule:** R̂ < 1.1 suggests convergence

### Effective Sample Size
Accounts for autocorrelation in MCMC samples:
```
ESS = M / (1 + 2∑ₖ₌₁^∞ ρₖ)
```

where ρₖ is lag-k autocorrelation.

### Burn-in
Discard initial samples before chain reaches stationarity:
- Typical: 10-50% of samples
- Assess using convergence diagnostics

## 12.7 Model Selection and Comparison

### Marginal Likelihood
For models M₁, ..., Mₖ:
```
P(Mⱼ|x) ∝ P(x|Mⱼ)P(Mⱼ)
```

where P(x|Mⱼ) = ∫ f(x|θ, Mⱼ)π(θ|Mⱼ) dθ

### Bayes Factors
```
BF₁₂ = P(x|M₁)/P(x|M₂)
```

**Interpretation:**
- BF₁₂ > 1: Evidence for M₁
- BF₁₂ = 1: Equal evidence
- BF₁₂ < 1: Evidence for M₂

**Jeffreys' Scale:**
| BF₁₂    | Evidence for M₁        |
|---------|------------------------|
| 1-3     | Barely worth mentioning|
| 3-10    | Substantial            |
| 10-30   | Strong                 |
| 30-100  | Very strong            |
| >100    | Decisive               |

### Information Criteria

**Deviance Information Criterion (DIC):**
```
DIC = D̄ + pD
```

where:
- D̄ = posterior mean deviance
- pD = effective number of parameters

**Watanabe-Akaike Information Criterion (WAIC):**
More robust alternative to DIC for hierarchical models.

### Cross-Validation
**Leave-one-out cross-validation:**
```
CV = ∑ᵢ₌₁ⁿ log p(yᵢ|y₋ᵢ)
```

Can be computed efficiently using Pareto smoothed importance sampling.

## 12.8 Hierarchical Models

### Structure
**Level 1:** Data model: yᵢⱼ|θᵢ ~ f(yᵢⱼ|θᵢ)
**Level 2:** Prior model: θᵢ|φ ~ π(θᵢ|φ)  
**Level 3:** Hyperprior: φ ~ π(φ)

### Example: Normal Hierarchical Model
```
yᵢⱼ|μᵢ, σ² ~ N(μᵢ, σ²)
μᵢ|μ, τ² ~ N(μ, τ²)
```

**Benefits:**
- **Shrinkage:** Individual estimates pulled toward group mean
- **Borrowing strength:** Information shared across groups
- **Handles unbalanced data** naturally

### Empirical Bayes
Estimate hyperparameters from data:
1. Estimate φ̂ from marginal distribution of y
2. Use φ̂ as known in posterior π(θ|y, φ̂)

**Pros:** Computationally simpler
**Cons:** Underestimates uncertainty

## 12.9 Bayesian Linear Regression

### Model
```
Y|β, σ² ~ N(Xβ, σ²I)
β ~ N(β₀, Σ₀)
σ² ~ IG(a, b)
```

### Posterior (conjugate case)
```
β|Y, σ² ~ N(βₙ, σ²Σₙ)
σ²|Y ~ IG(aₙ, bₙ)
```

where:
```
Σₙ = (Σ₀⁻¹ + XᵀX)⁻¹
βₙ = Σₙ(Σ₀⁻¹β₀ + XᵀY)
aₙ = a + n/2
bₙ = b + (1/2)[YᵀY + β₀ᵀΣ₀⁻¹β₀ - βₙᵀΣₙ⁻¹βₙ]
```

### Predictive Distribution
For new observation x*:
```
Y*|Y ~ tₙ₋ₚ(x*ᵀβₙ, sₙ²(1 + x*ᵀΣₙx*))
```

where sₙ² = bₙ/aₙ and degrees of freedom = 2aₙ.

### Variable Selection
**Spike-and-slab priors:**
```
βⱼ = γⱼδⱼ
γⱼ ~ Bernoulli(πⱼ)
δⱼ ~ N(0, τ²)
```

- γⱼ = 1: variable included
- γⱼ = 0: variable excluded

## 12.10 Bayesian Hypothesis Testing

### Point Null Hypotheses
Testing H₀: θ = θ₀ vs H₁: θ ≠ θ₀

**Savage-Dickey ratio:** When θ₀ is in support of prior:
```
BF₀₁ = π(θ₀)/π(θ₀|x)
```

### Composite Hypotheses
Testing H₀: θ ∈ Θ₀ vs H₁: θ ∈ Θ₁

**Bayes factor:**
```
BF₀₁ = ∫Θ₀ f(x|θ)π(θ|H₀) dθ / ∫Θ₁ f(x|θ)π(θ|H₁) dθ
```

### Posterior Probabilities
```
P(H₀|x) = BF₀₁ × P(H₀) / [BF₀₁ × P(H₀) + P(H₁)]
```

## 12.11 Asymptotic Properties

### Bernstein-von Mises Theorem
Under regularity conditions, as n → ∞:
```
π(θ|x) → N(θ̂ₘₗₑ, I(θ̂ₘₗₑ)⁻¹/n)
```

**Implications:**
- Posterior concentrates around MLE
- Credible intervals ≈ confidence intervals
- Prior becomes negligible

### Consistency
Under mild conditions:
```
π(|θ - θ₀| > ε|x) → 0
```

as n → ∞, where θ₀ is true parameter.

### Posterior Convergence Rates
For smooth parametric models: O(n⁻¹/²)
For nonparametric models: Depends on smoothness and dimension

## 12.12 Robust Bayesian Methods

### Prior Sensitivity Analysis
Study how posterior changes with different priors:
- **Class of priors:** Vary hyperparameters
- **Bounds:** Find range of posterior quantities

### Contamination Models
```
F = (1-ε)F₀ + εG
```

where F₀ is ideal model, G is contamination.

### Heavy-tailed Models
Use t-distributions instead of normal:
- More robust to outliers
- Adaptive degrees of freedom

## 12.13 Empirical Bayes

### Method
1. Estimate hyperparameters from marginal likelihood
2. Plug in estimates to get "posterior"

### Example: Normal Means
```
Xᵢ|θᵢ ~ N(θᵢ, 1)
θᵢ ~ N(0, A)
```

**Empirical Bayes estimate:**
```
θ̂ᵢᴱᴮ = (1 - (p-2)/||X||²)Xᵢ
```

**James-Stein estimator:** Dominates MLE when p ≥ 3

## 12.14 Modern Computational Methods

### Variational Inference
Approximate intractable posterior with simpler distribution:
```
q*(θ) = argmin KL(q(θ)||π(θ|x))
```

**Mean-field approximation:**
```
q(θ) = ∏ᵢ qᵢ(θᵢ)
```

### Hamiltonian Monte Carlo (HMC)
Uses gradient information for more efficient sampling:
- Fewer correlated samples
- Better for high-dimensional problems
- Implemented in Stan

### Approximate Bayesian Computation (ABC)
For likelihood-free inference:
1. Simulate data from prior predictive
2. Keep simulations where summary statistics ≈ observed
3. Use corresponding parameters as posterior sample

## 12.15 Practical Considerations

### Prior Specification
**Weakly informative priors:** Provide regularization without being dogmatic

**Example (logistic regression):**
```
βⱼ ~ N(0, 2.5²)
```

### Computational Efficiency
- **Reparameterization:** Improve mixing
- **Centered vs non-centered:** For hierarchical models
- **Parallel chains:** Use multiple cores

### Model Checking
**Posterior predictive checks:**
1. Generate replicated data from posterior predictive
2. Compare to observed data
3. Look for systematic discrepancies

**Test statistics:** T(y, θ)
- p-value = P(T(yʳᵉᵖ, θ) ≥ T(y, θ)|y)

## 12.16 Applications

### Clinical Trials
- **Adaptive designs:** Modify trial based on interim data
- **Historical controls:** Incorporate prior information
- **Safety monitoring:** Early stopping rules

### Machine Learning
- **Bayesian neural networks:** Uncertainty quantification
- **Gaussian processes:** Nonparametric regression
- **Bayesian optimization:** Efficient hyperparameter tuning

### A/B Testing
- **Beta-binomial model:** For conversion rates
- **Sequential testing:** Continuous monitoring
- **Multi-armed bandits:** Exploration vs exploitation

## 12.17 Comparison with Frequentist Methods

### Philosophical Differences
| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| Parameters | Fixed constants | Random variables |
| Probability | Long-run frequency | Degree of belief |
| Uncertainty | Via sampling distribution | Via posterior distribution |

### Practical Differences
**Advantages of Bayesian:**
- Natural uncertainty quantification
- Incorporates prior information
- Coherent decision theory
- Handles complex models naturally

**Advantages of Frequentist:**
- Objective procedures
- Well-understood properties
- Computational simplicity (traditionally)
- No need to specify priors

### When Methods Agree
- Large sample sizes
- Vague priors
- Regular parametric models

## Key Insights

1. **Coherent Framework:** Bayesian inference provides a complete, coherent framework for statistical inference.

2. **Uncertainty Quantification:** Posterior distributions directly quantify uncertainty about parameters.

3. **Prior Information:** Ability to incorporate prior knowledge is both strength and potential weakness.

4. **Computational Revolution:** MCMC has made Bayesian methods practical for complex models.

5. **Model Comparison:** Natural framework for comparing different models.

## Common Pitfalls

1. **Prior Sensitivity:** Results may depend heavily on prior specification
2. **Computational Issues:** MCMC can be slow to converge or mix poorly
3. **Model Misspecification:** All models are wrong, but some are useful
4. **Overfitting:** Complex models may not generalize
5. **Interpretation:** Credible intervals ≠ confidence intervals

This chapter provides a comprehensive foundation for Bayesian statistical inference, connecting theory with modern computational practice.