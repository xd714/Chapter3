# Chapter 10: Parametric Inference - Mathematical Explanations

## Overview
Parametric inference assumes the data comes from a distribution that belongs to a known family, characterized by a finite-dimensional parameter vector. This chapter covers maximum likelihood estimation, method of moments, and the asymptotic theory that underlies modern statistical inference.

## 10.1 Parametric Models

### Definition
A **parametric model** is a family of distributions:
```
𝒫 = {Pθ : θ ∈ Θ}
```

where Θ ⊆ ℝᵈ is the parameter space.

### Examples

**Normal family:** 𝒫 = {N(μ, σ²) : μ ∈ ℝ, σ > 0}
- Parameter: θ = (μ, σ²)
- Parameter space: Θ = ℝ × ℝ⁺

**Exponential family:** 𝒫 = {Exp(λ) : λ > 0}
- Parameter: θ = λ
- Parameter space: Θ = ℝ⁺

**Binomial family:** 𝒫 = {Binomial(n, p) : p ∈ [0,1]}
- Parameter: θ = p (n assumed known)
- Parameter space: Θ = [0,1]

### Identifiability
A model is **identifiable** if different parameters yield different distributions:
```
θ ≠ θ' ⟹ Pθ ≠ Pθ'
```

**Non-identifiable example:** Mixture of two identical normals with unknown weights.

## 10.2 Method of Moments

### Population Moments
The **k-th population moment** is:
```
μₖ(θ) = E_θ[Xᵏ]
```

### Sample Moments
The **k-th sample moment** is:
```
m̂ₖ = (1/n) ∑ᵢ₌₁ⁿ Xᵢᵏ
```

### Method of Moments Estimator
**Equate sample and population moments:**
```
m̂ₖ = μₖ(θ), k = 1, ..., d
```

Solve this system for θ to get θ̂ₘₘ.

### Examples

**Normal distribution N(μ, σ²):**
```
μ̂ = m̂₁ = X̄
σ̂² = m̂₂ - (m̂₁)² = (1/n)∑Xᵢ² - X̄²
```

**Gamma distribution Γ(α, β):**
```
E[X] = αβ, Var(X) = αβ²
α̂ = X̄²/S², β̂ = S²/X̄
```

where S² is sample variance.

### Properties

**Consistency:** Under regularity conditions, θ̂ₘₘ →ᵖ θ

**Asymptotic normality:**
```
√n(θ̂ₘₘ - θ) ⇝ N(0, Σ)
```

where Σ depends on moments and their derivatives.

**Efficiency:** Generally not efficient (doesn't achieve Cramér-Rao bound).

## 10.3 Maximum Likelihood Estimation

### Likelihood Function
For observations x₁, ..., xₙ, the **likelihood function** is:
```
L(θ) = ∏ᵢ₌₁ⁿ f(xᵢ; θ)
```

### Log-likelihood
```
ℓ(θ) = log L(θ) = ∑ᵢ₌₁ⁿ log f(xᵢ; θ)
```

### Maximum Likelihood Estimator (MLE)
```
θ̂ₘₗₑ = argmax_θ L(θ) = argmax_θ ℓ(θ)
```

### First-order Conditions
**Score function:**
```
S(θ) = ∇ℓ(θ) = ∑ᵢ₌₁ⁿ ∇ log f(xᵢ; θ)
```

**MLE satisfies:** S(θ̂ₘₗₑ) = 0 (if interior maximum)

### Examples

**Normal distribution (known σ²):**
```
ℓ(μ) = -n log(σ√(2π)) - (1/(2σ²))∑(xᵢ - μ)²
∂ℓ/∂μ = (1/σ²)∑(xᵢ - μ) = 0
μ̂ = x̄
```

**Exponential distribution:**
```
ℓ(λ) = n log λ - λ∑xᵢ
∂ℓ/∂λ = n/λ - ∑xᵢ = 0
λ̂ = n/∑xᵢ = 1/x̄
```

**Binomial distribution:**
```
ℓ(p) = ∑xᵢ log p + (n-∑xᵢ) log(1-p)
∂ℓ/∂p = ∑xᵢ/p - (n-∑xᵢ)/(1-p) = 0
p̂ = ∑xᵢ/n = x̄
```

## 10.4 Properties of Maximum Likelihood Estimators

### Consistency
Under regularity conditions:
```
θ̂ₘₗₑ →ᵖ θ₀
```

where θ₀ is the true parameter.

### Asymptotic Normality
```
√n(θ̂ₘₗₑ - θ₀) ⇝ N(0, I⁻¹(θ₀))
```

where I(θ) is the Fisher information matrix.

### Asymptotic Efficiency
MLE achieves the Cramér-Rao lower bound asymptotically.

### Invariance Property
If θ̂ is MLE of θ, then g(θ̂) is MLE of g(θ) for any function g.

### Equivariance
MLE is equivariant under reparameterization.

## 10.5 Fisher Information

### Definition
**Fisher information** measures the amount of information about parameter θ contained in the data:
```
I(θ) = E[(∂ log f(X; θ)/∂θ)²] = -E[∂² log f(X; θ)/∂θ²]
```

### Properties

**Additivity:** For iid observations:
```
Iₙ(θ) = nI(θ)
```

**Information inequality:**
```
I(θ) = -E[∂²ℓ/∂θ²]
```

**Reparameterization:** If φ = g(θ):
```
I_φ(φ) = I_θ(g⁻¹(φ)) / (g'(g⁻¹(φ)))²
```

### Examples

**Normal N(μ, σ²) (μ unknown, σ² known):**
```
I(μ) = 1/σ²
```

**Exponential Exp(λ):**
```
I(λ) = 1/λ²
```

**Bernoulli Ber(p):**
```
I(p) = 1/(p(1-p))
```

## 10.6 Cramér-Rao Lower Bound

### Theorem
For any unbiased estimator T of θ:
```
Var(T) ≥ 1/I(θ)
```

**Multivariate version:** For unbiased estimator of θ ∈ ℝᵈ:
```
Cov(T) ⪰ I⁻¹(θ)
```

### Efficient Estimators
An estimator achieving the Cramér-Rao bound is called **efficient**.

**Theorem:** In exponential families, MLE is efficient.

### Rao-Blackwell Theorem
If T is unbiased for θ and S is sufficient, then:
```
T* = E[T|S]
```

is unbiased with Var(T*) ≤ Var(T).

## 10.7 Exponential Families

### Definition
A family of distributions is an **exponential family** if:
```
f(x; θ) = h(x) exp{η(θ)ᵀT(x) - A(θ)}
```

where:
- T(x): sufficient statistic
- η(θ): natural parameter  
- A(θ): log partition function
- h(x): base measure

### Canonical Form
When η(θ) = θ, we have **canonical** or **natural** exponential family:
```
f(x; θ) = h(x) exp{θᵀT(x) - A(θ)}
```

### Properties

**Sufficient statistic:** T(X) = ∑ᵢT(Xᵢ)

**Mean and variance:**
```
E[T(X)] = ∇A(θ)
Var(T(X)) = ∇²A(θ)
```

**Fisher information:** I(θ) = ∇²A(θ)

**MLE efficiency:** MLE achieves Cramér-Rao bound.

### Examples

**Normal (both parameters unknown):**
```
T(x) = (x, x²), θ = (μ/σ², -1/(2σ²))
```

**Poisson:**
```
T(x) = x, θ = log λ, A(θ) = eᶿ
```

**Binomial:**
```
T(x) = x, θ = log(p/(1-p)), A(θ) = n log(1 + eᶿ)
```

## 10.8 Sufficiency and Completeness

### Sufficient Statistic
T(X) is **sufficient** for θ if the conditional distribution of X given T(X) doesn't depend on θ.

**Factorization theorem:** T is sufficient if and only if:
```
f(x; θ) = g(T(x); θ)h(x)
```

### Minimal Sufficient Statistic
T is **minimal sufficient** if it's a function of every other sufficient statistic.

### Complete Statistic
T is **complete** if E[g(T)] = 0 for all θ implies g(T) = 0 a.s.

### Lehmann-Scheffé Theorem
If T is complete and sufficient, and W is unbiased for θ, then E[W|T] is the unique UMVU estimator.

## 10.9 Asymptotic Theory

### Regularity Conditions
1. **Identifiability:** Different θ give different distributions
2. **Common support:** Support doesn't depend on θ  
3. **Differentiability:** Log-likelihood is twice differentiable
4. **Information conditions:** Fisher information exists and is finite

### Consistency of MLE
Under regularity conditions:
```
θ̂ₙ →ᵖ θ₀
```

**Proof sketch:** Uses uniform law of large numbers and identification conditions.

### Asymptotic Normality
```
√n(θ̂ₙ - θ₀) ⇝ N(0, I⁻¹(θ₀))
```

**Proof sketch:** Taylor expansion of score function around true parameter.

### Delta Method Applications
For smooth function g(θ):
```
√n(g(θ̂ₙ) - g(θ₀)) ⇝ N(0, ∇g(θ₀)ᵀI⁻¹(θ₀)∇g(θ₀))
```

## 10.10 Hypothesis Testing in Parametric Models

### Likelihood Ratio Test
For testing H₀: θ ∈ Θ₀ vs H₁: θ ∈ Θ₁:
```
λ = 2[ℓ(θ̂) - ℓ(θ̂₀)]
```

**Wilks' theorem:** Under H₀:
```
λ ⇝ χ²_k
```

where k = dim(Θ) - dim(Θ₀).

### Wald Test
```
W = (θ̂ - θ₀)ᵀI(θ̂)(θ̂ - θ₀) ⇝ χ²_k
```

### Score Test (Lagrange Multiplier)
```
LM = S(θ₀)ᵀI⁻¹(θ₀)S(θ₀) ⇝ χ²_k
```

### Relationships
All three tests are asymptotically equivalent under H₀:
```
LRT ≈ Wald ≈ Score
```

## 10.11 Confidence Intervals and Regions

### Wald Confidence Interval
```
θ̂ ± z_{α/2}/√(nI(θ̂))
```

### Likelihood-based Intervals
```
{θ : 2[ℓ(θ̂) - ℓ(θ)] ≤ χ²₁,α}
```

### Profile Likelihood
For parameter of interest ψ = g(θ):
```
ℓₚ(ψ) = max_{θ:g(θ)=ψ} ℓ(θ)
```

**Profile likelihood interval:**
```
{ψ : 2[ℓₚ(ψ̂) - ℓₚ(ψ)] ≤ χ²₁,α}
```

## 10.12 Bayesian vs Frequentist Paradigms

### Frequentist Approach
- Parameters are fixed unknown constants
- Probability refers to repeated sampling
- Confidence intervals have coverage probability

### Bayesian Approach  
- Parameters are random variables
- Probability represents degree of belief
- Credible intervals contain parameter with given probability

### Asymptotic Agreement
Under regularity conditions:
- Posterior → N(θ̂ₘₗₑ, I⁻¹(θ̂ₘₗₑ)/n)
- Credible intervals ≈ confidence intervals

## 10.13 Model Selection

### Akaike Information Criterion (AIC)
```
AIC = -2ℓ(θ̂) + 2k
```

where k is the number of parameters.

### Bayesian Information Criterion (BIC)
```
BIC = -2ℓ(θ̂) + k log n
```

### Model Selection Procedure
1. Fit candidate models
2. Compute information criteria
3. Select model with smallest criterion

### Properties
- **AIC:** Asymptotically optimal for prediction
- **BIC:** Consistent for model selection

## 10.14 Robust Estimation

### M-estimators
Solve:
```
∑ᵢ ψ((xᵢ - θ)/σ) = 0
```

where ψ is chosen for robustness.

### Huber Estimator
```
ψ(x) = {
  x           if |x| ≤ k
  k·sign(x)   if |x| > k
}
```

### Breakdown Point
Fraction of outliers estimator can handle:
- Sample mean: 0%
- Sample median: 50%
- Huber estimator: depends on k

## 10.15 Bootstrap for Parametric Models

### Parametric Bootstrap
1. Estimate θ̂ from data
2. Generate bootstrap sample from f(·; θ̂)
3. Compute bootstrap statistic
4. Repeat to approximate sampling distribution

### Advantages
- More accurate than normal approximation
- Works with small samples
- Handles complex statistics

### Bootstrap Confidence Intervals
- Percentile method
- Bias-corrected methods  
- Bootstrap-t intervals

## 10.16 Computational Methods

### Newton-Raphson Algorithm
```
θ⁽ᵏ⁺¹⁾ = θ⁽ᵏ⁾ - H⁻¹(θ⁽ᵏ⁾)S(θ⁽ᵏ⁾)
```

where H is Hessian matrix.

### Fisher Scoring
Replace Hessian with Fisher information:
```
θ⁽ᵏ⁺¹⁾ = θ⁽ᵏ⁾ + I⁻¹(θ⁽ᵏ⁾)S(θ⁽ᵏ⁾)
```

### EM Algorithm
For models with missing data or latent variables:
- **E-step:** Compute expected complete data log-likelihood
- **M-step:** Maximize to update parameters

## 10.17 Multiparameter Extensions

### Score Vector
```
S(θ) = (∂ℓ/∂θ₁, ..., ∂ℓ/∂θₖ)ᵀ
```

### Fisher Information Matrix
```
I(θ) = E[S(θ)S(θ)ᵀ] = -E[∇²ℓ(θ)]
```

### Asymptotic Distribution
```
√n(θ̂ - θ) ⇝ N(0, I⁻¹(θ))
```

### Marginal vs Conditional Inference
- **Marginal:** Ignore nuisance parameters
- **Conditional:** Condition on sufficient statistic for nuisance parameters

## Key Insights

1. **Efficiency:** MLE is asymptotically efficient in regular exponential families.

2. **Invariance:** MLE is invariant under reparameterization.

3. **Large Sample Theory:** Provides foundation for inference via normal approximations.

4. **Information:** Fisher information quantifies parameter estimability.

5. **Sufficiency:** Sufficient statistics contain all information about parameters.

## Common Pitfalls

1. **Regularity violations:** Boundary parameters, non-smooth likelihoods
2. **Finite sample bias:** MLE can be biased in small samples  
3. **Multiple maxima:** Likelihood may have local maxima
4. **Model misspecification:** Wrong parametric family
5. **Numerical issues:** Optimization algorithms may fail

## Practical Guidelines

### Model Specification
- Use domain knowledge and exploratory analysis
- Check distributional assumptions with diagnostic plots
- Consider multiple candidate models

### Estimation
- Check convergence of optimization algorithms
- Verify MLE is global maximum
- Use multiple starting values

### Inference
- Report both estimates and uncertainties
- Use appropriate inference method (Wald, LRT, Score)
- Check robustness to model assumptions

## Connections to Other Chapters

### To Chapter 4 (Expectation)
- Method of moments based on population moments
- Fisher information as expectation of score squares

### To Chapter 6 (Convergence)
- Consistency and asymptotic normality of estimators
- Delta method applications

### To Chapter 11 (Hypothesis Testing)
- Likelihood ratio, Wald, and Score tests
- Connection between confidence intervals and tests

### To Chapter 12 (Bayesian Inference)
- Maximum likelihood vs. MAP estimation
- Asymptotic equivalence of frequentist and Bayesian methods

This chapter provides the foundation for parametric statistical inference, connecting probability theory with practical estimation and testing procedures under specific distributional assumptions.