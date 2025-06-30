# Mathematical Equations for Bayesian Methods

## 1. General Bayesian Shrinkage Estimation Framework

### Basic Model
All methods start with the linear model:
```
y = Xβ + ε
```
where:
- `y` = n×1 response vector (phenotypes, outcomes)
- `X` = n×p design matrix (predictors, genetic markers)
- `β` = p×1 coefficient vector (effects)
- `ε` = n×1 error vector

### Error Distribution
```
ε ~ N(0, σ²ₑI_n)
```

### Likelihood
```
p(y|β, σ²ₑ) = (2πσ²ₑ)^(-n/2) exp{-1/(2σ²ₑ) ||y - Xβ||²}
```

### General Bayesian Framework
The posterior distribution follows:
```
p(β|y, σ²ₑ) ∝ p(y|β, σ²ₑ) × p(β|hyperparameters)
```

The key difference between methods is in the **prior distribution** `p(β)`.

---

## 2. Bayesian Ridge Regression

### Prior Distribution
```
βⱼ ~ N(0, σ²ᵦ)  for j = 1, ..., p
```
or in vector form:
```
β ~ N(0, σ²ᵦI_p)
```

### Hyperprior (for automatic tuning)
```
σ²ᵦ ~ InverseGamma(a, b)
```
or equivalently using precision τ = 1/σ²ᵦ:
```
τ ~ Gamma(a, b)
```

### Posterior Distribution
The posterior for β is multivariate normal:
```
β|y, σ²ₑ, σ²ᵦ ~ N(μ_post, Σ_post)
```

where:
```
Σ_post = (X'X/σ²ₑ + I_p/σ²ᵦ)⁻¹
μ_post = Σ_post × X'y/σ²ₑ
```

### MAP (Maximum A Posteriori) Estimate
```
β̂ = (X'X + λI_p)⁻¹X'y
```
where `λ = σ²ₑ/σ²ᵦ` is the regularization parameter.

### Marginal Posterior for σ²ₑ
```
p(σ²ₑ|y) ∝ (σ²ₑ)^(-(n+2a)/2) exp{-1/(2σ²ₑ)[||y - Xμ_post||² + 2b]}
```

---

## 3. BayesA (Meuwissen et al., 2001)

### Prior Distribution
Each coefficient has its own variance:
```
βⱼ|σ²ᵦⱼ ~ N(0, σ²ᵦⱼ)  for j = 1, ..., p
```

### Hyperprior
```
σ²ᵦⱼ ~ InverseGamma(νᵦ/2, Sᵦνᵦ/2)
```

where:
- `νᵦ` = degrees of freedom (typically 4-5)
- `Sᵦ` = scale parameter

### Full Conditional Distributions

**For βⱼ:**
```
βⱼ|y, σ²ₑ, σ²ᵦⱼ ~ N(μⱼ, σ²ⱼ)
```
where:
```
σ²ⱼ = 1/(x'ⱼxⱼ/σ²ₑ + 1/σ²ᵦⱼ)
μⱼ = σ²ⱼ × x'ⱼ(y - X₋ⱼβ₋ⱼ)/σ²ₑ
```

**For σ²ᵦⱼ:**
```
σ²ᵦⱼ|βⱼ ~ InverseGamma((νᵦ + 1)/2, (νᵦSᵦ + β²ⱼ)/2)
```

**For σ²ₑ:**
```
σ²ₑ|y, β ~ InverseGamma((n + νₑ)/2, (||y - Xβ||² + νₑSₑ)/2)
```

### Expected Values
```
E[βⱼ|data] ≈ posterior mean from MCMC
E[σ²ᵦⱼ|data] ≈ posterior mean from MCMC
```

---

## 4. BayesB (Meuwissen et al., 2001)

### Prior Distribution with Variable Selection
```
βⱼ = δⱼ × αⱼ
```

where:
- `δⱼ ~ Bernoulli(1-π)` (inclusion indicator)
- `αⱼ|σ²ᵦⱼ ~ N(0, σ²ᵦⱼ)` (effect size if included)
- `σ²ᵦⱼ ~ InverseGamma(νᵦ/2, Sᵦνᵦ/2)`

### Prior Probabilities
```
P(δⱼ = 0) = π        (marker has no effect)
P(δⱼ = 1) = 1-π      (marker has effect)
```

Typical values: π = 0.95 (95% of markers have no effect)

### Full Conditional Distributions

**For δⱼ (inclusion indicator):**
```
P(δⱼ = 1|y, α, σ²ₑ) = w₁/(w₀ + w₁)
```

where:
```
w₀ = π
w₁ = (1-π) × √(σ²ⱼ/(2π)) × exp{μ²ⱼ/(2σ²ⱼ)}
```

**For αⱼ (if δⱼ = 1):**
```
αⱼ|δⱼ = 1, y, σ²ₑ, σ²ᵦⱼ ~ N(μⱼ, σ²ⱼ)
```
with the same μⱼ and σ²ⱼ as in BayesA.

**For σ²ᵦⱼ (if δⱼ = 1):**
```
σ²ᵦⱼ|δⱼ = 1, αⱼ ~ InverseGamma((νᵦ + 1)/2, (νᵦSᵦ + α²ⱼ)/2)
```

### Posterior Inclusion Probability
```
P(δⱼ = 1|data) = mean of δⱼ samples from MCMC
```

---

## 5. Comparison of Shrinkage Patterns

### Ridge Regression
```
β̂ⱼ = (1 + λ/x'ⱼxⱼ)⁻¹ × β̂ⱼ^(OLS)
```
**Shrinkage factor:** `sⱼ = 1/(1 + λ/x'ⱼxⱼ)`

### BayesA
```
E[βⱼ|data] ≈ sⱼ × β̂ⱼ^(conditional)
```
**Shrinkage factor:** `sⱼ = σ²ⱼ × x'ⱼxⱼ/σ²ₑ` (adaptive)

### BayesB
```
E[βⱼ|data] = P(δⱼ = 1|data) × E[αⱼ|δⱼ = 1, data]
```
**Shrinkage:** Can be complete (βⱼ = 0) or partial

---

## 6. Hyperparameter Updates (Empirical Bayes)

### For Ridge Regression
```
α̂ = p/||β||²
```

### For BayesA
```
Ŝᵦ = Σⱼ β²ⱼ/(p × νᵦ)
```

### For BayesB
```
π̂ = Σⱼ(1 - δⱼ)/p
Ŝᵦ = Σⱼ δⱼα²ⱼ/(Σⱼ δⱼ × νᵦ)
```

---

## 7. Prediction Equations

### Point Prediction
```
ŷ_new = X_new × E[β|data]
```

### Predictive Variance (Bayesian)
```
Var[y_new|data] = Var[X_new β|data] + σ²ₑ
                = X_new Var[β|data] X'_new + σ²ₑ
```

### Credible Intervals
```
y_new ± z_(α/2) × √Var[y_new|data]
```

---

## 8. MCMC Implementation Notes

### Gibbs Sampling Order
1. Update β (or α and δ for BayesB)
2. Update variance components σ²ᵦⱼ
3. Update residual variance σ²ₑ
4. Update hyperparameters if needed

### Convergence Diagnostics
- **Geweke statistic:** Compare means of first 10% and last 50% of chain
- **Heidelberger-Welch test:** Stationarity and half-width test
- **Effective sample size:** ESS > 100 per chain

### Burn-in and Thinning
- **Burn-in:** Discard first 10-20% of samples
- **Thinning:** Keep every k-th sample to reduce autocorrelation

---

## 9. Model Selection Criteria

### Deviance Information Criterion (DIC)
```
DIC = -2 × log p(y|E[θ]) + 2 × p_D
```
where `p_D` = effective number of parameters

### Watanabe-Akaike Information Criterion (WAIC)
```
WAIC = -2 × Σᵢ log E[p(yᵢ|θ)] + 2 × Σᵢ Var[log p(yᵢ|θ)]
```

### Cross-Validation
```
CV = Σᵢ (yᵢ - ŷᵢ^(-i))²
```
where `ŷᵢ^(-i)` is prediction with i-th observation removed.

---

## 10. Computational Complexity

### Per MCMC Iteration
- **Ridge:** O(p³) for matrix inversion, O(np²) for updates
- **BayesA:** O(np) for coefficient updates
- **BayesB:** O(np) for coefficient updates + O(p) for indicator variables

### Memory Requirements
- **Storage:** O(np) for data matrix X
- **MCMC samples:** O(T×p) where T = number of saved iterations

### Recommendations
- Use **conjugate priors** for computational efficiency
- Implement **efficient matrix operations** (BLAS/LAPACK)
- Consider **parallel computing** for large p
- Use **sparse matrix** representations when appropriate