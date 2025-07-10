# Quick Reference: Mathematical Differences

## Core Model (Same for All Methods)
```
y = Xβ + ε,   ε ~ N(0, σ²ₑIₙ)
```

## Key Mathematical Differences

| Method | Prior Distribution | Key Parameters | Shrinkage Pattern |
|--------|-------------------|----------------|-------------------|
| **Bayesian Ridge** | `βⱼ ~ N(0, σ²ᵦ)` | σ²ᵦ (common variance) | Equal shrinkage: `βⱼ ← sβⱼ` |
| **Bayesian Lasso** | `βⱼ ~ Laplace(0, λ)` | λ (regularization) | Sparse: some βⱼ → 0 |
| **BayesA** | `βⱼ ~ N(0, σ²ᵦⱼ)` | σ²ᵦⱼ (individual variances) | Adaptive shrinkage |
| **BayesB** | `βⱼ = δⱼαⱼ, δⱼ ~ Ber(1-π)` | π (sparsity), σ²ᵦⱼ | Sparse + adaptive |

## Prior Density Functions

### Bayesian Ridge
```
p(β) = ∏ⱼ (2πσ²ᵦ)^(-1/2) exp(-βⱼ²/2σ²ᵦ)
```

### Bayesian Lasso  
```
p(β) = ∏ⱼ (λ/2) exp(-λ|βⱼ|)
```

### BayesA
```
p(β) = ∏ⱼ (2πσ²ᵦⱼ)^(-1/2) exp(-βⱼ²/2σ²ᵦⱼ)
p(σ²ᵦⱼ) = IG(νᵦ/2, Sᵦνᵦ/2)
```

### BayesB
```
p(βⱼ) = π·δ(βⱼ) + (1-π)·N(βⱼ|0, σ²ᵦⱼ)
p(σ²ᵦⱼ) = IG(νᵦ/2, Sᵦνᵦ/2)
```

## Posterior Updates

### Ridge Regression (Closed Form)
```
β|y ~ N(μₚₒₛₜ, Σₚₒₛₜ)

Σₚₒₛₜ = (X'X/σ²ₑ + Iₚ/σ²ᵦ)⁻¹
μₚₒₛₜ = Σₚₒₛₜ X'y/σ²ₑ
```

### BayesA (MCMC Required)
```
βⱼ|· ~ N(μⱼ, σ²ⱼ)
μⱼ = σ²ⱼ · x'ⱼrⱼ/σ²ₑ
σ²ⱼ = (x'ⱼxⱼ/σ²ₑ + 1/σ²ᵦⱼ)⁻¹

σ²ᵦⱼ|βⱼ ~ IG((νᵦ+1)/2, (νᵦSᵦ+βⱼ²)/2)
```

### BayesB (MCMC Required)
```
P(δⱼ=1|·) = w₁/(w₀+w₁)
w₀ = π
w₁ = (1-π)√(σ²ⱼ/2π) exp(μⱼ²/2σ²ⱼ)

If δⱼ=1: αⱼ|· ~ N(μⱼ, σ²ⱼ)
If δⱼ=0: αⱼ = 0
```

## Shrinkage Interpretation

### Mathematical Shrinkage Factors

**Ridge:**
```
Shrinkage factor = σ²ⱼ · x'ⱼxⱼ/σ²ₑ
Always: 0 < shrinkage < 1
```

**BayesA:**
```
Shrinkage factor = σ²ⱼ · x'ⱼxⱼ/σ²ₑ (adaptive per marker)
Varies by marker based on estimated σ²ᵦⱼ
```

**BayesB:**
```
Effective shrinkage = P(δⱼ=1) × shrinkage_factor
Can be 0 (complete) or partial
```

## Prediction Equations

### Point Predictions
All methods:
```
ŷ = X_new E[β|data]
```

### Uncertainty Quantification
```
Var[ŷ] = X_new Var[β|data] X'_new + σ²ₑ

Ridge: Var[β|data] = Σₚₒₛₜ (known)
BayesA/B: Var[β|data] ≈ sample variance from MCMC
```

## Hyperparameter Learning

### Empirical Bayes Updates

**Ridge:**
```
α̂ = tr(Σₚₒₛₜ(X'X/σ²ₑ))/||μₚₒₛₜ||²
```

**BayesA:**
```
Ŝᵦ = (Σⱼ βⱼ² + νᵦSᵦ)/(p + νᵦ)
```

**BayesB:**
```
π̂ = Σⱼ(1-δⱼ)/p
Ŝᵦ = (Σⱼ δⱼαⱼ² + νᵦSᵦ)/(Σⱼ δⱼ + νᵦ)
```

## Computational Complexity Summary

| Method | Matrix Inversion | Per Iteration | Memory |
|--------|-----------------|---------------|---------|
| **Ridge** | O(p³) once | O(p²) | O(p²) |
| **BayesA** | None | O(np) | O(np) |
| **BayesB** | None | O(np) | O(np) |

## When Each Prior is Appropriate

### Mathematical Intuition

**Ridge (Gaussian Prior):**
- Smooth shrinkage: `exp(-β²/2σ²)`
- All coefficients shrunk proportionally
- Good for: dense signals

**Lasso (Laplace Prior):**
- Sharp peak at zero: `exp(-λ|β|)`
- Promotes exact zeros
- Good for: sparse signals

**BayesA (Adaptive Gaussian):**
- Individual shrinkage: `exp(-βⱼ²/2σ²ᵦⱼ)`
- Each coefficient gets own variance
- Good for: varying effect sizes

**BayesB (Spike-and-Slab):**
- Mixture: `π·δ(0) + (1-π)·N(0,σ²)`
- Explicit sparsity modeling
- Good for: known sparsity level