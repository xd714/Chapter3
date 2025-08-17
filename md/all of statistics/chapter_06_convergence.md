# Chapter 6: Convergence of Random Variables - Mathematical Explanations

## Overview
This chapter explores different modes of convergence for sequences of random variables, which form the theoretical foundation for statistical inference. Understanding convergence is crucial for asymptotic theory, limit theorems, and the justification of statistical procedures.

## 6.1 Types of Convergence

### Overview of Convergence Modes
For a sequence of random variables X₁, X₂, ... and a random variable X, we have several ways they can "converge":

1. **Convergence in distribution** (weakest)
2. **Convergence in probability**
3. **Convergence in Lᵖ (mean)**
4. **Almost sure convergence** (strongest)

### Hierarchy of Convergence
```
Almost sure ⟹ In probability ⟹ In distribution
       ⇓               ⇓
   In L² ⟹ In L¹ ⟹ In probability
```

## 6.2 Convergence in Distribution

### Definition
Xₙ **converges in distribution** to X, written Xₙ ⇝ X or Xₙ →ᵈ X, if:
```
lim_{n→∞} Fₙ(x) = F(x)
```

at all continuity points of F, where Fₙ and F are the CDFs of Xₙ and X.

### Alternative Characterizations

**Portmanteau Theorem:** The following are equivalent:
1. Xₙ ⇝ X
2. E[f(Xₙ)] → E[f(X)] for all bounded continuous functions f
3. E[f(Xₙ)] → E[f(X)] for all bounded Lipschitz functions f
4. lim sup P(Xₙ ∈ C) ≤ P(X ∈ C) for all closed sets C
5. lim inf P(Xₙ ∈ G) ≥ P(X ∈ G) for all open sets G

### Properties

**Continuous Mapping Theorem:** If Xₙ ⇝ X and g is continuous, then g(Xₙ) ⇝ g(X).

**Slutsky's Theorem:** If Xₙ ⇝ X and Yₙ →ᵖ c (constant), then:
- Xₙ + Yₙ ⇝ X + c
- XₙYₙ ⇝ cX
- Xₙ/Yₙ ⇝ X/c (if c ≠ 0)

### Examples

**Convergence to constant:**
If Xₙ ⇝ c (constant), then Xₙ →ᵖ c.

**Standard normal approximation:**
If Xₙ ~ N(μₙ, σₙ²) with μₙ → μ and σₙ → σ, then Xₙ ⇝ N(μ, σ²).

## 6.3 Convergence in Probability

### Definition
Xₙ **converges in probability** to X, written Xₙ →ᵖ X, if:
```
lim_{n→∞} P(|Xₙ - X| > ε) = 0
```

for every ε > 0.

### Equivalent Formulation
```
lim_{n→∞} P(|Xₙ - X| ≤ ε) = 1
```

for every ε > 0.

### Properties

**Uniqueness:** If Xₙ →ᵖ X and Xₙ →ᵖ Y, then P(X = Y) = 1.

**Continuous Mapping:** If Xₙ →ᵖ X and g is continuous, then g(Xₙ) →ᵖ g(X).

**Preservation under arithmetic:** If Xₙ →ᵖ X and Yₙ →ᵖ Y, then:
- Xₙ + Yₙ →ᵖ X + Y
- XₙYₙ →ᵖ XY
- Xₙ/Yₙ →ᵖ X/Y (if P(Y = 0) = 0)

### Relationship to Convergence in Distribution
- **Forward:** Xₙ →ᵖ X ⟹ Xₙ ⇝ X
- **Reverse:** If X is constant and Xₙ ⇝ X, then Xₙ →ᵖ X

## 6.4 Almost Sure Convergence

### Definition
Xₙ **converges almost surely** to X, written Xₙ →ᵃ·ˢ· X, if:
```
P(lim_{n→∞} Xₙ = X) = 1
```

Equivalently:
```
P({ω : lim_{n→∞} Xₙ(ω) = X(ω)}) = 1
```

### Alternative Characterization
Xₙ →ᵃ·ˢ· X if and only if:
```
lim_{n→∞} P(sup_{k≥n} |Xₖ - X| > ε) = 0
```

for every ε > 0.

### Properties

**Stronger than convergence in probability:** Xₙ →ᵃ·ˢ· X ⟹ Xₙ →ᵖ X

**Subsequence property:** If Xₙ →ᵖ X, then there exists a subsequence Xₙₖ →ᵃ·ˢ· X.

**Continuous mapping:** If Xₙ →ᵃ·ˢ· X and g is continuous, then g(Xₙ) →ᵃ·ˢ· g(X).

## 6.5 Convergence in Lᵖ

### Definition
Xₙ **converges in Lᵖ** to X, written Xₙ →ᴸᵖ X, if:
```
lim_{n→∞} E[|Xₙ - X|ᵖ] = 0
```

**Special cases:**
- p = 1: Convergence in mean
- p = 2: Convergence in mean square (quadratic mean)

### Properties

**Moments must exist:** Requires E[|Xₙ|ᵖ] < ∞ and E[|X|ᵖ] < ∞.

**Implies convergence in probability:** Xₙ →ᴸᵖ X ⟹ Xₙ →ᵖ X

**Hölder's inequality connection:** If p < q, then Xₙ →ᴸᵠ X ⟹ Xₙ →ᴸᵖ X

### Convergence in Mean Square
For p = 2, we have useful properties:

**Variance decomposition:**
```
E[(Xₙ - X)²] = Var(Xₙ - X) + [E(Xₙ - X)]²
```

**Sufficient condition:** If E[Xₙ] → E[X] and Var(Xₙ) → Var(X), and Cov(Xₙ, X) → Var(X), then Xₙ →ᴸ² X.

## 6.6 Relationships Between Convergence Types

### Summary of Implications
```
Almost sure ⟹ In probability ⟹ In distribution
    ⇓               ⇓
In Lᵖ (p > 0) ⟹ In probability
```

### When Converses Hold

**Convergence in probability to almost sure:**
If Xₙ →ᵖ X and ∑P(|Xₙ - X| > ε) < ∞ for all ε > 0, then Xₙ →ᵃ·ˢ· X.

**Convergence in distribution to in probability:**
Only when the limit is constant.

### Dominated Convergence and Uniform Integrability

**Dominated Convergence Theorem:** If Xₙ →ᵃ·ˢ· X and |Xₙ| ≤ Y with E[Y] < ∞, then Xₙ →ᴸ¹ X.

**Uniform Integrability:** {Xₙ} is uniformly integrable if:
```
lim_{M→∞} sup_n E[|Xₙ|I(|Xₙ| > M)] = 0
```

**Vitali Convergence Theorem:** Xₙ →ᴸ¹ X if and only if Xₙ →ᵖ X and {Xₙ} is uniformly integrable.

## 6.7 Weak Law of Large Numbers

### Statement
Let X₁, X₂, ... be iid with E[Xᵢ] = μ and Var(Xᵢ) = σ² < ∞. Then:
```
X̄ₙ = (1/n)∑ᵢ₌₁ⁿ Xᵢ →ᵖ μ
```

### Proof Sketch
Use Chebyshev's inequality:
```
P(|X̄ₙ - μ| > ε) ≤ Var(X̄ₙ)/ε² = σ²/(nε²) → 0
```

### Extensions

**Khintchine's WLLN:** Only need E[|X₁|] < ∞ (finite mean), not finite variance.

**Cesàro means:** If aₙ → a, then (1/n)∑ᵢ₌₁ⁿ aᵢ → a.

## 6.8 Strong Law of Large Numbers

### Statement
Let X₁, X₂, ... be iid with E[|X₁|] < ∞ and E[X₁] = μ. Then:
```
X̄ₙ →ᵃ·ˢ· μ
```

### Kolmogorov's SLLN
If ∑ᵢ Var(Xᵢ)/i² < ∞, then:
```
(1/n)∑ᵢ₌₁ⁿ (Xᵢ - E[Xᵢ]) →ᵃ·ˢ· 0
```

### Proof Ideas

**Three-series theorem:** For convergence of ∑Xᵢ, need convergence of three series involving truncated variables.

**Borel-Cantelli lemma:** If ∑P(Aₙ) < ∞, then P(Aₙ i.o.) = 0.

**Maximal inequalities:** Control the maximum of partial sums.

### Applications

**Monte Carlo methods:** Sample averages converge to expectations.

**Empirical distribution:** Fₙ(x) →ᵃ·ˢ· F(x) for each x.

## 6.9 Central Limit Theorem

### Classical CLT
Let X₁, X₂, ... be iid with E[Xᵢ] = μ and Var(Xᵢ) = σ² ∈ (0, ∞). Then:
```
√n(X̄ₙ - μ)/σ ⇝ N(0, 1)
```

Equivalently:
```
√n(X̄ₙ - μ) ⇝ N(0, σ²)
```

### Lindeberg-Lévy CLT
More general version for independent (not necessarily identical) random variables with Lindeberg condition.

### Lyapunov CLT
If the Lyapunov condition holds:
```
lim_{n→∞} (1/s²ₙ) ∑ᵢ₌₁ⁿ E[|Xᵢ - μᵢ|²⁺ᵈ] = 0
```

for some δ > 0, then CLT holds.

### Multivariate CLT
For random vectors X₁, X₂, ... iid with mean μ and covariance Σ:
```
√n(X̄ₙ - μ) ⇝ N(0, Σ)
```

## 6.10 Delta Method

### First-order Delta Method
If √n(X̄ₙ - μ) ⇝ N(0, σ²) and g is differentiable at μ with g'(μ) ≠ 0, then:
```
√n(g(X̄ₙ) - g(μ)) ⇝ N(0, [g'(μ)]²σ²)
```

### Multivariate Delta Method
If √n(X̄ₙ - μ) ⇝ N(0, Σ) and g is differentiable at μ, then:
```
√n(g(X̄ₙ) - g(μ)) ⇝ N(0, ∇g(μ)ᵀΣ∇g(μ))
```

### Second-order Delta Method
When g'(μ) = 0, need second-order expansion:
```
n(g(X̄ₙ) - g(μ)) ⇝ (1/2)g''(μ)χ²₁σ²
```

### Applications

**Sample variance:** For variance of sample mean
**Correlation coefficient:** Asymptotic distribution
**Ratios of means:** Delta method for g(x, y) = x/y

## 6.11 Continuous Mapping Theorem

### Statement
If Xₙ ⇝ X and g is continuous at all points in the support of X, then:
```
g(Xₙ) ⇝ g(X)
```

### Extensions

**Almost sure version:** If Xₙ →ᵃ·ˢ· X and g is continuous, then g(Xₙ) →ᵃ·ˢ· g(X).

**Probability version:** If Xₙ →ᵖ X and g is continuous, then g(Xₙ) →ᵖ g(X).

**Discontinuous functions:** If g has only countably many discontinuities and P(X ∈ D_g) = 0, then g(Xₙ) ⇝ g(X).

## 6.12 Slutsky's Theorem

### Statement
If Xₙ ⇝ X and Yₙ →ᵖ c (constant), then:
- Xₙ + Yₙ ⇝ X + c
- XₙYₙ ⇝ cX  
- Xₙ/Yₙ ⇝ X/c (if c ≠ 0)

### General Version
If Xₙ ⇝ X and Yₙ →ᵖ Y, then (Xₙ, Yₙ) ⇝ (X, Y) where Y is constant.

### Applications

**t-statistics:** If Tₙ ⇝ N(0, 1) and S²ₙ →ᵖ σ², then Tₙ/Sₙ ⇝ N(0, 1/σ²).

**Standardization:** Converting to standard normal form.

## 6.13 Characteristic Functions and Convergence

### Definition
The **characteristic function** of random variable X is:
```
φₓ(t) = E[e^{itX}] = E[cos(tX)] + iE[sin(tX)]
```

### Lévy Continuity Theorem
Xₙ ⇝ X if and only if φₙ(t) → φ(t) for all t, where φ is continuous at 0.

**Key insight:** Convergence in distribution ⟺ pointwise convergence of characteristic functions

### Applications

**Proving CLT:** Show characteristic function of normalized sum converges to that of standard normal.

**Sum of independent variables:** φₓ₊ᵧ(t) = φₓ(t)φᵧ(t) when X, Y independent.

## 6.14 Moment Generating Functions

### Convergence of MGFs
If MGFs Mₙ(t) exist in neighborhood of 0 and Mₙ(t) → M(t) for all t in this neighborhood, then Xₙ ⇝ X where M is the MGF of X.

### Method of Moments
If all moments of Xₙ converge to those of X, and the moment sequence uniquely determines the distribution, then Xₙ ⇝ X.

**Carleman's condition:** If ∑(1/μ₂ₖ)^(1/2k) = ∞ where μ₂ₖ are even moments, then the distribution is uniquely determined.

## 6.15 Skorohod's Representation Theorem

### Statement
If Xₙ ⇝ X, then there exist random variables Yₙ, Y on a common probability space such that:
- Yₙ has the same distribution as Xₙ
- Y has the same distribution as X  
- Yₙ →ᵃ·ˢ· Y

**Interpretation:** Can always find almost sure convergent versions of convergent in distribution sequences.

### Applications
- Converting weak convergence to strong convergence
- Proving limit theorems
- Constructing couplings

## 6.16 Uniform Convergence and Glivenko-Cantelli

### Empirical Distribution Function
```
F̂ₙ(x) = (1/n)∑ᵢ₌₁ⁿ I(Xᵢ ≤ x)
```

### Glivenko-Cantelli Theorem
If X₁, X₂, ... are iid with CDF F, then:
```
sup_x |F̂ₙ(x) - F(x)| →ᵃ·ˢ· 0
```

**Uniform strong law:** Convergence is uniform over all x.

### Dvoretzky-Kiefer-Wolfowitz Inequality
```
P(sup_x |F̂ₙ(x) - F(x)| > ε) ≤ 2e^{-2nε²}
```

**Application:** Confidence bands for CDF.

## 6.17 Central Limit Theorems for Dependent Data

### Martingale CLT
For martingale differences Mₙ with appropriate conditions:
```
∑ᵢ₌₁ⁿ Mᵢ/√n ⇝ N(0, σ²)
```

### Stationary Sequences
For stationary ergodic sequences with finite second moments:
```
√n(X̄ₙ - μ) ⇝ N(0, σ²)
```

where σ² includes autocovariances.

### m-dependent Sequences
If Xᵢ and Xⱼ are independent when |i - j| > m, then CLT holds with appropriate variance formula.

## 6.18 Rates of Convergence

### Berry-Esseen Theorem
If E[|X₁|³] < ∞, then:
```
sup_x |P(√n(X̄ₙ - μ)/σ ≤ x) - Φ(x)| ≤ CE[|X₁ - μ|³]/(σ³√n)
```

where C ≤ 0.7655 is an absolute constant.

**Rate:** O(n^{-1/2}) convergence to normal distribution.

### Edgeworth Expansions
Higher-order corrections to normal approximation:
```
P(√n(X̄ₙ - μ)/σ ≤ x) = Φ(x) + n^{-1/2}p₁(x)φ(x) + n^{-1}p₂(x)φ(x) + O(n^{-3/2})
```

where p₁, p₂ are polynomials involving cumulants.

### Cramér's Theorem
If characteristic function satisfies |φ(t)| < 1 for |t| > δ > 0, then:
```
P(|X̄ₙ - μ| > ε) ≤ Ce^{-nI(ε)}
```

for some rate function I(ε) > 0. **Exponential convergence.**

## 6.19 Functional Central Limit Theorem

### Weak Convergence in Function Spaces
Consider the partial sum process:
```
Sₙ(t) = (1/√n)∑ᵢ₌₁^⌊nt⌋ (Xᵢ - μ)/σ
```

**Donsker's Theorem:** Sₙ ⇝ W in D[0,1], where W is Brownian motion.

### Applications
- **Kolmogorov-Smirnov test:** √n sup_x |F̂ₙ(x) - F₀(x)| ⇝ sup_{0≤t≤1} |B(F₀⁻¹(t))|
- **Empirical processes:** General theory for statistical inference

## 6.20 Large Deviations

### Cramér's Theorem
For iid Xᵢ with MGF, if I(x) = sup_t(tx - log M(t)), then:
```
lim_{n→∞} (1/n)log P(X̄ₙ > x) = -I(x)
```

### Azuma's Inequality
For martingales with bounded differences:
```
P(|Mₙ - M₀| > t) ≤ 2exp(-2t²/∑cᵢ²)
```

### Applications
- **Concentration inequalities:** Tail bounds for sums
- **Statistical mechanics:** Phase transitions
- **Information theory:** Error exponents

## 6.21 Exchangeability and de Finetti's Theorem

### Exchangeable Sequences
X₁, X₂, ... are **exchangeable** if finite permutations don't change joint distribution.

### de Finetti's Theorem
Infinite exchangeable sequence of 0-1 variables can be represented as:
```
P(X₁ = x₁, ..., Xₙ = xₙ) = ∫₀¹ ∏ᵢ₌₁ⁿ p^{xᵢ}(1-p)^{1-xᵢ} dμ(p)
```

**Interpretation:** Exchangeable sequences behave like iid conditional on random parameter.

## 6.22 Ergodic Theory

### Ergodic Sequences
A stationary sequence is **ergodic** if time averages equal ensemble averages.

### Birkhoff's Ergodic Theorem
For ergodic stationary sequence:
```
(1/n)∑ᵢ₌₁ⁿ f(Xᵢ) →ᵃ·ˢ· E[f(X₁)]
```

### Applications
- **Time series analysis:** Long-run behavior
- **Markov chains:** Limiting distributions
- **Dynamical systems:** Orbit averages

## 6.23 Extreme Value Theory

### Types of Convergence for Extremes
Let Mₙ = max{X₁, ..., Xₙ}. Under appropriate normalization:
```
P((Mₙ - bₙ)/aₙ ≤ x) → G(x)
```

where G is one of three types:
- **Gumbel:** G(x) = exp(-e^{-x})
- **Fréchet:** G(x) = exp(-x^{-α}) for x > 0
- **Weibull:** G(x) = exp(-(-x)^α) for x < 0

### Applications
- **Risk management:** Value at risk
- **Engineering:** Reliability analysis
- **Climate science:** Extreme weather events

## 6.24 Applications in Statistics

### Consistency of Estimators
**Weak consistency:** θ̂ₙ →ᵖ θ
**Strong consistency:** θ̂ₙ →ᵃ·ˢ· θ

**Example:** Sample mean is strongly consistent for population mean.

### Asymptotic Normality
Many estimators satisfy:
```
√n(θ̂ₙ - θ) ⇝ N(0, V(θ))
```

**Applications:** Confidence intervals, hypothesis tests.

### Efficiency
**Cramér-Rao bound:** Lower bound on variance of unbiased estimators.

**Asymptotic efficiency:** Estimator achieving the bound asymptotically.

### Bootstrap Consistency
Under regularity conditions:
```
sup_x |P*(√n(θ̂* - θ̂) ≤ x) - P(√n(θ̂ - θ) ≤ x)| →ᵖ 0
```

## 6.25 Computational Considerations

### Monte Carlo Methods
Law of large numbers justifies:
```
(1/n)∑ᵢ₌₁ⁿ g(Xᵢ) → E[g(X)]
```

### Markov Chain Monte Carlo
Ergodic theorem for Markov chains:
```
(1/n)∑ᵢ₌₁ⁿ f(Xᵢ) → ∫ f(x)π(x)dx
```

where π is stationary distribution.

### Stochastic Approximation
Robbins-Monro algorithm:
```
θₙ₊₁ = θₙ - aₙg(θₙ)
```

Under conditions on {aₙ}, θₙ → θ* where E[g(θ*)] = 0.

## Key Theorems Summary

### Fundamental Limit Theorems
1. **Weak Law of Large Numbers:** X̄ₙ →ᵖ μ
2. **Strong Law of Large Numbers:** X̄ₙ →ᵃ·ˢ· μ  
3. **Central Limit Theorem:** √n(X̄ₙ - μ) ⇝ N(0, σ²)
4. **Glivenko-Cantelli:** sup_x |F̂ₙ(x) - F(x)| →ᵃ·ˢ· 0

### Key Techniques
1. **Continuous Mapping Theorem:** Preserve convergence under continuous functions
2. **Slutsky's Theorem:** Combine convergent sequences
3. **Delta Method:** Asymptotic distribution of smooth functions
4. **Characteristic Functions:** Tool for proving convergence in distribution

## Practical Implications

### Statistical Inference
- **Large sample theory:** Justification for normal approximations
- **Consistency:** Estimators converge to true values
- **Efficiency:** Optimal asymptotic behavior

### Machine Learning
- **Generalization:** Training error converges to test error
- **Consistency:** Learning algorithms converge to optimal predictors
- **Concentration:** Uniform convergence bounds

### Computational Statistics
- **Monte Carlo:** Sample averages approximate expectations
- **Bootstrap:** Resampling approximates sampling distributions
- **MCMC:** Markov chain averages approximate integrals

## Common Pitfalls

1. **Confusing convergence types:** Each has different implications
2. **Assuming independence:** Many results require careful verification
3. **Finite sample behavior:** Asymptotic results may not apply to small samples
4. **Regularity conditions:** Technical conditions often crucial

## Connections to Other Chapters

### To Chapter 5 (Inequalities)
- Concentration inequalities
- Probability bounds
- Moment bounds

### To Chapter 8 (Estimating CDF)
- Glivenko-Cantelli theorem
- Empirical processes
- Uniform convergence

### To Chapter 9 (Bootstrap)
- Consistency of bootstrap
- Convergence of bootstrap distributions

### To Chapter 10-11 (Inference)
- Asymptotic normality of estimators
- Consistency of tests
- Large sample theory

This chapter provides the theoretical foundation that underlies most of modern statistical inference, connecting probability theory with practical statistical methods.