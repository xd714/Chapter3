# Chapter 4: Expectation - Mathematical Explanations

## Overview
Expectation (expected value) is one of the most fundamental concepts in probability and statistics. This chapter covers the definition, properties, and applications of expectation for both discrete and continuous random variables, along with variance, covariance, and other important moments.

## 4.1 Definition of Expectation

### Discrete Random Variables
For discrete random variable X with probability mass function f(x):
```
E[X] = ∑ₓ x f(x)
```

**Existence condition:** The sum must converge absolutely: ∑ₓ |x|f(x) < ∞

### Continuous Random Variables
For continuous random variable X with probability density function f(x):
```
E[X] = ∫₋∞^∞ x f(x) dx
```

**Existence condition:** The integral must converge absolutely: ∫₋∞^∞ |x|f(x) dx < ∞

### General Definition (Lebesgue Integration)
For any random variable X:
```
E[X] = ∫ X dP
```

This unifies discrete and continuous cases under measure theory.

## 4.2 Expectation of Functions

### Law of the Unconscious Statistician (LOTUS)
For function g(X):

**Discrete case:**
```
E[g(X)] = ∑ₓ g(x) f(x)
```

**Continuous case:**
```
E[g(X)] = ∫₋∞^∞ g(x) f(x) dx
```

**Key insight:** No need to find the distribution of Y = g(X) to compute E[Y].

### Examples

**Quadratic function:** E[X²] = ∫ x² f(x) dx

**Exponential function:** E[e^X] (moment generating function at t = 1)

**Indicator function:** E[I_A] = P(A) where I_A is indicator of event A

## 4.3 Properties of Expectation

### Linearity
For constants a, b and random variables X, Y:
```
E[aX + bY] = aE[X] + bE[Y]
```

**Extension:** For X₁, ..., Xₙ and constants a₁, ..., aₙ:
```
E[∑ᵢ aᵢXᵢ] = ∑ᵢ aᵢE[Xᵢ]
```

**Note:** This holds regardless of dependence structure between variables.

### Monotonicity
If X ≤ Y (almost surely), then E[X] ≤ E[Y].

### Triangle Inequality
```
|E[X]| ≤ E[|X|]
```

### Expectation of Products
**Independent variables:** If X and Y are independent:
```
E[XY] = E[X]E[Y]
```

**General case:** E[XY] ≠ E[X]E[Y] in general when dependent.

## 4.4 Variance and Standard Deviation

### Definition
The **variance** of X is:
```
Var(X) = E[(X - E[X])²]
```

**Alternative formula:**
```
Var(X) = E[X²] - (E[X])²
```

### Standard Deviation
```
SD(X) = σ(X) = √Var(X)
```

### Properties of Variance

**Constant:** Var(c) = 0 for any constant c

**Scaling:** Var(aX) = a²Var(X)

**Translation invariance:** Var(X + b) = Var(X)

**General linear transformation:** Var(aX + b) = a²Var(X)

### Variance of Sums
**Independent variables:**
```
Var(X + Y) = Var(X) + Var(Y)
```

**General case:**
```
Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y)
```

**Extension:** For X₁, ..., Xₙ:
```
Var(∑ᵢ Xᵢ) = ∑ᵢ Var(Xᵢ) + 2∑ᵢ<ⱼ Cov(Xᵢ, Xⱼ)
```

## 4.5 Covariance and Correlation

### Covariance
```
Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
```

### Properties of Covariance

**Symmetry:** Cov(X, Y) = Cov(Y, X)

**Bilinearity:** Cov(aX + bZ, Y) = aCov(X, Y) + bCov(Z, Y)

**Self-covariance:** Cov(X, X) = Var(X)

**Independence:** If X and Y are independent, then Cov(X, Y) = 0

**Note:** Cov(X, Y) = 0 does not imply independence (except for jointly normal variables).

### Correlation Coefficient
```
ρ(X, Y) = Corr(X, Y) = Cov(X, Y) / (SD(X) · SD(Y))
```

### Properties of Correlation

**Range:** -1 ≤ ρ(X, Y) ≤ 1

**Perfect correlation:** |ρ(X, Y)| = 1 if and only if Y = aX + b for some constants a, b

**Unitless:** Correlation is invariant under linear transformations

**Cauchy-Schwarz:** |Cov(X, Y)| ≤ √(Var(X)Var(Y))

## 4.6 Conditional Expectation

### Definition
For discrete Y, the conditional expectation of X given Y = y is:
```
E[X|Y = y] = ∑ₓ x P(X = x|Y = y)
```

For continuous Y:
```
E[X|Y = y] = ∫ x f_{X|Y}(x|y) dx
```

### Properties

**Linearity:** E[aX + bZ|Y] = aE[X|Y] + bE[Z|Y]

**Law of total expectation:** E[X] = E[E[X|Y]]

**Independence:** If X and Y are independent, then E[X|Y] = E[X]

### Conditional Variance
```
Var(X|Y) = E[X²|Y] - (E[X|Y])²
```

**Law of total variance:**
```
Var(X) = E[Var(X|Y)] + Var(E[X|Y])
```

## 4.7 Moments and Moment Generating Functions

### Moments
The **k-th moment** of X is:
```
μₖ = E[Xᵏ]
```

The **k-th central moment** is:
```
μₖ' = E[(X - E[X])ᵏ]
```

**Common moments:**
- μ₁ = E[X] (mean)
- μ₂' = Var(X) (variance)
- μ₃' relates to skewness
- μ₄' relates to kurtosis

### Moment Generating Function (MGF)
```
M_X(t) = E[e^{tX}]
```

when this expectation exists for t in some neighborhood of 0.

### Properties of MGFs

**Uniqueness:** If M_X(t) = M_Y(t) for all t in a neighborhood of 0, then X and Y have the same distribution.

**Moments from MGF:** If MGF exists, then:
```
E[Xᵏ] = M_X^{(k)}(0)
```

**Sum of independent variables:** If X and Y are independent:
```
M_{X+Y}(t) = M_X(t) · M_Y(t)
```

**Linear transformation:** M_{aX+b}(t) = e^{bt} M_X(at)

## 4.8 Characteristic Functions

### Definition
```
φ_X(t) = E[e^{itX}] = E[cos(tX)] + iE[sin(tX)]
```

### Advantages over MGFs
- Always exists (unlike MGF)
- Uniquely determines distribution
- Useful for proving limit theorems

### Properties

**Inversion formula:** Can recover density/mass function from characteristic function

**Continuity theorem:** Convergence of characteristic functions implies convergence in distribution

## 4.9 Cumulant Generating Function

### Definition
```
K_X(t) = log M_X(t)
```

### Cumulants
```
κₙ = K_X^{(n)}(0)
```

**Relationships:**
- κ₁ = μ₁ = E[X]
- κ₂ = μ₂' = Var(X)
- κ₃ = μ₃' (related to skewness)
- κ₄ = μ₄' - 3(μ₂')² (related to excess kurtosis)

### Property
For independent X and Y:
```
K_{X+Y}(t) = K_X(t) + K_Y(t)
```

Cumulants of sums equal sums of cumulants.

## 4.10 Inequalities

### Markov's Inequality
For non-negative random variable X and a > 0:
```
P(X ≥ a) ≤ E[X]/a
```

### Chebyshev's Inequality
For any random variable X with finite variance and k > 0:
```
P(|X - E[X]| ≥ k) ≤ Var(X)/k²
```

**In terms of standard deviations:**
```
P(|X - μ| ≥ kσ) ≤ 1/k²
```

### Jensen's Inequality
For convex function g:
```
g(E[X]) ≤ E[g(X)]
```

For concave function g:
```
g(E[X]) ≥ E[g(X)]
```

**Applications:**
- E[X²] ≥ (E[X])² (since g(x) = x² is convex)
- log E[X] ≥ E[log X] (since log is concave, for positive X)

### Cauchy-Schwarz Inequality
```
(E[XY])² ≤ E[X²]E[Y²]
```

**Equality condition:** When Y = aX for some constant a.

### Hölder's Inequality
For p, q > 1 with 1/p + 1/q = 1:
```
E[|XY|] ≤ (E[|X|ᵖ])^{1/p} (E[|Y|ᵠ])^{1/q}
```

## 4.11 Expectation for Common Distributions

### Discrete Distributions

**Bernoulli(p):** E[X] = p, Var(X) = p(1-p)

**Binomial(n, p):** E[X] = np, Var(X) = np(1-p)

**Geometric(p):** E[X] = 1/p, Var(X) = (1-p)/p²

**Poisson(λ):** E[X] = λ, Var(X) = λ

**Negative Binomial(r, p):** E[X] = r(1-p)/p, Var(X) = r(1-p)/p²

### Continuous Distributions

**Uniform(a, b):** E[X] = (a+b)/2, Var(X) = (b-a)²/12

**Normal(μ, σ²):** E[X] = μ, Var(X) = σ²

**Exponential(λ):** E[X] = 1/λ, Var(X) = 1/λ²

**Gamma(α, β):** E[X] = αβ, Var(X) = αβ²

**Beta(α, β):** E[X] = α/(α+β), Var(X) = αβ/((α+β)²(α+β+1))

## 4.12 Multivariate Expectations

### Mean Vector
For random vector X = (X₁, ..., Xₚ):
```
E[X] = (E[X₁], ..., E[Xₚ])ᵀ
```

### Covariance Matrix
```
Cov(X) = E[(X - E[X])(X - E[X])ᵀ]
```

**Element-wise:** [Cov(X)]ᵢⱼ = Cov(Xᵢ, Xⱼ)

### Properties

**Positive semi-definite:** Cov(X) ≽ 0

**Linear transformation:** If Y = AX + b, then:
- E[Y] = AE[X] + b  
- Cov(Y) = ACov(X)Aᵀ

**Correlation matrix:**
```
Corr(X) = D⁻¹Cov(X)D⁻¹
```

where D = diag(√Var(X₁), ..., √Var(Xₚ))

## 4.13 Expectations and Limits

### Monotone Convergence Theorem
If 0 ≤ X₁ ≤ X₂ ≤ ... and Xₙ → X, then:
```
E[Xₙ] → E[X]
```

### Dominated Convergence Theorem
If |Xₙ| ≤ Y with E[Y] < ∞ and Xₙ → X, then:
```
E[Xₙ] → E[X]
```

### Fatou's Lemma
For non-negative random variables:
```
E[lim inf Xₙ] ≤ lim inf E[Xₙ]
```

## 4.14 Conditional Expectation as Random Variable

### General Definition
E[X|Y] is a random variable (function of Y) such that:
```
E[I_A E[X|Y]] = E[I_A X]
```

for all events A in σ(Y).

### Properties

**Tower property:** E[E[X|Y]] = E[X]

**Linearity:** E[aX + bZ|Y] = aE[X|Y] + bE[Z|Y]

**Taking out what is known:** E[XZ|Y] = ZE[X|Y] if Z is a function of Y

**Independence:** E[X|Y] = E[X] if X and Y are independent

### Best Predictor Property
E[X|Y] minimizes E[(X - g(Y))²] over all functions g.

## 4.15 Applications in Statistics

### Sample Mean
For X₁, ..., Xₙ iid with mean μ:
```
E[X̄] = μ
Var(X̄) = σ²/n
```

### Sample Variance
```
E[S²] = σ²
```

where S² = (1/(n-1))∑(Xᵢ - X̄)²

### Method of Moments
Estimate parameters by equating sample moments to population moments:
```
μ̂ₖ = (1/n)∑Xᵢᵏ = μₖ(θ)
```

### Linear Regression
In Y = β₀ + β₁X + ε with E[ε|X] = 0:
```
β₁ = Cov(X,Y)/Var(X)
β₀ = E[Y] - β₁E[X]
```

## 4.16 Computational Aspects

### Numerical Integration
For continuous distributions without closed-form expectations:
- Gaussian quadrature
- Monte Carlo integration
- Adaptive quadrature methods

### Monte Carlo Estimation
```
E[g(X)] ≈ (1/n)∑ᵢ₌₁ⁿ g(Xᵢ)
```

where X₁, ..., Xₙ are samples from the distribution of X.

**Standard error:** SE = SD(g(X))/√n

### Importance Sampling
When direct sampling difficult:
```
E[g(X)] = ∫ g(x) f(x) dx = ∫ g(x) f(x)/h(x) h(x) dx
```

Sample from h(x) and weight by f(x)/h(x).

## 4.17 Decision Theory Applications

### Loss Functions
Expected loss for action a when θ is true:
```
R(θ, a) = E[L(θ, a)]
```

### Utility Theory
Expected utility maximization:
```
max_a E[U(a, X)]
```

### Risk and Return
In finance:
- Expected return: E[R]
- Risk: Var(R) or SD(R)
- Sharpe ratio: (E[R] - r_f)/SD(R)

## Key Insights

1. **Linearity:** Expectation is a linear operator, which makes calculations tractable.

2. **Independence:** Expectation of products equals product of expectations for independent variables.

3. **Law of Total Expectation:** Powerful tool for computing expectations by conditioning.

4. **Variance Decomposition:** Understanding sources of variability through conditional variance.

5. **Moment Generating Functions:** Encode all moments and uniquely determine distributions.

## Common Pitfalls

1. **E[1/X] ≠ 1/E[X]:** Expectation of reciprocal is not reciprocal of expectation.

2. **Independence assumption:** E[XY] = E[X]E[Y] only when X and Y are independent.

3. **Existence:** Not all random variables have finite expectation.

4. **Jensen's inequality direction:** Must know if function is convex or concave.

5. **Conditional vs unconditional:** E[X|Y] is random, E[X] is constant.

## Connections to Other Chapters

### To Chapter 3 (Random Variables)
- Computing expectations for specific distributions
- Moment generating functions
- Transformations and LOTUS

### To Chapter 5 (Inequalities)  
- Markov and Chebyshev inequalities
- Concentration bounds
- Probabilistic inequalities

### To Chapter 6 (Convergence)
- Law of large numbers
- Central limit theorem
- Convergence of expectations

### To Chapter 14 (Regression)
- Conditional expectations as regression functions
- Best linear predictors
- Method of moments estimation

This chapter establishes the foundation for summarizing and analyzing probability distributions, which is essential for all subsequent statistical methodology.