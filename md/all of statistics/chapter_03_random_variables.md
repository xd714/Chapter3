# Chapter 3: Random Variables - Mathematical Explanations

## Overview
Random variables provide a mathematical framework for dealing with uncertainty by mapping outcomes of random experiments to real numbers. This chapter covers discrete and continuous random variables, probability distributions, and fundamental distribution families.

## 3.1 Introduction to Random Variables

### Definition
A **random variable** X is a function that assigns a real number X(ω) to each outcome ω in the sample space:
```
X: Ω → ℝ
```

**Key Point:** X is not actually "random" - it's a deterministic function. The randomness comes from the underlying probability space.

### Types of Random Variables
- **Discrete**: X takes countable values (finite or countably infinite)
- **Continuous**: X takes uncountable values (typically intervals of real numbers)

## 3.2 Discrete Random Variables

### Probability Mass Function (PMF)
For discrete random variable X, the **probability mass function** is:
```
f_X(x) = P(X = x)
```

**Properties:**
1. f_X(x) ≥ 0 for all x
2. ∑_x f_X(x) = 1 (sum over all possible values)
3. f_X(x) = 0 if x is not a possible value of X

### Cumulative Distribution Function (CDF)
The **cumulative distribution function** is:
```
F_X(x) = P(X ≤ x) = ∑_{t≤x} f_X(t)
```

**Properties:**
1. F_X is non-decreasing
2. lim_{x→-∞} F_X(x) = 0
3. lim_{x→∞} F_X(x) = 1
4. F_X is right-continuous
5. P(a < X ≤ b) = F_X(b) - F_X(a)

## 3.3 Continuous Random Variables

### Probability Density Function (PDF)
For continuous random variable X, the **probability density function** f_X(x) satisfies:
```
P(a ≤ X ≤ b) = ∫_a^b f_X(x) dx
```

**Properties:**
1. f_X(x) ≥ 0 for all x
2. ∫_{-∞}^∞ f_X(x) dx = 1
3. P(X = x) = 0 for any specific value x

### Relationship Between PDF and CDF
```
F_X(x) = ∫_{-∞}^x f_X(t) dt
f_X(x) = d/dx F_X(x) (where F_X is differentiable)
```

## 3.4 Important Discrete Distributions

### Bernoulli Distribution
X ~ Bernoulli(p), where 0 ≤ p ≤ 1

**PMF:**
```
f_X(x) = {
  p     if x = 1
  1-p   if x = 0
  0     otherwise
}
```

**Interpretation:** Models a single trial with success probability p.

### Binomial Distribution
X ~ Binomial(n, p), where n ∈ ℕ and 0 ≤ p ≤ 1

**PMF:**
```
f_X(x) = (n choose x) p^x (1-p)^{n-x}, x = 0, 1, ..., n
```

**Interpretation:** Number of successes in n independent Bernoulli(p) trials.

**Properties:**
- If X₁, ..., Xₙ are independent Bernoulli(p), then ∑ᵢ Xᵢ ~ Binomial(n, p)
- Sum property: If X ~ Binomial(n₁, p) and Y ~ Binomial(n₂, p) independently, then X + Y ~ Binomial(n₁ + n₂, p)

### Geometric Distribution
X ~ Geometric(p), where 0 < p ≤ 1

**PMF:**
```
f_X(x) = (1-p)^{x-1} p, x = 1, 2, 3, ...
```

**Interpretation:** Number of trials until first success in independent Bernoulli(p) trials.

**Memoryless Property:**
```
P(X > n + m | X > n) = P(X > m)
```

### Poisson Distribution
X ~ Poisson(λ), where λ > 0

**PMF:**
```
f_X(x) = e^{-λ} λ^x / x!, x = 0, 1, 2, ...
```

**Interpretation:** 
- Number of events in fixed time interval with rate λ
- Limit of Binomial(n, p) as n → ∞, p → 0, np → λ

**Properties:**
- Sum property: If X ~ Poisson(λ₁) and Y ~ Poisson(λ₂) independently, then X + Y ~ Poisson(λ₁ + λ₂)

## 3.5 Important Continuous Distributions

### Uniform Distribution
X ~ Uniform(a, b), where a < b

**PDF:**
```
f_X(x) = {
  1/(b-a)  if a ≤ x ≤ b
  0        otherwise
}
```

**CDF:**
```
F_X(x) = {
  0           if x < a
  (x-a)/(b-a) if a ≤ x ≤ b
  1           if x > b
}
```

### Normal (Gaussian) Distribution
X ~ N(μ, σ²), where μ ∈ ℝ and σ > 0

**PDF:**
```
f_X(x) = (1/(σ√(2π))) exp{-1/2 ((x-μ)/σ)²}
```

**Standard Normal:** Z ~ N(0, 1)
```
φ(z) = (1/√(2π)) exp(-z²/2)
Φ(z) = P(Z ≤ z) = ∫_{-∞}^z φ(t) dt
```

**Standardization:** If X ~ N(μ, σ²), then Z = (X - μ)/σ ~ N(0, 1)

**Properties:**
- Symmetric about μ
- Bell-shaped curve
- Approximately 68% within μ ± σ, 95% within μ ± 2σ, 99.7% within μ ± 3σ

### Exponential Distribution
X ~ Exponential(β), where β > 0

**PDF:**
```
f_X(x) = (1/β) e^{-x/β}, x > 0
```

**CDF:**
```
F_X(x) = 1 - e^{-x/β}, x > 0
```

**Memoryless Property:**
```
P(X > s + t | X > s) = P(X > t)
```

**Interpretation:** Waiting times between events in a Poisson process.

### Gamma Distribution
X ~ Gamma(α, β), where α > 0, β > 0

**PDF:**
```
f_X(x) = (1/(β^α Γ(α))) x^{α-1} e^{-x/β}, x > 0
```

Where Γ(α) = ∫₀^∞ y^{α-1} e^{-y} dy is the Gamma function.

**Properties:**
- Γ(α) = (α-1)! when α is a positive integer
- Γ(α) = (α-1)Γ(α-1)
- Exponential(β) = Gamma(1, β)

**Sum Property:** If X ~ Gamma(α₁, β) and Y ~ Gamma(α₂, β) independently, then X + Y ~ Gamma(α₁ + α₂, β)

### Beta Distribution
X ~ Beta(α, β), where α > 0, β > 0

**PDF:**
```
f_X(x) = (Γ(α+β)/(Γ(α)Γ(β))) x^{α-1} (1-x)^{β-1}, 0 < x < 1
```

**Properties:**
- Defined on interval (0, 1)
- Uniform(0, 1) = Beta(1, 1)
- Very flexible family of distributions on [0, 1]

## 3.6 Transformations of Random Variables

### General Method
If Y = g(X) where g is a function and X has PDF f_X(x):

**For strictly monotonic g:**
```
f_Y(y) = f_X(g^{-1}(y)) |d/dy g^{-1}(y)|
```

### Common Transformations

**Linear Transformation:** If Y = aX + b, then:
```
f_Y(y) = (1/|a|) f_X((y-b)/a)
```

**Logarithmic:** If Y = ln(X) and X > 0, then:
```
f_Y(y) = f_X(e^y) e^y
```

**Exponential:** If Y = e^X, then:
```
f_Y(y) = f_X(ln(y)) (1/y), y > 0
```

## 3.7 Joint Distributions

### Joint PMF (Discrete)
For discrete random variables X and Y:
```
f_{X,Y}(x,y) = P(X = x, Y = y)
```

### Joint PDF (Continuous)
For continuous random variables X and Y:
```
P((X,Y) ∈ A) = ∬_A f_{X,Y}(x,y) dx dy
```

### Marginal Distributions
**Discrete:**
```
f_X(x) = ∑_y f_{X,Y}(x,y)
f_Y(y) = ∑_x f_{X,Y}(x,y)
```

**Continuous:**
```
f_X(x) = ∫ f_{X,Y}(x,y) dy
f_Y(y) = ∫ f_{X,Y}(x,y) dx
```

### Conditional Distributions
```
f_{X|Y}(x|y) = f_{X,Y}(x,y) / f_Y(y)
```

**Interpretation:** Distribution of X given that Y = y.

### Independence
X and Y are independent if and only if:
```
f_{X,Y}(x,y) = f_X(x) f_Y(y)
```

**Equivalent condition:** f_{X|Y}(x|y) = f_X(x) for all y where f_Y(y) > 0.

## 3.8 Multivariate Normal Distribution

### Bivariate Normal
(X, Y) ~ N(μ₁, μ₂, σ₁², σ₂², ρ) where -1 < ρ < 1

**PDF:**
```
f(x,y) = 1/(2πσ₁σ₂√(1-ρ²)) exp{-1/(2(1-ρ²)) [(x-μ₁)²/σ₁² - 2ρ(x-μ₁)(y-μ₂)/(σ₁σ₂) + (y-μ₂)²/σ₂²]}
```

**Properties:**
- Marginals: X ~ N(μ₁, σ₁²), Y ~ N(μ₂, σ₂²)
- Independence ⟺ ρ = 0
- Linear combinations are normal

### General Multivariate Normal
X ~ N(μ, Σ) where μ is the mean vector and Σ is the covariance matrix

**PDF:**
```
f(x) = 1/((2π)^{k/2} |Σ|^{1/2}) exp{-1/2 (x-μ)ᵀ Σ^{-1} (x-μ)}
```

## 3.9 Order Statistics

### Definition
Given X₁, ..., Xₙ, the **order statistics** are:
```
X₍₁₎ ≤ X₍₂₎ ≤ ... ≤ X₍ₙ₎
```

### Distribution of Order Statistics
For continuous X with CDF F and PDF f:

**Minimum:** X₍₁₎
```
F₁(x) = 1 - [1 - F(x)]ⁿ
f₁(x) = n[1 - F(x)]^{n-1} f(x)
```

**Maximum:** X₍ₙ₎
```
Fₙ(x) = [F(x)]ⁿ
fₙ(x) = n[F(x)]^{n-1} f(x)
```

**k-th order statistic:** X₍ₖ₎
```
fₖ(x) = n!/(k-1)!(n-k)! [F(x)]^{k-1} [1-F(x)]^{n-k} f(x)
```

## 3.10 Key Properties and Theorems

### Probability Integral Transformation
If X is continuous with CDF F, then F(X) ~ Uniform(0, 1).

**Inverse Transform Method:** If U ~ Uniform(0, 1) and F is a CDF, then F^(-1)(U) has CDF F.

### Convergence in Distribution
A sequence of random variables X₁, X₂, ... converges in distribution to X if:
```
lim_{n→∞} F_n(x) = F(x)
```
at all continuity points of F.

## 3.11 Moment Generating Functions

### Definition
The **moment generating function** (MGF) of X is:
```
M_X(t) = E[e^{tX}]
```
when this expectation exists for t in some neighborhood of 0.

### Properties
1. **Uniqueness:** If M_X(t) = M_Y(t) in a neighborhood of 0, then X and Y have the same distribution
2. **Moments:** If M_X(t) exists, then E[X^n] = M_X^{(n)}(0)
3. **Independence:** If X and Y are independent, then M_{X+Y}(t) = M_X(t)M_Y(t)

### Common MGFs
- **Bernoulli(p):** M(t) = 1 - p + pe^t
- **Binomial(n,p):** M(t) = (1 - p + pe^t)^n
- **Poisson(λ):** M(t) = e^{λ(e^t - 1)}
- **Normal(μ,σ²):** M(t) = e^{μt + σ²t²/2}
- **Exponential(β):** M(t) = 1/(1 - βt) for t < 1/β

## 3.12 Inequalities and Bounds

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

### Chernoff Bound
For any random variable X and a > 0:
```
P(X ≥ a) ≤ inf_{t>0} e^{-ta} M_X(t)
```

## 3.13 Simulation and Random Number Generation

### Inverse Transform Method
1. Generate U ~ Uniform(0, 1)
2. Return X = F^(-1)(U)

### Acceptance-Rejection Method
To generate from f(x):
1. Find g(x) easy to sample from and c such that f(x) ≤ cg(x)
2. Generate Y from g and U ~ Uniform(0, 1)
3. If U ≤ f(Y)/(cg(Y)), return Y; otherwise repeat

### Box-Muller Transform
To generate normal random variables:
1. Generate U₁, U₂ ~ Uniform(0, 1)
2. Set Z₁ = √(-2 ln U₁) cos(2πU₂)
3. Set Z₂ = √(-2 ln U₁) sin(2πU₂)
4. Then Z₁, Z₂ ~ N(0, 1) independently

## 3.14 Applications and Examples

### Reliability Engineering
- **Exponential distribution:** Models time between failures for memoryless systems
- **Weibull distribution:** Models time to failure with increasing/decreasing hazard rates
- **Gamma distribution:** Models time until k-th failure

### Quality Control
- **Normal distribution:** Natural variation in manufacturing processes
- **Control charts:** Using ±3σ limits based on normal distribution properties

### Finance
- **Log-normal distribution:** Stock prices (geometric Brownian motion)
- **Heavy-tailed distributions:** Extreme value modeling for risk management

### Queueing Theory
- **Poisson process:** Arrival times in service systems
- **Exponential distribution:** Service times in many queueing models

## 3.15 Connections to Other Chapters

### To Chapter 4 (Expectation)
- Expected value and variance definitions
- Moment generating functions
- Law of large numbers foundations

### To Chapter 5 (Inequalities)
- Markov and Chebyshev inequalities
- Concentration inequalities
- Tail bounds

### To Chapter 6 (Convergence)
- Convergence in distribution
- Central limit theorem setup
- Weak law of large numbers

## Key Insights

1. **Distribution Families:** Understanding the relationships between different distribution families (e.g., exponential as special case of gamma) provides insight into their properties.

2. **Transformations:** Many complex distributions can be understood as transformations of simpler ones.

3. **Limiting Behavior:** Many distributions arise as limits of others (e.g., Poisson as limit of binomial, normal as limit via CLT).

4. **Simulation:** Understanding the mathematical properties enables efficient simulation methods.

## Common Pitfalls

1. **Confusing independence and uncorrelatedness** for non-normal distributions
2. **Misunderstanding the role of parameters** in different parameterizations
3. **Incorrectly applying transformations** without accounting for Jacobians
4. **Assuming normality** without justification

## Exercises and Problem Types

### Typical Problems Include:
1. **Distribution identification:** Recognizing which distribution models a given scenario
2. **Parameter estimation:** Finding parameters from given information
3. **Probability calculations:** Using PDFs, CDFs, and properties
4. **Transformations:** Finding distributions of functions of random variables
5. **Simulation:** Implementing random number generation algorithms

This chapter provides the essential foundation for all statistical inference, as it establishes how we model uncertainty mathematically through probability distributions.