# Chapter 1: Introduction and Review - Mathematical Explanations

## Overview
This introductory chapter establishes the mathematical foundations needed for statistical inference, reviews key concepts from probability theory, and provides the framework for understanding statistical methodology. It serves as a bridge between pure mathematics and applied statistics.

## 1.1 What is Statistics?

### Definition
**Statistics** is the science of collecting, organizing, analyzing, and interpreting data to make decisions or draw conclusions about populations based on sample information.

### Core Components
1. **Descriptive Statistics:** Summarizing and describing data
2. **Inferential Statistics:** Making conclusions about populations from samples
3. **Statistical Modeling:** Mathematical representation of real-world phenomena
4. **Decision Theory:** Framework for optimal decision-making under uncertainty

### Statistical vs. Mathematical Thinking
- **Mathematics:** Deals with exact relationships and proofs
- **Statistics:** Deals with uncertainty, variability, and probabilistic statements
- **Key insight:** Perfect prediction is impossible; we quantify uncertainty instead

## 1.2 Populations and Samples

### Population
The **population** is the complete collection of all individuals or objects of interest.

**Examples:**
- All voters in a country
- All light bulbs produced by a factory
- All possible outcomes of an experiment

### Sample
A **sample** is a subset of the population that we actually observe.

**Notation:**
- Population size: N (often infinite)
- Sample size: n
- Population parameter: θ
- Sample statistic: θ̂ or T

### Sampling Methods

**Simple Random Sampling:** Each subset of size n has equal probability of selection
```
P(selecting any specific sample) = 1/(N choose n)
```

**Stratified Sampling:** Divide population into strata, sample from each
**Systematic Sampling:** Select every k-th element
**Cluster Sampling:** Sample entire clusters/groups

## 1.3 Statistical Models

### Model Definition
A **statistical model** is a mathematical representation that describes how data are generated:
```
𝒫 = {P_θ : θ ∈ Θ}
```

where 𝒫 is the family of probability distributions indexed by parameter θ.

### Types of Models

**Parametric Models:** Finite-dimensional parameter space
- Example: X ~ N(μ, σ²) with θ = (μ, σ²)

**Nonparametric Models:** Infinite-dimensional parameter space  
- Example: X has continuous distribution F

**Semiparametric Models:** Mix of parametric and nonparametric components
- Example: Regression with unknown error distribution

### Model Selection Criteria
1. **Goodness of fit:** How well does model explain data?
2. **Parsimony:** Simpler models preferred (Occam's razor)
3. **Interpretability:** Can we understand the model?
4. **Predictive power:** How well does it predict new data?

## 1.4 Basic Set Theory and Probability

### Set Operations
**Union:** A ∪ B = {x : x ∈ A or x ∈ B}
**Intersection:** A ∩ B = {x : x ∈ A and x ∈ B}
**Complement:** A^c = {x : x ∉ A}
**Difference:** A \ B = A ∩ B^c

### De Morgan's Laws
```
(A ∪ B)^c = A^c ∩ B^c
(A ∩ B)^c = A^c ∪ B^c
```

### Probability Axioms (Kolmogorov)
1. **Non-negativity:** P(A) ≥ 0 for all events A
2. **Normalization:** P(Ω) = 1
3. **Countable additivity:** For disjoint events A₁, A₂, ...:
   ```
   P(⋃ᵢ Aᵢ) = ∑ᵢ P(Aᵢ)
   ```

### Conditional Probability
```
P(A|B) = P(A ∩ B)/P(B), provided P(B) > 0
```

**Bayes' Theorem:**
```
P(A|B) = P(B|A)P(A)/P(B)
```

## 1.5 Random Variables and Distributions

### Random Variable
A **random variable** X is a function mapping outcomes to real numbers:
```
X: Ω → ℝ
```

### Distribution Function
**Cumulative Distribution Function (CDF):**
```
F(x) = P(X ≤ x)
```

**Properties:**
- Non-decreasing: x ≤ y ⟹ F(x) ≤ F(y)
- Right-continuous: lim_{h↓0} F(x+h) = F(x)
- Limits: lim_{x→-∞} F(x) = 0, lim_{x→∞} F(x) = 1

### Probability Mass/Density Functions

**Discrete case (PMF):**
```
f(x) = P(X = x)
```

**Continuous case (PDF):**
```
f(x) = dF(x)/dx where F(x) = ∫_{-∞}^x f(t) dt
```

## 1.6 Common Distributions

### Discrete Distributions

**Bernoulli(p):**
- PMF: f(x) = p^x(1-p)^{1-x}, x ∈ {0,1}
- Mean: p, Variance: p(1-p)

**Binomial(n,p):**
- PMF: f(x) = (n choose x)p^x(1-p)^{n-x}
- Mean: np, Variance: np(1-p)

**Poisson(λ):**
- PMF: f(x) = e^{-λ}λ^x/x!, x = 0,1,2,...
- Mean: λ, Variance: λ

### Continuous Distributions

**Uniform(a,b):**
- PDF: f(x) = 1/(b-a), x ∈ [a,b]
- Mean: (a+b)/2, Variance: (b-a)²/12

**Normal(μ,σ²):**
- PDF: f(x) = (1/(σ√(2π)))exp(-(x-μ)²/(2σ²))
- Mean: μ, Variance: σ²

**Exponential(λ):**
- PDF: f(x) = λe^{-λx}, x > 0
- Mean: 1/λ, Variance: 1/λ²

## 1.7 Expectation and Variance

### Expectation
**Discrete:** E[X] = ∑ₓ x f(x)
**Continuous:** E[X] = ∫ x f(x) dx

### Properties of Expectation
1. **Linearity:** E[aX + bY] = aE[X] + bE[Y]
2. **Monotonicity:** X ≤ Y ⟹ E[X] ≤ E[Y]
3. **Constant:** E[c] = c

### Variance
```
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
```

### Properties of Variance
1. **Constant:** Var(c) = 0
2. **Scaling:** Var(aX) = a²Var(X)
3. **Independence:** Var(X + Y) = Var(X) + Var(Y) if X,Y independent

## 1.8 Joint Distributions

### Joint CDF
```
F(x,y) = P(X ≤ x, Y ≤ y)
```

### Marginal Distributions
```
F_X(x) = lim_{y→∞} F(x,y)
F_Y(y) = lim_{x→∞} F(x,y)
```

### Independence
X and Y are independent if:
```
F(x,y) = F_X(x)F_Y(y) for all x,y
```

**Equivalent conditions:**
- f(x,y) = f_X(x)f_Y(y)
- E[g(X)h(Y)] = E[g(X)]E[h(Y)]

### Covariance and Correlation
**Covariance:**
```
Cov(X,Y) = E[(X-E[X])(Y-E[Y])] = E[XY] - E[X]E[Y]
```

**Correlation:**
```
ρ(X,Y) = Cov(X,Y)/(SD(X)·SD(Y))
```

**Properties:** -1 ≤ ρ ≤ 1, ρ = 0 if independent

## 1.9 Transformations

### Functions of Random Variables
If Y = g(X), then for strictly monotonic g:
```
f_Y(y) = f_X(g^{-1}(y)) |dg^{-1}(y)/dy|
```

### Linear Transformations
If Y = aX + b:
- E[Y] = aE[X] + b
- Var(Y) = a²Var(X)
- If X ~ N(μ,σ²), then Y ~ N(aμ+b, a²σ²)

### Moment Generating Functions
```
M_X(t) = E[e^{tX}]
```

**Properties:**
- Uniquely determines distribution
- M_{aX+b}(t) = e^{bt}M_X(at)
- If X,Y independent: M_{X+Y}(t) = M_X(t)M_Y(t)

## 1.10 Limit Theorems

### Law of Large Numbers
For iid X₁,X₂,... with E[Xᵢ] = μ:
```
X̄ₙ = (1/n)∑ᵢ₌₁ⁿ Xᵢ →ᵖ μ
```

### Central Limit Theorem
For iid X₁,X₂,... with E[Xᵢ] = μ, Var(Xᵢ) = σ² < ∞:
```
√n(X̄ₙ - μ)/σ ⇝ N(0,1)
```

**Practical implication:** Sample means are approximately normal for large n.

## 1.11 Order Statistics

### Definition
For sample X₁,...,Xₙ, the **order statistics** are:
```
X₍₁₎ ≤ X₍₂₎ ≤ ... ≤ X₍ₙ₎
```

### Sample Quantiles
**Sample median:** X₍₍ₙ₊₁₎/₂₎ if n odd, (X₍ₙ/₂₎ + X₍ₙ/₂₊₁₎)/2 if n even

**Sample α-quantile:** X₍⌈nα⌉₎

### Empirical Distribution Function
```
F̂ₙ(x) = (1/n) ∑ᵢ₌₁ⁿ I(Xᵢ ≤ x)
```

## 1.12 Statistical Inference Framework

### Parameter Estimation
**Point estimation:** Single best guess for parameter
**Interval estimation:** Range of plausible values

### Hypothesis Testing
**Null hypothesis H₀:** Default assumption
**Alternative hypothesis H₁:** What we're testing for
**Test statistic:** Function of data used for decision
**p-value:** Probability of observing data as extreme under H₀

### Types of Errors
- **Type I error:** Reject true H₀ (false positive)
- **Type II error:** Fail to reject false H₀ (false negative)

## 1.13 Frequentist vs. Bayesian Approaches

### Frequentist Philosophy
- Parameters are fixed unknown constants
- Probability refers to long-run frequency
- Sample provides evidence about fixed parameter

### Bayesian Philosophy  
- Parameters are random variables
- Probability represents degree of belief
- Prior beliefs updated with data to get posterior

### Comparison
**Frequentist confidence interval:** In repeated sampling, 95% of intervals contain true parameter
**Bayesian credible interval:** Given observed data, 95% probability parameter is in interval

## 1.14 Computational Statistics

### Monte Carlo Methods
Use random sampling to approximate:
- Integrals: ∫g(x)f(x)dx ≈ (1/n)∑g(Xᵢ)
- Probabilities: P(X ∈ A) ≈ (1/n)∑I(Xᵢ ∈ A)

### Bootstrap
Resample from data to approximate sampling distribution:
1. Sample with replacement from observed data
2. Compute statistic on bootstrap sample
3. Repeat many times
4. Use empirical distribution of bootstrap statistics

### Cross-Validation
Assess model performance:
1. Split data into training and validation sets
2. Fit model on training data
3. Evaluate on validation data
4. Repeat with different splits

## 1.15 Data Types and Structures

### Types of Variables
**Categorical (Qualitative):**
- Nominal: No natural ordering (colors, brands)
- Ordinal: Natural ordering (grades, ratings)

**Numerical (Quantitative):**
- Discrete: Countable values (number of children)
- Continuous: Uncountable values (height, weight)

### Measurement Scales
**Nominal:** Classification only
**Ordinal:** Classification + ordering
**Interval:** Classification + ordering + equal intervals
**Ratio:** Classification + ordering + equal intervals + true zero

## 1.16 Exploratory Data Analysis

### Summary Statistics
**Center:** Mean, median, mode
**Spread:** Range, variance, standard deviation, IQR
**Shape:** Skewness, kurtosis

### Graphical Methods
**Univariate:** Histograms, box plots, density plots
**Bivariate:** Scatter plots, correlation plots
**Multivariate:** Pair plots, parallel coordinates

### Five-Number Summary
Minimum, Q₁, Median, Q₃, Maximum

**Box plot:** Visual representation of five-number summary

## 1.17 Probability Inequalities (Preview)

### Markov's Inequality
For non-negative X and a > 0:
```
P(X ≥ a) ≤ E[X]/a
```

### Chebyshev's Inequality
For any X with finite variance:
```
P(|X - E[X]| ≥ k) ≤ Var(X)/k²
```

**Interpretation:** At least 75% of distribution within 2 standard deviations

## 1.18 Linear Algebra Review

### Vectors and Matrices
**Vector:** x = (x₁,...,xₙ)ᵀ
**Matrix:** A = [aᵢⱼ] with i rows, j columns

### Matrix Operations
**Addition:** (A + B)ᵢⱼ = aᵢⱼ + bᵢⱼ
**Multiplication:** (AB)ᵢⱼ = ∑ₖ aᵢₖbₖⱼ
**Transpose:** (Aᵀ)ᵢⱼ = aⱼᵢ

### Special Matrices
**Identity:** I with ones on diagonal, zeros elsewhere
**Symmetric:** A = Aᵀ
**Positive definite:** xᵀAx > 0 for all x ≠ 0

## 1.19 Calculus Review

### Derivatives
**Basic rules:** Power, product, quotient, chain rules
**Partial derivatives:** ∂f/∂x for multivariable functions
**Gradient:** ∇f = (∂f/∂x₁,...,∂f/∂xₙ)ᵀ

### Taylor Series
```
f(x) ≈ f(a) + f'(a)(x-a) + (1/2)f''(a)(x-a)² + ...
```

### Integration
**Fundamental theorem:** ∫ₐᵇ f'(x)dx = f(b) - f(a)
**Integration by parts:** ∫udv = uv - ∫vdu

## 1.20 Key Statistical Concepts

### Bias and Variance
**Bias:** E[θ̂] - θ (systematic error)
**Variance:** Var(θ̂) (random error)
**MSE:** Bias² + Variance

### Consistency
Estimator θ̂ₙ is consistent if θ̂ₙ →ᵖ θ

### Efficiency
Estimator achieving Cramér-Rao lower bound

### Sufficiency
Statistic contains all information about parameter

### Completeness
Family of distributions such that E[g(T)] = 0 ∀θ implies g(T) = 0

## Road Map for the Course

### Part I: Probability Foundation (Chapters 2-6)
- Basic probability and random variables
- Expectation, variance, and moment generating functions
- Important inequalities and limit theorems
- Convergence concepts

### Part II: Statistical Inference (Chapters 7-12)
- Estimation theory and methods
- Confidence intervals and hypothesis testing
- Bootstrap and nonparametric methods
- Bayesian inference

### Part III: Statistical Models (Chapters 13-23)
- Decision theory and linear regression
- Advanced topics in modern statistics
- Classification and machine learning connections

## Key Insights

1. **Uncertainty Quantification:** Statistics provides tools to measure and communicate uncertainty.

2. **Sample to Population:** We use samples to make inferences about populations.

3. **Model-Based Thinking:** Statistical models help us understand and predict phenomena.

4. **Trade-offs:** Bias vs. variance, simplicity vs. complexity, interpretability vs. accuracy.

5. **Computational Revolution:** Modern statistics relies heavily on computational methods.

## Common Misconceptions

1. **Correlation implies causation:** Strong correlation doesn't prove causal relationship
2. **Larger samples always better:** Quality matters more than quantity
3. **Statistical significance = practical importance:** p < 0.05 doesn't mean effect is large
4. **Confidence interval interpretation:** 95% CI doesn't mean 95% chance parameter is in interval
5. **Normal distribution universality:** Not all data is normally distributed

## Prerequisites and Preparation

### Mathematical Background
- Single and multivariable calculus
- Linear algebra (vectors, matrices, eigenvalues)
- Basic proof techniques
- Set theory and logic

### Programming Skills
- Statistical software (R, Python, SAS)
- Data manipulation and visualization
- Simulation and computational methods

### Statistical Thinking
- Comfort with uncertainty and variability
- Understanding of logical reasoning
- Appreciation for careful methodology

## Connections to Other Fields

### Computer Science
- Machine learning algorithms
- Data structures and algorithms
- Computational complexity

### Mathematics
- Measure theory and real analysis
- Optimization theory
- Numerical analysis

### Applied Fields
- Economics (econometrics)
- Biology (biostatistics)
- Engineering (quality control)
- Psychology (psychometrics)

This introductory chapter sets the stage for the entire course, providing both the mathematical foundations and the conceptual framework needed to understand modern statistical inference.