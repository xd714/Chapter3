# Chapter 5: Inequalities - Mathematical Explanations

## Overview
Probability inequalities are fundamental tools in probability theory and statistics. They provide bounds on probabilities and expectations, enable concentration results, and form the theoretical foundation for statistical inference. This chapter covers the most important inequalities and their applications.

## 5.1 Basic Probability Inequalities

### Union Bound (Boole's Inequality)
For any collection of events A₁, A₂, ..., Aₙ:
```
P(⋃ᵢ₌₁ⁿ Aᵢ) ≤ ∑ᵢ₌₁ⁿ P(Aᵢ)
```

**Infinite version:**
```
P(⋃ᵢ₌₁^∞ Aᵢ) ≤ ∑ᵢ₌₁^∞ P(Aᵢ)
```

**Applications:**
- Multiple testing corrections
- Controlling family-wise error rates
- Large deviation theory

### Bonferroni Inequality
```
P(⋂ᵢ₌₁ⁿ Aᵢ) ≥ 1 - ∑ᵢ₌₁ⁿ P(Aᵢᶜ)
```

**Interpretation:** Probability that all events occur is at least 1 minus the sum of probabilities that each fails.

### Inclusion-Exclusion Bounds
For events A₁, ..., Aₙ, define:
```
S₁ = ∑ᵢ P(Aᵢ)
S₂ = ∑ᵢ<ⱼ P(Aᵢ ∩ Aⱼ)
⋮
Sₙ = P(A₁ ∩ ... ∩ Aₙ)
```

**Bonferroni inequalities:**
```
S₁ - S₂ ≤ P(⋃ᵢ Aᵢ) ≤ S₁
S₁ - S₂ + S₃ ≤ P(⋃ᵢ Aᵢ) ≤ S₁ - S₂ + S₃
```

## 5.2 Markov's Inequality

### Statement
For non-negative random variable X and a > 0:
```
P(X ≥ a) ≤ E[X]/a
```

### Proof
Let A = {X ≥ a}. Then:
```
E[X] = ∫ x dP ≥ ∫_A x dP ≥ ∫_A a dP = a·P(A) = a·P(X ≥ a)
```

### Generalizations

**For any non-negative function g:**
If g is non-negative and non-decreasing on [0,∞), then:
```
P(X ≥ a) ≤ E[g(X)]/g(a)
```

**One-sided version:**
```
P(X ≥ a) ≤ E[X]/a   (for a > 0)
P(X ≤ a) ≤ E[-X]/(-a) (for a < 0)
```

### Applications

**Tail bounds:** Crude but general bounds on tail probabilities

**Cantelli's inequality:** More refined one-sided bound using variance

**Chernoff bounds:** Apply with exponential functions

## 5.3 Chebyshev's Inequality

### Standard Form
For random variable X with finite mean μ and variance σ², and any k > 0:
```
P(|X - μ| ≥ k) ≤ σ²/k²
```

### Alternative Forms

**In terms of standard deviations:**
```
P(|X - μ| ≥ kσ) ≤ 1/k²
```

**Two-sided bound:**
```
P(|X - μ| < kσ) ≥ 1 - 1/k²
```

**Practical interpretation:** At least 75% of distribution lies within 2 standard deviations, at least 89% within 3 standard deviations.

### Proof
Apply Markov's inequality to (X - μ)²:
```
P(|X - μ| ≥ k) = P((X - μ)² ≥ k²) ≤ E[(X - μ)²]/k² = σ²/k²
```

### One-sided Chebyshev (Cantelli's Inequality)
```
P(X - μ ≥ k) ≤ σ²/(σ² + k²)
P(X - μ ≤ -k) ≤ σ²/(σ² + k²)
```

**Better bound:** Tighter than two-sided Chebyshev when only one tail matters.

## 5.4 Chernoff Bounds

### Method
For any random variable X and real number a:
```
P(X ≥ a) = P(e^{tX} ≥ e^{ta}) ≤ E[e^{tX}]/e^{ta} = M_X(t)e^{-ta}
```

**Optimization:** Choose t to minimize the bound:
```
P(X ≥ a) ≤ \inf_{t>0} M_X(t)e^{-ta}
```

### Chernoff Bound for Sum of Independent Variables
For X₁, ..., Xₙ independent:
```
M_{S_n}(t) = ∏ᵢ₌₁ⁿ M_{X_i}(t)
```

### Hoeffding's Inequality
For X₁, ..., Xₙ independent with Xᵢ ∈ [aᵢ, bᵢ]:
```
P(S_n - E[S_n] ≥ t) ≤ \exp\left(-\frac{2t²}{\sum_{i=1}^n (b_i - a_i)²}\right)
```

**Symmetric bound:**
```
P(|S_n - E[S_n]| ≥ t) ≤ 2\exp\left(-\frac{2t²}{\sum_{i=1}^n (b_i - a_i)²}\right)
```

### Bennett's Inequality
For X₁, ..., Xₙ independent with E[Xᵢ] = 0, |Xᵢ| ≤ M, and variance σᵢ²:
```
P(S_n ≥ t) ≤ \exp\left(-\frac{v}{M²}h\left(\frac{Mt}{v}\right)\right)
```

where v = ∑σᵢ² and h(u) = (1+u)log(1+u) - u.

### Bernstein's Inequality
**Simplified version of Bennett:** For the same setup:
```
P(S_n ≥ t) ≤ \exp\left(-\frac{t²}{2(v + Mt/3)}\right)
```

## 5.5 Concentration Inequalities

### Sub-Gaussian Random Variables
X is **sub-Gaussian** with parameter σ if:
```
E[e^{t(X - E[X])}] ≤ e^{σ²t²/2}
```

**Examples:**
- Gaussian random variables
- Bounded random variables (Hoeffding's lemma)
- Rademacher random variables

### Sub-Gaussian Concentration
For sub-Gaussian X₁, ..., Xₙ with parameter σ:
```
P(|S_n - E[S_n]| ≥ t) ≤ 2e^{-t²/(2nσ²)}
```

### Sub-Exponential Random Variables
X is **sub-exponential** with parameters (ν, α) if:
```
E[e^{t(X - E[X])}] ≤ e^{ν²t²/2}
```

for |t| ≤ 1/α.

### Concentration for Lipschitz Functions
If f is L-Lipschitz and X₁, ..., Xₙ are independent sub-Gaussian:
```
P(|f(X₁,...,Xₙ) - E[f(X₁,...,Xₙ)]| ≥ t) ≤ 2e^{-ct²/L²}
```

## 5.6 Martingale Inequalities

### Azuma's Inequality
For martingale M₀, M₁, ..., Mₙ with bounded differences |Mᵢ - Mᵢ₋₁| ≤ cᵢ:
```
P(|M_n - M_0| ≥ t) ≤ 2\exp\left(-\frac{t²}{2\sum_{i=1}^n c_i²}\right)
```

### McDiarmid's Inequality
For function f(X₁,...,Xₙ) with bounded differences property:
|f(x) - f(x')| ≤ cᵢ when x and x' differ only in the i-th coordinate.

Then:
```
P(|f(X₁,...,Xₙ) - E[f(X₁,...,Xₙ)]| ≥ t) ≤ 2\exp\left(-\frac{2t²}{\sum_{i=1}^n c_i²}\right)
```

### Doob's Inequality
For non-negative submartingale M₀, M₁, ..., Mₙ:
```
P(\max_{0≤k≤n} M_k ≥ λ) ≤ E[M_n]/λ
```

**Maximal inequality:** Controls the maximum over all time points.

## 5.7 Moment Inequalities

### Jensen's Inequality
For convex function φ:
```
φ(E[X]) ≤ E[φ(X)]
```

For concave function φ:
```
φ(E[X]) ≥ E[φ(X)]
```

### Hölder's Inequality
For p, q > 1 with 1/p + 1/q = 1:
```
E[|XY|] ≤ (E[|X|^p])^{1/p} (E[|Y|^q])^{1/q}
```

**Special case (p = q = 2):** Cauchy-Schwarz inequality:
```
E[|XY|] ≤ \sqrt{E[X²]E[Y²]}
```

### Minkowski's Inequality
For p ≥ 1:
```
(E[|X + Y|^p])^{1/p} ≤ (E[|X|^p])^{1/p} + (E[|Y|^p])^{1/p}
```

**Interpretation:** Lᵖ norm satisfies triangle inequality.

### Lyapunov's Inequality
For 0 < r < s:
```
(E[|X|^r])^{1/r} ≥ (E[|X|^s])^{1/s}
```

## 5.8 Tail Bounds for Specific Distributions

### Normal Distribution
For X ~ N(0,1):
```
P(X ≥ t) ≤ \frac{1}{t\sqrt{2π}} e^{-t²/2}   (for t > 0)
```

**Mills' ratio:** More precise bounds using Q-function.

### Chi-square Distribution
For X ~ χ²_k:
```
P(X ≥ k + 2√(kt) + 2t) ≤ e^{-t}
P(X ≤ k - 2√(kt)) ≤ e^{-t}   (for t ≤ k/2)
```

### Binomial Distribution
For X ~ Binomial(n, p):

**Chernoff bound:**
```
P(X ≥ (1+δ)np) ≤ e^{-D(δ)np}
```

where D(δ) is the relative entropy.

**Hoeffding bound:**
```
P(|X - np| ≥ t) ≤ 2e^{-2t²/n}
```

### Poisson Distribution
For X ~ Poisson(λ):
```
P(X ≥ λ + t) ≤ e^{-t²/(2(λ+t/3))}   (Bernstein)
P(X ≥ (1+δ)λ) ≤ e^{-D(δ)λ}         (Chernoff)
```

## 5.9 Maximal Inequalities

### Kolmogorov's Inequality
For independent X₁, ..., Xₙ with E[Xᵢ] = 0 and finite variances:
```
P(\max_{1≤k≤n} |S_k| ≥ t) ≤ Var(S_n)/t²
```

where S_k = X₁ + ... + X_k.

### Doob's Maximal Inequality (L^p version)
For submartingale M_n with p > 1:
```
E[\max_{0≤k≤n} M_k^p] ≤ \left(\frac{p}{p-1}\right)^p E[M_n^p]
```

### Burkholder-Davis-Gundy Inequality
For martingale M_n with quadratic variation [M]_n:
```
c_p E[[M]_n^{p/2}] ≤ E[\max_{0≤k≤n} |M_k|^p] ≤ C_p E[[M]_n^{p/2}]
```

for universal constants c_p, C_p.

## 5.10 Empirical Process Inequalities

### Glivenko-Cantelli and Dvoretzky-Kiefer-Wolfowitz
For empirical distribution F̂_n:

**DKW inequality:**
```
P(\sup_x |F̂_n(x) - F(x)| > ε) ≤ 2e^{-2nε²}
```

**Massart's improvement:**
```
P(\sup_x |F̂_n(x) - F(x)| > ε) ≤ 2e^{-2nε² + 2ε}
```

### Rademacher Complexity
For function class ℱ and Rademacher variables σᵢ:
```
R_n(ℱ) = E[\sup_{f∈ℱ} \frac{1}{n}\sum_{i=1}^n σᵢf(Xᵢ)]
```

**Rademacher bound:**
```
E[\sup_{f∈ℱ} |P_n f - P f|] ≤ 2R_n(ℱ)
```

where P_n f = (1/n)∑f(Xᵢ) and P f = E[f(X)].

### VC Theory
For function class with VC dimension d:
```
R_n(ℱ) ≤ C\sqrt{\frac{d \log n}{n}}
```

## 5.11 Information-Theoretic Inequalities

### Entropy and Mutual Information
**Jensen's inequality for entropy:**
```
H(X) ≤ \log |\mathcal{X}|
```

**Data processing inequality:**
```
I(X; Y) ≥ I(X; Z)
```

when X → Y → Z forms a Markov chain.

### Fano's Inequality
For estimating parameter θ from data X:
```
H(θ|X) ≤ h_b(P_e) + P_e \log(|\Theta| - 1)
```

where P_e = P(θ̂ ≠ θ) and h_b is binary entropy.

**Application:** Lower bounds on estimation error.

## 5.12 Isoperimetric Inequalities

### Gaussian Isoperimetric Inequality
For measurable set A in Gaussian space:
```
Φ^{-1}(P(A^ε)) - Φ^{-1}(P(A)) ≥ ε
```

where A^ε is the ε-enlargement of A.

### Concentration of Lipschitz Functions
For Gaussian vector X and L-Lipschitz function f:
```
P(|f(X) - E[f(X)]| ≥ t) ≤ 2e^{-t²/(2L²)}
```

### Logarithmic Sobolev Inequalities
Control concentration through functional inequalities involving entropy and energy.

## 5.13 Matrix Concentration Inequalities

### Matrix Chernoff Bound
For independent random matrices X₁, ..., X_n:
```
P(λ_{\max}(\sum X_i - E[\sum X_i]) ≥ t) ≤ d \cdot e^{-t²/(2σ²)}
```

under appropriate conditions.

### Matrix Bernstein
For sum of independent bounded random matrices:
```
P(||S_n - E[S_n]|| ≥ t) ≤ (d₁ + d₂) \exp\left(-\frac{t²}{2(σ² + Rt/3)}\right)
```

### Applications
- Random matrix theory
- Covariance estimation
- Principal component analysis

## 5.14 Non-commutative Inequalities

### Noncommutative Hölder
For matrices A, B and p, q with 1/p + 1/q = 1:
```
||AB||₁ ≤ ||A||_p ||B||_q
```

### Golden-Thompson Inequality
For Hermitian matrices A, B:
```
\text{tr}(e^{A+B}) ≤ \text{tr}(e^A e^B)
```

## 5.15 Applications in Statistics

### Confidence Intervals
**Hoeffding's inequality** gives:
```
P(|X̄ - μ| ≤ ε) ≥ 1 - 2e^{-2nε²/(b-a)²}
```

for X̄ from [a,b]-valued variables.

### Hypothesis Testing
**Chernoff bounds** provide exponential error rates:
```
P(\text{Type I error}) ≤ e^{-nI₀}
P(\text{Type II error}) ≤ e^{-nI₁}
```

### High-dimensional Statistics
**Concentration inequalities** control:
- Sample covariance matrices
- Empirical risk minimization
- Sparse recovery (Lasso)

### Machine Learning
**Generalization bounds:**
```
P(R(h) - R̂(h) ≥ ε) ≤ δ
```

where R(h) is true risk, R̂(h) is empirical risk.

## 5.16 Minimax Theory Applications

### Lower Bounds via Fano's Inequality
For parameter estimation:
```
\inf_{\hat{θ}} \sup_{θ} E[d(θ̂, θ)] ≥ \text{function of} \max_{θ,θ'} I(P_θ; P_{θ'})
```

### Le Cam's Method
Two-point testing problems provide lower bounds:
```
\inf_T \sup_{i=0,1} P_i(T ≠ i) ≥ \frac{1}{2}(1 - TV(P_0, P_1))
```

### Assouad's Lemma
Hypercube arguments for minimax rates:
```
\inf_{\hat{θ}} \sup_{θ} E[||θ̂ - θ||²] ≥ cs²/M
```

under appropriate conditions.

## 5.17 Computational Applications

### PAC Learning
**Sample complexity bounds:**
```
P(R(h) - R̂(h) ≥ ε) ≤ \exp(-c_1 nε² + c_2 \log |\mathcal{H}|)
```

### Online Learning
**Regret bounds** using inequalities:
```
\text{Regret}_n ≤ O(\sqrt{n \log |\mathcal{H}|})
```

### Stochastic Optimization
**Convergence rates** for SGD:
```
E[f(x̄_n) - f*] ≤ O(1/√n)
```

using concentration and martingale inequalities.

## 5.18 Recent Developments

### Transportation Inequalities
**Talagrand's inequality:** For product measures:
```
W₂(μ, ν) ≤ \sqrt{2H(ν|μ)}
```

where W₂ is Wasserstein distance.

### Stein's Method
**Bound distributional distances:**
```
d_{TV}(X, Z) ≤ E[|Xf'(X) - f(X) + f'(X)|]
```

for appropriate test functions f.

### Optimal Transport
**Brenier's theorem** and **Kantorovich duality** provide:
- Generalized concentration inequalities
- Robust statistics bounds
- Dimension-free results

## 5.19 Practical Guidelines

### Choosing the Right Inequality

**Markov:** Use when only mean is known
**Chebyshev:** Use when mean and variance known
**Chernoff:** Use for exponential bounds
**Hoeffding:** Use for bounded variables
**Bernstein:** Use when variance is small

### Tightness Considerations

**Chebyshev is tight** for two-point distributions
**Hoeffding is tight** for Rademacher variables
**Gaussian tail bounds are tight** for normal distributions

### Trade-offs
- **Generality vs. tightness**
- **Computational simplicity vs. sharpness**
- **Assumption requirements vs. bound quality**

## Key Insights

1. **Hierarchy of Bounds:** More assumptions typically yield tighter bounds.

2. **Exponential vs. Polynomial:** Chernoff-type bounds give exponential decay vs. polynomial for Markov/Chebyshev.

3. **Concentration Principle:** Independent variables concentrate around their mean.

4. **Dimension Dependence:** Many bounds scale favorably with dimension.

5. **Martingale Framework:** Unifies many concentration results.

## Common Pitfalls

1. **Wrong tail:** One-sided vs. two-sided bounds
2. **Dependence issues:** Many bounds require independence
3. **Moment conditions:** Checking existence of required moments
4. **Constants:** Inequality constants often not sharp
5. **Asymptotic vs. finite sample:** Distinguishing limiting behavior

## Advanced Topics

### Sub-Gaussian Processes
Extension to infinite-dimensional settings using:
- Metric entropy
- Covering numbers
- Chaining arguments

### Sharp Thresholds
Phase transitions in random systems:
- Percolation
- Random graphs
- Satisfiability

### Concentration on Manifolds
Geometric probability with:
- Ricci curvature bounds
- Heat kernel estimates
- Functional inequalities

## Connections to Other Chapters

### To Chapter 4 (Expectation)
- Jensen's inequality
- Moment bounds
- Variance relationships

### To Chapter 6 (Convergence)
- Law of large numbers
- Central limit theorem
- Empirical process theory

### To Chapter 8 (CDF Estimation)
- Glivenko-Cantelli theorem
- DKW inequality
- Uniform convergence

### To Chapter 23 (Classification)
- Generalization bounds
- PAC learning theory
- Empirical risk minimization

This chapter provides the analytical tools needed to control randomness and uncertainty, forming the backbone of modern probability theory and statistical learning.