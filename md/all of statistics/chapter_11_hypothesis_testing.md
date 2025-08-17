# Chapter 11: Hypothesis Testing and p-values - Mathematical Explanations

## Overview
Hypothesis testing provides a framework for making decisions about population parameters based on sample data. This chapter covers the fundamental concepts of null and alternative hypotheses, test statistics, p-values, and various testing procedures.

## 11.1 Introduction to Hypothesis Testing

### Basic Framework
We partition the parameter space Θ into two disjoint sets Θ₀ and Θ₁ and wish to test:
```
H₀: θ ∈ Θ₀  (null hypothesis)
versus
H₁: θ ∈ Θ₁  (alternative hypothesis)
```

### Decision Rule
Let X be a random variable with range 𝒳. We test a hypothesis by finding an appropriate subset R ⊆ 𝒳 called the **rejection region**:
```
X ∈ R    ⟹ reject H₀
X ∉ R    ⟹ retain (do not reject) H₀
```

Usually the rejection region has the form:
```
R = {x : T(x) > c}
```
where T is a **test statistic** and c is a **critical value**.

## 11.2 Types of Errors

### Type I and Type II Errors
|               | H₀ true    | H₀ false   |
|---------------|------------|------------|
| Reject H₀     | Type I     | Correct    |
| Retain H₀     | Correct    | Type II    |

### Error Probabilities
- **Type I Error (α):** P(reject H₀ | H₀ true) = P(Type I error)
- **Type II Error (β):** P(retain H₀ | H₀ false) = P(Type II error)
- **Power:** 1 - β = P(reject H₀ | H₀ false)

### Size and Level
- **Size of test:** α = sup_{θ∈Θ₀} P_θ(X ∈ R)
- **Level-α test:** A test with size ≤ α

## 11.3 Test Statistics and p-values

### Wald Test
For parameter θ with estimate θ̂:
```
W = (θ̂ - θ₀) / ŝe(θ̂)
```

Under H₀: θ = θ₀, W approximately follows N(0,1) for large samples.

**Rejection region:** |W| > z_{α/2}

### Likelihood Ratio Test
```
λ(x) = L(θ̂₀)/L(θ̂)
```

where:
- θ̂₀ = MLE under H₀
- θ̂ = unrestricted MLE

**Wilks' Theorem:** Under regularity conditions and H₀:
```
-2 ln λ(X) → χ²_k
```
where k = dim(Θ) - dim(Θ₀).

### Score Test
Based on the score function at θ₀:
```
S = ∂ℓ/∂θ|_{θ=θ₀}
```

**Test statistic:**
```
T = S²/I(θ₀)
```

where I(θ₀) is the Fisher information.

Under H₀: T → χ²₁

## 11.4 p-values

### Definition
The **p-value** is the probability of observing a test statistic as extreme or more extreme than what was actually observed, assuming H₀ is true:
```
p-value = sup_{θ∈Θ₀} P_θ(T(X) ≥ T(x))
```

### Properties
1. If H₀ is true, p-value ~ Uniform(0,1)
2. Smaller p-values indicate stronger evidence against H₀
3. **p-value ≠ P(H₀|data)**

### Evidence Scale
| p-value    | Evidence                           |
|------------|----------------------------------- |
| < 0.01     | Very strong evidence against H₀    |
| 0.01-0.05  | Strong evidence against H₀         |
| 0.05-0.10  | Weak evidence against H₀           |
| > 0.10     | Little or no evidence against H₀   |

### Common p-value Calculations

**Two-sided test with normal statistic:**
```
p-value = 2P(|Z| > |z|) = 2Φ(-|z|)
```

**One-sided test:**
```
p-value = P(Z > z) = 1 - Φ(z)  [upper tail]
p-value = P(Z < z) = Φ(z)      [lower tail]
```

## 11.5 Specific Tests

### One-Sample t-test
Testing H₀: μ = μ₀ for normal population with unknown σ.

**Test statistic:**
```
T = (X̄ - μ₀)/(S/√n)
```

Under H₀: T ~ t_{n-1}

**p-value:**
- Two-sided: p = 2P(t_{n-1} > |t|)
- One-sided: p = P(t_{n-1} > t) or P(t_{n-1} < t)

### Two-Sample t-test
Testing H₀: μ₁ = μ₂ for two normal populations.

**Equal variances assumed:**
```
T = (X̄₁ - X̄₂)/(S_p√(1/n₁ + 1/n₂))
```

where S²_p = [(n₁-1)S²₁ + (n₂-1)S²₂]/(n₁+n₂-2)

Under H₀: T ~ t_{n₁+n₂-2}

**Welch's t-test (unequal variances):**
```
T = (X̄₁ - X̄₂)/√(S²₁/n₁ + S²₂/n₂)
```

Degrees of freedom approximated by Welch-Satterthwaite formula.

### Chi-square Goodness-of-fit Test
Testing H₀: distribution follows specified form.

**Pearson's χ² statistic:**
```
T = Σ (O_i - E_i)²/E_i
```

where O_i = observed frequency, E_i = expected frequency.

Under H₀: T → χ²_{k-1-p} where p = number of estimated parameters.

### Chi-square Test of Independence
Testing independence in contingency tables.

**Test statistic:**
```
T = Σᵢ Σⱼ (O_{ij} - E_{ij})²/E_{ij}
```

where E_{ij} = (row total × column total)/grand total.

Under H₀: T → χ²_{(r-1)(c-1)}

## 11.6 Multiple Testing

### Family-wise Error Rate (FWER)
Probability of making at least one Type I error:
```
FWER = P(reject at least one true H₀)
```

### Bonferroni Correction
For m tests, use significance level α/m for each test:
```
P(at least one Type I error) ≤ m × (α/m) = α
```

**Adjusted p-values:** p̃ᵢ = min(1, m·pᵢ)

### Holm's Method
1. Order p-values: p₍₁₎ ≤ ... ≤ p₍ₘ₎
2. Find smallest k such that p₍ₖ₎ > α/(m-k+1)
3. Reject H₍₁₎, ..., H₍ₖ₋₁₎

### False Discovery Rate (FDR)
Expected proportion of false discoveries among rejected hypotheses:
```
FDR = E[V/R | R > 0]
```

where V = false discoveries, R = total rejections.

**Benjamini-Hochberg procedure:**
1. Order p-values: p₍₁₎ ≤ ... ≤ p₍ₘ₎
2. Find largest k such that p₍ₖ₎ ≤ (k/m)α
3. Reject H₍₁₎, ..., H₍ₖ₎

## 11.7 Power and Sample Size

### Power Function
The **power function** is:
```
β(θ) = P_θ(X ∈ R)
```

**Properties:**
- β(θ) ≤ α for θ ∈ Θ₀ (size constraint)
- β(θ) should be large for θ ∈ Θ₁ (power requirement)

### Sample Size Calculation
For testing H₀: μ = μ₀ vs H₁: μ = μ₁ with significance level α and power 1-β:
```
n = [σ²(z_{α/2} + z_β)²] / (μ₁ - μ₀)²
```

**For one-sided test:**
```
n = [σ²(z_α + z_β)²] / (μ₁ - μ₀)²
```

### Effect Size
**Cohen's d:**
```
d = (μ₁ - μ₀) / σ
```

**Interpretation:**
- d = 0.2: small effect
- d = 0.5: medium effect  
- d = 0.8: large effect

## 11.8 Nonparametric Tests

### Sign Test
For median θ, testing H₀: θ = θ₀.

**Test statistic:** S = number of observations > θ₀

Under H₀: S ~ Binomial(n, 1/2)

**p-value (two-sided):** p = 2min{P(S ≤ s), P(S ≥ s)}

### Wilcoxon Signed-Rank Test
For symmetric distribution around θ, testing H₀: θ = θ₀.

**Procedure:**
1. Compute differences: Dᵢ = Xᵢ - θ₀
2. Rank absolute differences: |D₍₁₎| ≤ ... ≤ |D₍ₙ₎|
3. Test statistic: W⁺ = sum of ranks for positive differences

**Distribution:** For large n, W⁺ is approximately normal with:
- Mean: n(n+1)/4
- Variance: n(n+1)(2n+1)/24

### Mann-Whitney U Test
For testing H₀: F_X = F_Y (equal distributions).

**Test statistic:**
```
U = Σᵢ Σⱼ I(Xᵢ > Yⱼ)
```

**Equivalent form:**
```
U = R_X - n_X(n_X + 1)/2
```

where R_X = sum of ranks for X sample.

**Distribution:** For large samples, U is approximately normal.

### Kolmogorov-Smirnov Test
Testing H₀: F = F₀ (specified distribution).

**Test statistic:**
```
D_n = sup_x |F̂_n(x) - F₀(x)|
```

**Critical values:** Depends on Kolmogorov distribution.

**Two-sample version:**
```
D_{m,n} = sup_x |F̂_m(x) - Ĝ_n(x)|
```

## 11.9 Permutation Tests

### Principle
If H₀ implies exchangeability, all permutations of the data are equally likely under H₀.

### General Algorithm
1. Compute test statistic T_obs for observed data
2. Generate all (or many) permutations of the data
3. Compute test statistic for each permutation
4. p-value = proportion of permutation statistics ≥ T_obs

### Two-Sample Permutation Test
Testing H₀: F_X = F_Y.

**Test statistic:** T = |X̄ - Ȳ|

**Permutation distribution:** All (m+n choose m) ways to assign labels to combined sample.

**p-value:**
```
p = (1/N!) Σ I(T_j > T_obs)
```

where sum is over all N! permutations.

## 11.10 Bootstrap Tests

### Bootstrap p-values
1. Compute test statistic T_obs
2. Generate bootstrap samples under H₀
3. Compute bootstrap test statistics T₁*, ..., T_B*
4. p-value = proportion of T_i* ≥ T_obs

### Advantages
- Doesn't require asymptotic theory
- Adapts to actual sampling distribution
- Works for complex statistics

## 11.11 Bayesian Testing

### Bayes Factors
**Bayes factor** for H₁ vs H₀:
```
BF₁₀ = P(Data|H₁) / P(Data|H₀)
```

**Interpretation:**
- BF₁₀ > 1: Evidence for H₁
- BF₁₀ < 1: Evidence for H₀
- BF₁₀ = 1: No evidence either way

### Posterior Odds
```
Posterior Odds = Prior Odds × Bayes Factor
```

```
P(H₁|Data) / P(H₀|Data) = [P(H₁)/P(H₀)] × BF₁₀
```

### Jeffreys' Scale for Bayes Factors
| BF₁₀        | Evidence for H₁                |
|-------------|--------------------------------|
| 1-3         | Barely worth mentioning        |
| 3-10        | Substantial                    |
| 10-30       | Strong                         |
| 30-100      | Very strong                    |
| >100        | Decisive                       |

## 11.12 Model Selection and Information Criteria

### Likelihood Ratio Tests for Nested Models
For nested models M₀ ⊂ M₁:
```
LRT = -2[ℓ(θ̂₀) - ℓ(θ̂₁)]
```

Under H₀ (M₀ is correct): LRT → χ²_k where k = difference in parameters.

### Akaike Information Criterion (AIC)
```
AIC = -2ℓ(θ̂) + 2p
```

where p = number of parameters.

**Model selection:** Choose model with smallest AIC.

### Bayesian Information Criterion (BIC)
```
BIC = -2ℓ(θ̂) + p ln(n)
```

**Properties:**
- BIC penalizes complexity more than AIC
- BIC is consistent for model selection
- AIC minimizes prediction error

## 11.13 Sequential Testing

### Sequential Probability Ratio Test (SPRT)
Testing H₀: θ = θ₀ vs H₁: θ = θ₁.

**Likelihood ratio:**
```
Λ_n = L(θ₁)/L(θ₀)
```

**Decision rule:**
- If Λ_n ≥ B: reject H₀
- If Λ_n ≤ A: accept H₀  
- If A < Λ_n < B: take another observation

**Boundaries:**
```
A = β/(1-α)
B = (1-β)/α
```

### Properties
- Minimizes expected sample size
- Maintains error probabilities α and β
- Stopping time is finite with probability 1

## 11.14 Robustness and Assumptions

### Robustness of t-tests
- **Normality:** t-test robust to moderate deviations from normality
- **Equal variances:** Welch's t-test when variances unequal
- **Independence:** Violations can severely affect validity

### Diagnostic Methods
1. **Q-Q plots:** Check normality assumption
2. **Residual plots:** Check homoscedasticity
3. **Tests for assumptions:** Shapiro-Wilk, Levene's test

### Transformations
- **Log transformation:** For right-skewed data
- **Square root:** For count data
- **Box-Cox:** General power transformations

## 11.15 Modern Developments

### False Discovery Rate Control
Control expected proportion of false discoveries rather than family-wise error rate.

### High-dimensional Testing
- **Sparse signals:** Most null hypotheses true
- **Dependency:** Tests not independent
- **New methods:** Higher criticism, adaptive procedures

### Testing in Machine Learning
- **Cross-validation:** For model comparison
- **Permutation importance:** For feature selection
- **Stability selection:** For variable selection

## 11.16 Common Pitfalls and Misconceptions

### p-hacking and Multiple Comparisons
- **Cherry-picking:** Testing many hypotheses, reporting only significant ones
- **Data dredging:** Analyzing data until finding significance
- **HARKing:** Hypothesizing after results are known

### Misinterpretation of p-values
1. **p-value ≠ P(H₀|data):** This is the prosecutor's fallacy
2. **Significance ≠ practical importance:** Statistical vs practical significance
3. **Non-significance ≠ no effect:** Absence of evidence ≠ evidence of absence

### Base Rate Fallacy
Low base rate can make even accurate tests misleading:
```
PPV = (Sensitivity × Prevalence) / [Sensitivity × Prevalence + (1-Specificity) × (1-Prevalence)]
```

## 11.17 Practical Guidelines

### Choosing a Test
1. **Data type:** Continuous, discrete, categorical
2. **Sample size:** Large sample vs small sample methods
3. **Assumptions:** Parametric vs nonparametric
4. **Research question:** One-sided vs two-sided

### Reporting Results
1. **Effect size:** Not just p-value
2. **Confidence intervals:** Provide range of plausible values
3. **Assumptions:** State and check assumptions
4. **Multiple testing:** Adjust for multiple comparisons when appropriate

### Power Analysis
- **Prospective:** Determine sample size needed
- **Retrospective:** Assess power of completed study
- **Sensitivity analysis:** How results depend on assumptions

## 11.18 Connections to Other Chapters

### To Chapter 9 (Bootstrap)
- Bootstrap hypothesis tests
- Bootstrap p-values
- Permutation tests

### To Chapter 12 (Bayesian Inference)
- Bayesian vs frequentist testing
- Bayes factors vs p-values
- Posterior probabilities

### To Chapter 13 (Decision Theory)
- Optimal tests (Neyman-Pearson lemma)
- Minimax tests
- Risk functions

## Key Insights

1. **Testing Framework:** Hypothesis testing provides a systematic approach to decision-making under uncertainty.

2. **Error Control:** The trade-off between Type I and Type II errors is fundamental to test design.

3. **p-values:** Widely used but often misunderstood; they measure compatibility with null hypothesis, not evidence for it.

4. **Multiple Testing:** When testing many hypotheses, error rates must be carefully controlled.

5. **Assumptions Matter:** Violations of assumptions can invalidate test results.

## Important Formulas Summary

**Wald test:** W = (θ̂ - θ₀)/ŝe(θ̂)

**Likelihood ratio:** λ = L(θ̂₀)/L(θ̂)

**Chi-square statistic:** χ² = Σ(O - E)²/E

**Sample size (two-sided):** n = σ²(z_{α/2} + z_β)²/(μ₁ - μ₀)²

**Bonferroni correction:** α_adj = α/m

**FDR procedure:** Reject if p₍ᵢ₎ ≤ (i/m)α

This chapter provides the foundation for making statistical decisions and forms the backbone of applied statistics across all fields.