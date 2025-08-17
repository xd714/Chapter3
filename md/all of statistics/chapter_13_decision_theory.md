# Chapter 13: Statistical Decision Theory - Mathematical Explanations

## Overview
Statistical decision theory provides a unified framework for making optimal decisions under uncertainty. It formalizes how to choose actions based on data when the consequences of decisions depend on unknown parameters, connecting statistics with economics, game theory, and optimization.

## 13.1 Elements of Decision Theory

### Basic Framework
A **statistical decision problem** consists of:

1. **Parameter space** Θ: Set of possible states of nature
2. **Action space** 𝒜: Set of possible actions/decisions  
3. **Sample space** 𝒳: Set of possible observations
4. **Loss function** L(θ, a): Cost of taking action a when true parameter is θ
5. **Statistical model** {P_θ : θ ∈ Θ}: Family of distributions for data

### Decision Rules
A **decision rule** δ is a function that maps observations to actions:
```
δ: 𝒳 → 𝒜
```

For observed data x, we take action δ(x).

### Randomized Decision Rules
A **randomized decision rule** δ maps observations to probability distributions over actions:
```
δ: 𝒳 → ℳ(𝒜)
```

where ℳ(𝒜) is the set of probability measures on 𝒜.

## 13.2 Loss Functions and Risk

### Common Loss Functions

**Squared error loss:** L(θ, a) = (θ - a)²
- Used for estimation problems
- Penalizes large errors heavily

**Absolute error loss:** L(θ, a) = |θ - a|
- More robust to outliers
- Leads to median as optimal estimator

**0-1 loss:** L(θ, a) = I(θ ≠ a)
- Used for classification/testing
- All errors weighted equally

**Asymmetric loss:** Different penalties for over/under-estimation
```
L(θ, a) = \begin{cases}
c₁(θ - a) & \text{if } a < θ \\
c₂(a - θ) & \text{if } a ≥ θ
\end{cases}
```

### Risk Function
The **risk** of decision rule δ is:
```
R(θ, δ) = E_θ[L(θ, δ(X))] = \int L(θ, δ(x)) dP_θ(x)
```

**Interpretation:** Expected loss when true parameter is θ and we use rule δ.

### Examples

**Point estimation:** 
- Action space: 𝒜 = Θ (estimate the parameter)
- Common loss: L(θ, a) = (θ - a)²
- Risk: R(θ, δ) = E_θ[(θ - δ(X))²] = MSE

**Hypothesis testing:**
- Action space: 𝒜 = {0, 1} (reject/accept H₀)
- 0-1 loss: L(θ, a) = I(wrong decision)
- Risk: R(θ, δ) = P_θ(wrong decision)

## 13.3 Comparing Decision Rules

### Dominance
Decision rule δ₁ **dominates** δ₂ if:
```
R(θ, δ₁) ≤ R(θ, δ₂) for all θ ∈ Θ
```

with strict inequality for at least one θ.

**Admissible rule:** Not dominated by any other rule.
**Inadmissible rule:** Dominated by some other rule.

### Complete Class
A class C of decision rules is **complete** if for every rule δ ∉ C, there exists δ' ∈ C such that δ' dominates δ.

**Minimal complete class:** Complete class with no proper complete subset.

### Optimal Rules
**Problem:** Generally no single rule dominates all others.

**Solutions:**
1. Restrict attention to specific criteria
2. Average over prior distribution (Bayesian)
3. Consider worst-case scenario (minimax)

## 13.4 Bayesian Decision Theory

### Prior Distribution
Assume parameter θ has **prior distribution** π(θ) representing beliefs before seeing data.

### Posterior Distribution
After observing data x, update beliefs using Bayes' theorem:
```
π(θ|x) = \frac{f(x|θ)π(θ)}{m(x)}
```

where m(x) = ∫ f(x|θ)π(θ) dθ is marginal likelihood.

### Posterior Risk
For given data x, the **posterior risk** of action a is:
```
ρ(a|x) = E[L(θ, a)|x] = \int L(θ, a) π(θ|x) dθ
```

### Bayes Action
The **Bayes action** minimizes posterior risk:
```
a*(x) = \argmin_a ρ(a|x)
```

### Bayes Risk
The **Bayes risk** of decision rule δ is:
```
r(π, δ) = E_π[R(θ, δ)] = \int R(θ, δ) π(θ) dθ
```

**Bayes rule:** Decision rule δ_π that minimizes Bayes risk:
```
δ_π = \argmin_δ r(π, δ)
```

### Examples

**Point estimation under squared error:**
- Posterior risk: ρ(a|x) = E[(θ - a)²|x]
- Bayes action: a*(x) = E[θ|x] (posterior mean)

**Point estimation under absolute error:**
- Bayes action: a*(x) = median of π(θ|x)

**Hypothesis testing:**
- H₀: θ ∈ Θ₀ vs H₁: θ ∈ Θ₁
- 0-1 loss with costs c₀, c₁
- Bayes action: Choose H₁ if P(θ ∈ Θ₁|x) > c₀/(c₀ + c₁)

## 13.5 Minimax Decision Theory

### Minimax Criterion
When no prior information available, consider worst-case scenario.

**Minimax rule:** Minimizes maximum risk:
```
δ_M = \argmin_δ \sup_θ R(θ, δ)
```

**Minimax risk:**
```
R_M = \inf_δ \sup_θ R(θ, δ)
```

### Least Favorable Prior
Prior π* is **least favorable** if:
```
\sup_π \inf_δ r(π, δ) = \inf_δ r(π*, δ)
```

**Connection to minimax:** Under regularity conditions, minimax rule equals Bayes rule for least favorable prior.

### Examples

**Normal mean estimation:**
- X ~ N(θ, 1), squared error loss
- Minimax estimator: δ_M(x) = x (sample mean)
- Minimax risk: R_M = 1

**Bernoulli parameter estimation:**
- X ~ Binomial(n, θ), squared error loss  
- Minimax estimator: δ_M(x) = (x + √n/2)/(n + √n)

## 13.6 Admissibility

### Characterizing Admissible Rules
**Theorem:** If δ is a unique Bayes rule for some prior π, then δ is admissible.

**Theorem:** If δ is a limit of Bayes rules and has finite risk, then δ is admissible.

### Complete Class Theorems
**Theorem:** Under mild conditions, the class of all Bayes rules forms a complete class.

**Corollary:** Admissible rules are either Bayes rules or limits of Bayes rules.

### Examples of Inadmissible Estimators

**Stein's phenomenon:** For θ ∈ ℝᵖ with p ≥ 3:
- MLE θ̂ = X is inadmissible under squared error loss
- James-Stein estimator dominates MLE:
  ```
  δ_JS(X) = (1 - (p-2)/||X||²)X
  ```

**Baseball batting averages:** Stein's insight applied to predicting performance.

## 13.7 Invariance

### Group Actions
A **group** G acts on parameter space Θ and sample space 𝒳.

**Invariant decision problem:** Loss function satisfies:
```
L(gθ, ga) = L(θ, a) for all g ∈ G
```

### Invariant Decision Rules
Rule δ is **invariant** if:
```
δ(gx) = gδ(x) for all g ∈ G
```

### Principle of Invariance
**Principle:** When decision problem has invariance structure, restrict attention to invariant rules.

**Justification:** If non-invariant rule δ is optimal, then orbit average of δ is also optimal and invariant.

### Hunt-Stein Theorem
Under certain conditions, best invariant rule is minimax among all rules.

### Examples

**Location parameter:** X = θ + ε where ε ~ F
- Group: G = {translations}
- Invariant estimators: δ(x) = x + c for constant c
- Under squared error: c = 0, so δ(x) = x

**Scale parameter:** X = θε where ε ~ F  
- Group: G = {scalings}
- Invariant estimators: δ(x) = cx for constant c

## 13.8 Empirical Bayes

### Setup
Assume θ ~ G (unknown prior) and X|θ ~ F(·|θ).

**Goal:** Estimate θ based on X without knowing G.

### Empirical Bayes Approach
1. Estimate prior G from marginal distribution of X
2. Use estimated prior in Bayes rule

### Parametric Empirical Bayes
Assume G belongs to parametric family G_α.

**Two-stage procedure:**
1. Estimate α from marginal likelihood
2. Use G_α̂ as prior in Bayes rule

### Nonparametric Empirical Bayes
Estimate G nonparametrically from marginal distribution.

**Example (Robbins):** For Poisson model:
- X|θ ~ Poisson(θ), θ ~ G
- Bayes estimator: E[θ|X = x] = (x+1)f(x+1)/f(x)
- Empirical Bayes: Replace f with empirical probabilities

### Asymptotic Properties
**Theorem:** Under regularity conditions, empirical Bayes risk approaches Bayes risk as sample size increases.

## 13.9 Sequential Decision Theory

### Sequential Sampling
At each stage, decide whether to:
1. **Stop** and make terminal decision
2. **Continue** and take another observation

### Optimal Stopping
**Value function:** V(x) = expected future payoff starting from state x

**Bellman equation:**
```
V(x) = \max{\text{stop value}, \text{continuation value}}
```

### Sequential Probability Ratio Test (SPRT)
**Problem:** Test H₀: θ = θ₀ vs H₁: θ = θ₁

**Likelihood ratio:** Λₙ = ∏ᵢ₌₁ⁿ f(Xᵢ|θ₁)/f(Xᵢ|θ₀)

**Decision rule:**
- If Λₙ ≥ B: reject H₀
- If Λₙ ≤ A: accept H₀  
- If A < Λₙ < B: continue sampling

**Optimality:** SPRT minimizes expected sample size among all tests with given error probabilities.

### Multi-armed Bandits
**Setup:** K arms, each with unknown reward distribution

**Goal:** Maximize total reward over T periods

**Explore vs exploit dilemma:**
- **Exploration:** Try different arms to learn rewards
- **Exploitation:** Choose best-known arm

**UCB algorithm:** Choose arm with highest upper confidence bound
**Thompson sampling:** Choose arm randomly based on posterior probabilities

## 13.10 Game Theory Connections

### Two-Person Zero-Sum Games
**Players:** Statistician (chooses δ) vs Nature (chooses θ)
**Payoff:** -R(θ, δ) to statistician

**Minimax theorem:** Under mild conditions:
```
\inf_δ \sup_θ R(θ, δ) = \sup_π \inf_δ r(π, δ)
```

### Mixed Strategies
**Statistician's mixed strategy:** Randomized decision rule
**Nature's mixed strategy:** Prior distribution on θ

### Nash Equilibrium
**Definition:** Strategy profile where no player can improve by unilateral deviation

**Connection:** Minimax strategies form Nash equilibrium.

## 13.11 Information Theory and Decision Theory

### Mutual Information
**Definition:** I(X; θ) = H(θ) - H(θ|X)

**Interpretation:** Expected reduction in uncertainty about θ after observing X.

### Value of Information
**Expected value of sample information:**
```
EVSI = r(π, δ₀) - r(π, δ*)
```

where δ₀ is optimal action without data, δ* is Bayes rule.

### Information and Risk
**General principle:** More informative experiments lead to better decisions (lower Bayes risk).

## 13.12 Robust Decision Theory

### Model Uncertainty
**Problem:** True model may not be in assumed class.

**Γ-minimax:** Minimize maximum risk over class Γ of possible models:
```
\inf_δ \sup_{P \in Γ} R_P(δ)
```

### Contamination Models
**ε-contamination:** True distribution in {(1-ε)F + εG : G ∈ 𝒢}

**Robust Bayes:** Use least favorable prior within contamination class.

### Sensitivity Analysis
Study how Bayes decisions change with prior specification:
- **Prior robustness:** Range of priors leading to same decision
- **Posterior robustness:** How much data needed to overcome prior

## 13.13 Computational Aspects

### Numerical Integration
Bayesian decisions often require computing:
```
\int L(θ, a) π(θ|x) dθ
```

**Methods:**
- Quadrature rules
- Monte Carlo integration
- Importance sampling

### MCMC for Decision Theory
**Gibbs sampling:** Sample from posterior π(θ|x)
**Metropolis-Hastings:** General MCMC for complex posteriors

**Decision making:** Use MCMC samples to approximate posterior risk.

### Approximation Methods
**Laplace approximation:** Approximate posterior with normal distribution
**Variational Bayes:** Optimize over simpler family of distributions

## 13.14 Multiple Decision Problems

### Simultaneous Inference
**Problem:** Make K decisions simultaneously
**Loss:** L(θ, a₁, ..., aₖ) for joint actions

**Compound decisions:** Many similar problems
**Empirical Bayes approach:** Learn from similar problems

### False Discovery Rate
**Multiple testing:** Control expected proportion of false discoveries
**Connection to decision theory:** FDR procedures are Bayes rules under specific loss functions

### Hierarchical Models
**Structure:** θᵢ ~ G, Xᵢ|θᵢ ~ F(·|θᵢ)
**Shrinkage:** Individual estimates pulled toward group mean

## 13.15 Applications

### Medical Diagnosis
**Elements:**
- θ: Disease state
- X: Test results  
- A: Treatment decisions
- L: Cost of incorrect treatment

**ROC analysis:** Trade-off between sensitivity and specificity

### Quality Control
**Process monitoring:** Decide when to adjust/stop process
**Sequential sampling:** Continue monitoring vs take action

### Portfolio Optimization
**Mean-variance framework:** Balance expected return vs risk
**Bayesian portfolio:** Incorporate parameter uncertainty

### Clinical Trials
**Adaptive designs:** Modify trial based on interim results
**Stopping rules:** Balance statistical power vs patient welfare

## 13.16 Asymptotic Decision Theory

### Large Sample Behavior
**Asymptotic risk:** lim_{n→