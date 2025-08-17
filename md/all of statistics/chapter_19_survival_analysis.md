# Chapter 19: Survival Analysis - Mathematical Explanations

## Overview
Survival analysis studies the time until an event occurs, accounting for censoring and truncation. This chapter covers survival functions, hazard rates, censoring mechanisms, nonparametric estimation methods, and regression models for survival data.

## 19.1 Introduction to Survival Analysis

### Survival Data Characteristics
**Time-to-event data:** T represents time until event of interest
**Censoring:** Event not observed for all subjects
**Truncation:** Only observe subjects meeting certain criteria

### Examples
- **Medical:** Time to death, disease recurrence, treatment response
- **Engineering:** Time to equipment failure, component lifetime
- **Economics:** Duration of unemployment, time to default
- **Marketing:** Customer retention time, time to purchase

### Key Concepts
1. **Survival time:** T ≥ 0 (non-negative random variable)
2. **Event indicator:** δ = 1 if event observed, 0 if censored
3. **Observation:** (t, δ) where t is observed time
4. **Follow-up period:** Administrative end of study

## 19.2 Survival Functions and Hazard Rates

### Survival Function
**Definition:** S(t) = P(T > t) = probability of surviving beyond time t

**Properties:**
- S(0) = 1 (everyone alive at start)
- S(∞) = 0 (assuming everyone eventually experiences event)
- S(t) is decreasing and right-continuous
- S(t) = 1 - F(t) where F is cumulative distribution function

### Probability Density Function
```
f(t) = lim_{Δt→0} P(t ≤ T < t + Δt)/Δt = -dS(t)/dt
```

**Relationship:** f(t) = -S'(t)

### Hazard Function
**Definition:** 
```
λ(t) = lim_{Δt→0} P(t ≤ T < t + Δt | T ≥ t)/Δt
```

**Interpretation:** Instantaneous risk of event at time t given survival to time t.

**Alternative form:**
```
λ(t) = f(t)/S(t) = -d log S(t)/dt
```

### Cumulative Hazard Function
```
Λ(t) = ∫₀ᵗ λ(u) du
```

**Relationship to survival function:**
```
S(t) = exp(-Λ(t))
```

### Fundamental Relationships
```
S(t) = exp(-∫₀ᵗ λ(u) du)
f(t) = λ(t) exp(-∫₀ᵗ λ(u) du)
λ(t) = f(t)/(1 - F(t))
```

## 19.3 Common Survival Distributions

### Exponential Distribution
**Survival function:** S(t) = e^(-λt)
**Hazard function:** λ(t) = λ (constant)
**Mean survival time:** E[T] = 1/λ

**Memoryless property:** P(T > s + t | T > s) = P(T > t)

### Weibull Distribution
**Survival function:** S(t) = exp(-(t/α)^β)
**Hazard function:** λ(t) = (β/α)(t/α)^(β-1)

**Shape parameter β:**
- β < 1: Decreasing hazard
- β = 1: Constant hazard (exponential)
- β > 1: Increasing hazard

### Log-normal Distribution
If log(T) ~ N(μ, σ²), then T has log-normal distribution.
**Hazard function:** Initially increases then decreases (unimodal)

### Gamma Distribution
**Survival function:** No closed form
**Hazard function:** Flexible shape depending on parameters

### Gompertz Distribution
**Hazard function:** λ(t) = λe^(γt) (exponentially increasing)
**Application:** Human mortality at advanced ages

## 19.4 Censoring Mechanisms

### Right Censoring
**Definition:** Event time T > C where C is censoring time
**Observation:** (min(T, C), I(T ≤ C))

**Types:**
1. **Type I censoring:** Fixed censoring time
2. **Type II censoring:** Study ends after r events
3. **Random censoring:** Censoring time is random

### Left Censoring
**Definition:** Event occurred before observation began
**Example:** Age at disease onset when only current age known

### Interval Censoring
**Definition:** Event known to occur within interval [L, R]
**Example:** Disease progression detected between clinic visits

### Truncation
**Left truncation:** Only observe subjects who survived past time L
**Right truncation:** Only observe subjects who experienced event before time R

### Informative vs Non-informative Censoring
**Non-informative:** Censoring mechanism independent of survival time
**Informative:** Censoring related to survival (violates standard methods)

**Assumption:** Most survival analysis methods assume non-informative censoring.

## 19.5 Nonparametric Estimation

### Kaplan-Meier Estimator
**For right-censored data:**
```
Ŝ(t) = ∏_{tᵢ≤t} (1 - dᵢ/nᵢ)
```

where:
- tᵢ are distinct event times
- dᵢ = number of events at time tᵢ  
- nᵢ = number at risk at time tᵢ

**Properties:**
- Step function with jumps at event times
- Decreases only at uncensored event times
- Self-consistent: uses all available information

### Greenwood's Formula
**Variance of Kaplan-Meier estimator:**
```
Var(Ŝ(t)) ≈ (Ŝ(t))² ∑_{tᵢ≤t} dᵢ/(nᵢ(nᵢ - dᵢ))
```

**Standard error:**
```
se(Ŝ(t)) = Ŝ(t)√(∑_{tᵢ≤t} dᵢ/(nᵢ(nᵢ - dᵢ)))
```

### Confidence Intervals for S(t)

**Linear scale:**
```
Ŝ(t) ± z_{α/2} · se(Ŝ(t))
```

**Log scale (better properties):**
```
Ŝ(t) exp(±z_{α/2} · se(log Ŝ(t)))
```

where se(log Ŝ(t)) = se(Ŝ(t))/Ŝ(t)

**Log-log scale:**
```
Ŝ(t)^{exp(±z_{α/2} · se(log(-log Ŝ(t))))}
```

### Nelson-Aalen Estimator
**Cumulative hazard estimator:**
```
Λ̂(t) = ∑_{tᵢ≤t} dᵢ/nᵢ
```

**Relationship:** Ŝ(t) ≈ exp(-Λ̂(t))

**Variance:**
```
Var(Λ̂(t)) ≈ ∑_{tᵢ≤t} dᵢ/nᵢ²
```

## 19.6 Comparing Survival Curves

### Log-rank Test
**Null hypothesis:** H₀: S₁(t) = S₂(t) for all t

**Test statistic:**
```
Z = (O₁ - E₁)/√V₁
```

where:
- O₁ = observed events in group 1
- E₁ = expected events in group 1 under H₀
- V₁ = variance of O₁ - E₁

**At each event time tᵢ:**
```
e₁ᵢ = n₁ᵢdᵢ/nᵢ
v₁ᵢ = (n₁ᵢn₂ᵢdᵢ(nᵢ - dᵢ))/(nᵢ²(nᵢ - 1))
```

**Overall statistics:**
```
E₁ = ∑ᵢ e₁ᵢ
V₁ = ∑ᵢ v₁ᵢ
```

**Distribution:** Under H₀, Z² ~ χ²₁

### Weighted Log-rank Tests
**General form:** Weight each time point differently
```
Z = ∑ᵢ wᵢ(O₁ᵢ - E₁ᵢ)/√(∑ᵢ wᵢ²V₁ᵢ)
```

**Gehan-Breslow test:** wᵢ = nᵢ (weights by sample size)
**Tarone-Ware test:** wᵢ = √nᵢ
**Peto test:** wᵢ = Ŝ(tᵢ₋)

### Multiple Group Comparisons
**K groups:** Test H₀: S₁(t) = ... = Sₖ(t)

**Test statistic:**
```
Q = (O - E)ᵀV⁻¹(O - E) ~ χ²ₖ₋₁
```

where O, E are vectors of observed and expected events.

## 19.7 Cox Proportional Hazards Model

### Model Specification
```
λ(t|x) = λ₀(t) exp(βᵀx)
```

where:
- λ₀(t) is baseline hazard (unspecified)
- β is vector of regression coefficients
- x is vector of covariates

**Key feature:** Semi-parametric (no assumption about λ₀(t))

### Hazard Ratio
**For covariate change from x to x + Δx:**
```
HR = λ(t|x + Δx)/λ(t|x) = exp(βᵀΔx)
```

**Interpretation:** 
- HR > 1: Increased hazard
- HR < 1: Decreased hazard
- HR = 1: No effect

### Proportional Hazards Assumption
**Key assumption:** Hazard ratio constant over time
```
λᵢ(t)/λⱼ(t) = exp(β(xᵢ - xⱼ)) for all t
```

**Violation:** Time-varying coefficients needed

### Partial Likelihood
**Cox's insight:** Can eliminate λ₀(t) using conditional likelihood

**At event time tᵢ with event in subject j:**
```
L(β) = ∏ᵢ exp(βᵀxⱼ)/∑_{ℓ∈R(tᵢ)} exp(βᵀxℓ)
```

where R(tᵢ) is risk set at time tᵢ.

**Properties:**
- Treats λ₀(t) as nuisance parameter
- Standard likelihood methods apply
- Asymptotically efficient

### Estimation and Inference
**Score function:**
```
U(β) = ∑ᵢ [xⱼ - Ē(tᵢ, β)]
```

where Ē(tᵢ, β) = ∑_{ℓ∈R(tᵢ)} xℓexp(βᵀxℓ)/∑_{ℓ∈R(tᵢ)} exp(βᵀxℓ)

**Information matrix:**
```
I(β) = ∑ᵢ V̄(tᵢ, β)
```

**Asymptotic distribution:**
```
√n(β̂ - β) ⇝ N(0, I⁻¹(β))
```

### Baseline Hazard Estimation
**Breslow estimator:**
```
Λ̂₀(t) = ∑_{tᵢ≤t} 1/∑_{ℓ∈R(tᵢ)} exp(β̂ᵀxℓ)
```

**Baseline survival:**
```
Ŝ₀(t) = exp(-Λ̂₀(t))
```

## 19.8 Model Diagnostics and Extensions

### Checking Proportional Hazards
**Schoenfeld residuals:**
```
rᵢⱼ = xⱼᵢ - Ē(tᵢ, β̂)
```

**Test:** Plot residuals vs time; slope ≠ 0 suggests violation

**Formal test:** Test correlation between residuals and time/rank of time

### Martingale Residuals
```
Mᵢ = δᵢ - Λ̂(tᵢ|xᵢ)
```

**Properties:**
- Sum to zero
- Detect functional form of covariates
- Asymptotically normal

### Deviance Residuals
**Normalized martingale residuals:**
```
Dᵢ = sign(Mᵢ)√(-2[Mᵢ + δᵢ log(δᵢ - Mᵢ)])
```

**More symmetric than martingale residuals**

### Time-Varying Coefficients
**Extended Cox model:**
```
λ(t|x) = λ₀(t) exp(β(t)ᵀx)
```

**Methods:**
- Stratification by time periods
- Smooth time-varying coefficients
- Interaction with functions of time

### Stratified Cox Model
**When proportional hazards violated:**
```
λₕ(t|x) = λ₀ₕ(t) exp(βᵀx)
```

for strata h = 1, ..., H

**Separate baseline hazards, common covariate effects**

## 19.9 Accelerated Failure Time Models

### Model Specification
```
log T = βᵀx + σε
```

where ε has known distribution (e.g., standard extreme value, normal)

**Survival function:**
```
S(t|x) = S₀((log t - βᵀx)/σ)
```

### Interpretation
**Acceleration factor:** exp(-βᵀx)
- AF > 1: Accelerates time to event (shorter survival)
- AF < 1: Decelerates time to event (longer survival)

### Common Distributions
**Weibull AFT:** ε has extreme value distribution
**Log-normal AFT:** ε ~ N(0,1)
**Log-logistic AFT:** ε has logistic distribution

### Estimation
**Maximum likelihood:** When distribution specified
**Rank-based methods:** Distribution-free approaches
**Least squares:** For normal and logistic errors

### Connection to Cox Model
**Weibull distribution:** Both Cox and AFT models apply
```
λ(t|x) = λ₀(t) exp(βᵀx) ⟺ log T = α + β*ᵀx + σε
```

with β* = -σβ

## 19.10 Competing Risks

### Setup
**Multiple event types:** T₁, T₂, ..., Tₖ
**Observed:** T = min(T₁, ..., Tₖ) and cause J

### Cause-Specific Hazards
```
λⱼ(t) = lim_{Δt→0} P(t ≤ T < t + Δt, J = j | T ≥ t)/Δt
```

**Overall hazard:** λ(t) = ∑ⱼ λⱼ(t)

### Cumulative Incidence Function
**Probability of experiencing event j by time t:**
```
Fⱼ(t) = ∫₀ᵗ S(u)λⱼ(u) du
```

where S(t) = exp(-∫₀ᵗ λ(u) du)

**Note:** ∑ⱼ Fⱼ(∞) ≤ 1 (some may be censored)

### Estimation
**Aalen-Johansen estimator:**
```
F̂ⱼ(t) = ∫₀ᵗ Ŝ(u-) dΛ̂ⱼ(u)
```

### Gray's Test
**Compare cumulative incidence between groups**
**Weights by probability of being at risk for specific cause**

## 19.11 Frailty Models

### Motivation
**Unobserved heterogeneity:** Individuals differ in susceptibility

### Shared Frailty Model
```
λᵢⱼ(t) = ωᵢλ₀(t) exp(βᵀxᵢⱼ)
```

where ωᵢ is cluster-specific frailty (usually gamma distributed)

**Applications:**
- Medical: Patients within hospitals
- Engineering: Components from same manufacturer
- Family studies: Genetic susceptibility

### Estimation
**Marginal likelihood:** Integrate out frailty
**EM algorithm:** Treat frailty as missing data
**Penalized partial likelihood:** Approximate methods

## 19.12 Interval Censored Data

### Types of Interval Censoring
**Case I:** Left-censored or observed exactly
**Case II:** General interval censoring
**Case III:** Current status data (only know if event occurred by observation time)

### Nonparametric Estimation
**Turnbull estimator:** Self-consistent algorithm
**NPMLE:** May not be unique without smoothing

### Parametric Models
**Assume distribution family and estimate parameters**
**Maximum likelihood with interval-censored likelihood contributions**

## 19.13 Multivariate Survival Analysis

### Clustered Data
**Observations correlated within clusters**
**Examples:** Twins, matched pairs, multicenter studies

### Marginal Models
**Model marginal survival functions**
**Use robust standard errors to account for correlation**

### Copula Models
**Separate marginal survival from dependence structure**
```
S(t₁, t₂) = C(S₁(t₁), S₂(t₂))
```

where C is copula function

### Recurrent Events
**Multiple events per subject**
**Models:**
- Andersen-Gill (treat as counting process)
- Wei-Lin-Weissfeld (marginal approach)
- Frailty models (random effects)

## 19.14 Bayesian Survival Analysis

### Prior Specification
**Parametric models:** Priors on parameters
**Semi-parametric:** Priors on baseline hazard (e.g., gamma process)

### MCMC Implementation
**Data augmentation:** Treat exact event times as missing data
**Slice sampling:** For complex posterior distributions

### Bayesian Cox Model
**Prior on regression coefficients:** Usually normal
**Prior on baseline hazard:** Gamma process or beta process

### Model Comparison
**Bayes factors:** Compare different models
**DIC:** Deviance information criterion
**Cross-validation:** Predictive accuracy

## 19.15 Software and Implementation

### R Packages
**survival:** Core survival analysis functions
**survminer:** Visualization
**coxme:** Mixed effects Cox models
**frailtypack:** Frailty models

### Key Functions
- `Surv()`: Create survival objects
- `survfit()`: Kaplan-Meier estimation
- `survdiff()`: Log-rank tests
- `coxph()`: Cox proportional hazards
- `survreg()`: Parametric models

### Data Format
**Standard format:** (time, status, covariates)
**Counting process:** (start, stop, status, covariates)

## 19.16 Applications

### Clinical Trials
**Primary endpoint:** Overall survival, progression-free survival
**Interim analyses:** Monitoring for efficacy/safety
**Sample size calculation:** Based on hazard ratios and event rates

### Epidemiology
**Disease onset:** Time to disease development
**Risk factors:** Identify prognostic factors
**Public health:** Population-level survival patterns

### Engineering Reliability
**Component lifetime:** Time to failure analysis
**Maintenance scheduling:** Optimize replacement timing
**Quality control:** Monitor manufacturing processes

### Business Analytics
**Customer churn:** Time to customer departure
**Marketing:** Time to purchase, response to campaigns
**Credit risk:** Time to default, bankruptcy

## Key Insights

1. **Censoring is Information:** Partial information is valuable and should be used properly.

2. **Time-to-Event is Special:** Requires different methods than standard regression.

3. **Proportional Hazards:** Powerful assumption but should be verified.

4. **Semi-parametric Flexibility:** Cox model balances flexibility with interpretability.

5. **Clinical Relevance:** Methods designed with practical applications in mind.

## Common Pitfalls

1. **Ignoring censoring:** Using standard methods inappropriate for survival data
2. **Informative censoring:** Assuming censoring mechanism is non-informative
3. **Proportional hazards violation:** Not checking key assumption
4. **Time-scale choice:** Analysis time vs calendar time can affect conclusions
5. **Competing risks:** Treating competing events as censoring

## Practical Guidelines

### Study Design
1. **Define endpoints clearly:** What constitutes an event?
2. **Plan follow-up:** Minimize loss to follow-up
3. **Sample size calculation:** Account for censoring patterns
4. **Monitor assumptions:** Check proportional hazards regularly
5. **Handle missing data:** Use appropriate imputation methods

### Analysis Strategy
1. **Descriptive analysis:** Plot Kaplan-Meier curves first
2. **Check assumptions:** Graphical and formal tests
3. **Model building:** Start simple, add complexity gradually
4. **Validate models:** Internal and external validation
5. **Clinical interpretation:** Focus on clinically meaningful effects

## Connections to Other Chapters

### To Chapter 8 (CDF Estimation)
- Kaplan-Meier as empirical CDF with censoring
- Nonparametric estimation principles
- Confidence bands construction

### To Chapter 14-16 (Regression)
- Cox model as regression extension
- Proportional hazards vs proportional odds
- Model building and diagnostics

### To Chapter 18 (Time Series)
- Counting processes and martingales
- Time-varying coefficients
- Recurrent events as point processes

### To Chapter 12 (Bayesian Inference)
- Bayesian survival models
- Prior specification strategies
- MCMC for complex survival models

This chapter provides comprehensive coverage of survival analysis, essential for analyzing time-to-event data across medical, engineering, and business applications.