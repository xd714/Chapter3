# Chapter 18: Time Series Analysis - Mathematical Explanations

## Overview
Time series analysis deals with data collected sequentially over time. This chapter covers fundamental concepts of temporal dependence, stationarity, autoregressive and moving average models, and modern approaches to forecasting and inference with dependent data.

## 18.1 Introduction to Time Series

### Definition
A **time series** is a sequence of observations {X₁, X₂, ..., Xₙ} indexed by time t.

**Examples:**
- Stock prices over days
- Temperature measurements over hours
- GDP growth over quarters
- Heart rate measurements over seconds

### Key Characteristics
1. **Temporal dependence:** Observations are not independent
2. **Trend:** Long-term increase or decrease
3. **Seasonality:** Regular periodic patterns
4. **Autocorrelation:** Correlation with past values
5. **Heteroscedasticity:** Changing variance over time

### Time Series vs Cross-sectional Data
**Cross-sectional:** Independent observations at single time point
**Time series:** Dependent observations over multiple time points
**Panel data:** Multiple time series (cross-sectional + temporal)

## 18.2 Stationarity

### Strict Stationarity
Process {Xₜ} is **strictly stationary** if for any finite collection of time points and any lag h:
```
(X_{t₁}, X_{t₂}, ..., X_{tₖ}) ≡ (X_{t₁+h}, X_{t₂+h}, ..., X_{tₖ+h})
```

**Interpretation:** Statistical properties unchanged by time shifts.

### Weak Stationarity (Second-order Stationarity)
Process {Xₜ} is **weakly stationary** if:
1. **Constant mean:** E[Xₜ] = μ for all t
2. **Finite variance:** Var(Xₜ) = σ² < ∞ for all t  
3. **Autocovariance depends only on lag:** Cov(Xₜ, Xₜ₊ₕ) = γ(h)

### Autocovariance Function
```
γ(h) = Cov(Xₜ, Xₜ₊ₗ) = E[(Xₜ - μ)(Xₜ₊ₗ - μ)]
```

**Properties:**
- γ(0) = Var(Xₜ) = σ²
- γ(h) = γ(-h) (symmetry)
- |γ(h)| ≤ γ(0) (Cauchy-Schwarz)

### Autocorrelation Function (ACF)
```
ρ(h) = γ(h)/γ(0) = Cov(Xₜ, Xₜ₊ₗ)/√(Var(Xₜ)Var(Xₜ₊ₗ))
```

**Properties:**
- ρ(0) = 1
- ρ(h) = ρ(-h)
- |ρ(h)| ≤ 1

### Sample Autocorrelation Function
```
r(h) = ĉ(h)/ĉ(0)
```

where:
```
ĉ(h) = (1/n)∑ₜ₌₁ⁿ⁻ʰ (Xₜ - X̄)(Xₜ₊ₗ - X̄)
```

## 18.3 White Noise and Random Walks

### White Noise
{εₜ} is **white noise** if:
1. E[εₜ] = 0
2. Var(εₜ) = σ²
3. Cov(εₜ, εₛ) = 0 for t ≠ s

**Gaussian white noise:** εₜ ~ iid N(0, σ²)

### Random Walk
```
Xₜ = Xₜ₋₁ + εₜ
```

**Solution:** Xₜ = X₀ + ∑ᵢ₌₁ᵗ εᵢ

**Properties:**
- E[Xₜ] = X₀
- Var(Xₜ) = tσ² (increases with time)
- **Non-stationary:** Variance depends on t

### Random Walk with Drift
```
Xₜ = δ + Xₜ₋₁ + εₜ
```

**Solution:** Xₜ = X₀ + δt + ∑ᵢ₌₁ᵗ εᵢ

**Properties:**
- E[Xₜ] = X₀ + δt (linear trend)
- Var(Xₜ) = tσ²

## 18.4 Autoregressive Models

### AR(1) Model
```
Xₜ = φXₜ₋₁ + εₜ
```

where εₜ ~ WN(0, σ²).

**Stationarity condition:** |φ| < 1

**Mean:** E[Xₜ] = 0 (assuming mean-zero process)

**Autocovariance:**
- γ(0) = σ²/(1 - φ²)
- γ(h) = φʰγ(0) = φʰσ²/(1 - φ²)

**Autocorrelation:** ρ(h) = φʰ

### AR(p) Model
```
Xₜ = φ₁Xₜ₋₁ + φ₂Xₜ₋₂ + ... + φₚXₜ₋ₚ + εₜ
```

**Lag operator notation:**
```
φ(L)Xₜ = εₜ
```

where φ(L) = 1 - φ₁L - φ₂L² - ... - φₚLᵖ

**Stationarity:** All roots of φ(z) = 0 lie outside unit circle.

### Yule-Walker Equations
For AR(p), autocorrelations satisfy:
```
ρ(h) = φ₁ρ(h-1) + φ₂ρ(h-2) + ... + φₚρ(h-p)
```

for h > 0.

**Matrix form:**
```
Γρ = φ
```

where Γ is p×p autocorrelation matrix.

### Partial Autocorrelation Function (PACF)
**Definition:** φₖₖ is correlation between Xₜ and Xₜ₊ₖ after removing linear dependence on X_{t+1}, ..., X_{t+k-1}.

**For AR(p):**
- φₖₖ ≠ 0 for k ≤ p
- φₖₖ = 0 for k > p

**Identification:** PACF cuts off after lag p for AR(p) process.

## 18.5 Moving Average Models

### MA(1) Model
```
Xₜ = εₜ + θεₜ₋₁
```

**Always stationary** (finite sum of white noise).

**Autocovariance:**
- γ(0) = (1 + θ²)σ²
- γ(1) = θσ²
- γ(h) = 0 for h > 1

**Autocorrelation:**
- ρ(1) = θ/(1 + θ²)
- ρ(h) = 0 for h > 1

### MA(q) Model
```
Xₜ = εₜ + θ₁εₜ₋₁ + θ₂εₜ₋₂ + ... + θqεₜ₋q
```

**ACF:** ρ(h) = 0 for h > q (cuts off after lag q)

**Identification:** ACF cuts off after lag q for MA(q) process.

### Invertibility
MA model is **invertible** if it can be written as infinite AR:
```
π(L)Xₜ = εₜ
```

**Condition:** All roots of θ(z) = 0 lie outside unit circle.

**Canonical representation:** Require invertibility for uniqueness.

## 18.6 ARMA Models

### ARMA(p,q) Model
```
φ(L)Xₜ = θ(L)εₜ
```

or equivalently:
```
Xₜ = φ₁Xₜ₋₁ + ... + φₚXₜ₋ₚ + εₜ + θ₁εₜ₋₁ + ... + θqεₜ₋q
```

**Conditions:**
- **Stationarity:** φ(z) = 0 has roots outside unit circle
- **Invertibility:** θ(z) = 0 has roots outside unit circle

### Autocovariance Structure
**Complex pattern:** Neither ACF nor PACF cuts off cleanly.

**ACF:** Exponential decay for h > q
**PACF:** Exponential decay for h > p

### Wold Decomposition
Any stationary process can be written as:
```
Xₜ = ∑ⱼ₌₀^∞ ψⱼεₜ₋ⱼ + Vₜ
```

where Vₜ is deterministic and ∑ψⱼ² < ∞.

**Interpretation:** Any stationary process is MA(∞) plus deterministic part.

## 18.7 Non-stationary Models

### Integration and Differencing
Time series Xₜ is **integrated of order d**, denoted I(d), if:
- Xₜ is non-stationary
- ∇ᵈXₜ is stationary

where ∇ is difference operator: ∇Xₜ = Xₜ - Xₜ₋₁

### ARIMA Models
**ARIMA(p,d,q):** Apply ARMA(p,q) to d-th difference:
```
φ(L)∇ᵈXₜ = θ(L)εₜ
```

**Common case:** ARIMA(p,1,q) for I(1) processes:
```
φ(L)(Xₜ - Xₜ₋₁) = θ(L)εₜ
```

### Unit Root Tests
**Augmented Dickey-Fuller Test:**
```
∇Xₜ = α + βXₜ₋₁ + ∑ᵢ₌₁ᵖ γᵢ∇Xₜ₋ᵢ + εₜ
```

**Hypotheses:**
- H₀: β = 0 (unit root, non-stationary)
- H₁: β < 0 (stationary)

**Test statistic:** t-ratio for β̂ (non-standard distribution under H₀)

### Phillips-Perron Test
**Non-parametric correction** for serial correlation and heteroscedasticity.

**KPSS Test:** Reverse hypotheses
- H₀: Stationary
- H₁: Unit root

## 18.8 Seasonal Models

### Seasonal Patterns
**Deterministic seasonality:** Fixed seasonal pattern
**Stochastic seasonality:** Evolving seasonal pattern

### Seasonal ARIMA Models
**SARIMA(p,d,q)×(P,D,Q)ₛ:**
```
φ(L)Φ(Lˢ)∇ᵈ∇ₛᴰXₜ = θ(L)Θ(Lˢ)εₜ
```

where:
- s = seasonal period
- ∇ₛ = seasonal difference operator: ∇ₛXₜ = Xₜ - Xₜ₋ₛ
- Φ(Lˢ), Θ(Lˢ) = seasonal AR and MA polynomials

### Example: SARIMA(0,1,1)×(0,1,1)₁₂
```
∇∇₁₂Xₜ = (1 + θL)(1 + ΘL¹²)εₜ
```

**Interpretation:** Monthly data with both regular and seasonal differencing.

## 18.9 Estimation Methods

### Method of Moments
**Yule-Walker estimation** for AR(p):
1. Estimate sample autocorrelations r(1), ..., r(p)
2. Solve Yule-Walker equations:
   ```
   Γ̂φ̂ = ρ̂
   ```
3. Estimate σ²: σ̂² = ĉ(0)(1 - φ̂ᵀρ̂)

### Maximum Likelihood Estimation
**Likelihood for ARMA(p,q):**
- **Exact likelihood:** Complex but optimal
- **Conditional likelihood:** Condition on initial values
- **Gaussian assumption:** εₜ ~ iid N(0, σ²)

**Optimization:** Use numerical methods (Newton-Raphson, BFGS)

### Least Squares Methods
**Conditional least squares:** Minimize
```
∑ₜ₌ₚ₊₁ⁿ εₜ²(φ, θ)
```

**Non-linear optimization** required for MA and ARMA models.

### State Space Methods
**Kalman filter:** Recursive estimation for linear state space models
```
Xₜ = AXₜ₋₁ + Bεₜ    (state equation)
Yₜ = CXₜ + Dηₜ       (observation equation)
```

**Advantages:**
- Handles missing data
- Provides filtered and smoothed estimates
- Likelihood evaluation

## 18.10 Model Identification and Diagnostic Checking

### Box-Jenkins Methodology
1. **Identification:** Use ACF/PACF to identify model order
2. **Estimation:** Fit parameters using MLE or least squares  
3. **Diagnostic checking:** Examine residuals
4. **Forecasting:** Generate predictions

### Model Identification Guidelines
**AR(p):** PACF cuts off at lag p, ACF decays
**MA(q):** ACF cuts off at lag q, PACF decays
**ARMA(p,q):** Both ACF and PACF decay

### Information Criteria
**AIC:** AIC = -2ℓ + 2k
**BIC:** BIC = -2ℓ + k log(n)
**AICc:** Small sample correction

**Model selection:** Choose model minimizing criterion.

### Residual Diagnostics
**Residuals:** êₜ = observed - fitted

**Tests:**
1. **Ljung-Box test:** H₀: residuals are white noise
   ```
   Q = n(n+2)∑ₕ₌₁ʰ r²ₑ(h)/(n-h) ~ χ²ₕ
   ```

2. **ARCH test:** H₀: constant variance in residuals

3. **Normality tests:** Jarque-Bera, Shapiro-Wilk

## 18.11 Forecasting

### Point Forecasts
**h-step ahead forecast:** X̂ₙ₊ₕ|ₙ = E[Xₙ₊ₕ|X₁, ..., Xₙ]

**Minimum mean squared error property:**
```
X̂ₙ₊ₕ|ₙ = argmin E[(Xₙ₊ₕ - f)²|X₁, ..., Xₙ]
```

### Forecast for ARMA Models
**AR(1):** X̂ₙ₊ₕ|ₙ = φʰXₙ

**MA(1):** 
- X̂ₙ₊₁|ₙ = θεₙ
- X̂ₙ₊ₕ|ₙ = 0 for h > 1

**ARMA(p,q):** Use recursion with estimated parameters.

### Forecast Variance
**h-step ahead forecast error:** eₙ₊ₕ|ₙ = Xₙ₊ₕ - X̂ₙ₊ₕ|ₙ

**Variance:**
```
Var(eₙ₊ₗ|ₙ) = σ²(1 + ψ₁² + ... + ψₕ₋₁²)
```

where ψⱼ are MA representation coefficients.

### Prediction Intervals
**Gaussian assumption:**
```
X̂ₙ₊ₕ|ₙ ± z_{α/2}√Var(eₙ₊ₗ|ₙ)
```

### Forecast Evaluation
**Metrics:**
- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error  
- **MAPE:** Mean Absolute Percentage Error

**Out-of-sample testing:** Reserve data for forecast evaluation.

## 18.12 Multivariate Time Series

### Vector Autoregression (VAR)
**VAR(p) model:**
```
Xₜ = A₁Xₜ₋₁ + A₂Xₜ₋₂ + ... + AₚXₜ₋ₚ + εₜ
```

where Xₜ is k×1 vector, Aᵢ are k×k coefficient matrices.

**Stationarity:** Eigenvalues of companion matrix inside unit circle.

### Granger Causality
Y **Granger-causes** X if past values of Y help predict X beyond what X's own past values provide.

**Test:** F-test in regression:
```
Xₜ = ∑ᵢ₌₁ᵖ αᵢXₜ₋ᵢ + ∑ⱼ₌₁ᵍ βⱼYₜ₋ⱼ + εₜ
```

H₀: β₁ = ... = βᵍ = 0

### Cointegration
Variables are **cointegrated** if:
1. Individual series are I(1)
2. Linear combination is I(0)

**Error Correction Model (ECM):**
```
∇Xₜ = αβ'Xₜ₋₁ + Γ₁∇Xₜ₋₁ + ... + Γₚ₋₁∇Xₜ₋ₚ₊₁ + εₜ
```

where β'Xₜ₋₁ is error correction term.

### Vector Error Correction Model (VECM)
**Johansen test:** Test for number of cointegrating relationships.

**Rank of Π = αβ':** Number of cointegrating vectors.

## 18.13 Volatility Modeling

### ARCH Models
**ARCH(q):** Autoregressive Conditional Heteroscedasticity
```
Xₜ = μₜ + εₜ
εₜ = σₜzₜ, zₜ ~ iid N(0,1)
σₜ² = α₀ + α₁εₜ₋₁² + ... + αᵧεₜ₋ᵧ²
```

**Properties:**
- E[εₜ|ℱₜ₋₁] = 0
- Var(εₜ|ℱₜ₋₁) = σₜ²
- Unconditional variance > conditional variance

### GARCH Models
**GARCH(p,q):** Generalized ARCH
```
σₜ² = α₀ + ∑ᵢ₌₁ᵍ αᵢεₜ₋ᵢ² + ∑ⱼ₌₁ᵖ βⱼσₜ₋ⱼ²
```

**GARCH(1,1):** Most popular specification
```
σₜ² = α₀ + α₁εₜ₋₁² + β₁σₜ₋₁²
```

**Stationarity:** α₁ + β₁ < 1

### Extensions
**EGARCH:** Exponential GARCH (asymmetric effects)
**TGARCH:** Threshold GARCH
**GJR-GARCH:** Glosten-Jagannathan-Runkle GARCH

### Estimation
**Maximum likelihood:** Assume conditional normality
**Quasi-MLE:** Robust to distributional assumptions

## 18.14 State Space Models

### Linear State Space Form
**State equation:** αₜ₊₁ = Tₜαₜ + Rₜηₜ
**Observation equation:** yₜ = Zₜαₜ + εₜ

**Kalman Filter:**
1. **Prediction:** α̂ₜ|ₜ₋₁ = Tₜα̂ₜ₋₁|ₜ₋₁
2. **Updating:** α̂ₜ|ₜ = α̂ₜ|ₜ₋₁ + Kₜvₜ

where Kₜ is Kalman gain.

### Local Level Model
```
yₜ = μₜ + εₜ
μₜ₊₁ = μₜ + ηₜ
```

**Random walk with noise:** Level evolves as random walk.

### Structural Time Series Models
**Local linear trend:**
```
yₜ = μₜ + εₜ
μₜ₊₁ = μₜ + βₜ + η₁ₜ  
βₜ₊₁ = βₜ + η₂ₜ
```

**Seasonal component:**
```
γₜ₊₁ = -γₜ - γₜ₋₁ - ... - γₜ₋ₛ₊₂ + ωₜ
```

## 18.15 Spectral Analysis

### Spectral Density
**Definition:** Fourier transform of autocovariance function
```
f(ω) = (1/2π) ∑ₕ₌₋∞^∞ γ(h)e^{-iωh}
```

**Interpretation:** Variance contribution at frequency ω.

### Periodogram
**Sample spectral density:**
```
I(ωⱼ) = (1/2πn)|∑ₜ₌₁ⁿ Xₜe^{-iωⱼt}|²
```

where ωⱼ = 2πj/n are Fourier frequencies.

### Smoothed Spectral Estimators
**Bartlett's method:** Average periodograms over frequency bands
**Welch's method:** Average periodograms from overlapping segments

### Applications
- **Identify periodicities:** Peaks in spectral density
- **Filter design:** Remove specific frequency components
- **Signal processing:** Decompose into frequency components

## Key Insights

1. **Temporal Dependence:** Time series require specialized methods due to dependence structure.

2. **Stationarity:** Fundamental assumption for most time series methods.

3. **Parsimony:** Simple models often forecast as well as complex ones.

4. **Model Uncertainty:** Consider multiple models rather than single "best" model.

5. **Diagnostic Checking:** Essential to verify model adequacy.

## Common Pitfalls

1. **Spurious regression:** Regressing non-stationary series gives misleading results
2. **Overfitting:** Including too many parameters relative to sample size
3. **Ignoring structural breaks:** Model parameters may change over time
4. **Extrapolation:** Forecasting far beyond sample period unreliable
5. **Model selection bias:** Using same data for identification and testing

## Practical Guidelines

### Model Building
1. **Plot the data:** Visual inspection reveals patterns
2. **Check stationarity:** Use unit root tests
3. **Transform if needed:** Differencing, logging
4. **Identify model:** Use ACF/PACF patterns
5. **Estimate and diagnose:** Check residuals thoroughly

### Forecasting
1. **Out-of-sample evaluation:** Test on hold-out data
2. **Model averaging:** Combine forecasts from multiple models  
3. **Prediction intervals:** Quantify forecast uncertainty
4. **Update regularly:** Re-estimate as new data arrive
5. **Monitor performance:** Track forecast accuracy over time

## Connections to Other Chapters

### To Chapter 6 (Convergence)
- Limit theorems for dependent sequences
- Ergodic theory and time series
- Central limit theorems for stationary processes

### To Chapter 14-15 (Regression)
- Time series regression models
- Spurious regression problem
- Dynamic regression models

### To Chapter 12 (Bayesian Inference)
- Bayesian time series models
- State space models with MCMC
- Dynamic linear models

This chapter provides comprehensive coverage of time series analysis, essential for understanding and modeling temporal dependence in sequential data across numerous applications.