# Chapter 14: Linear Regression - Mathematical Explanations

## Overview
Linear regression is a fundamental statistical method for modeling the relationship between a response variable and one or more predictor variables. This chapter covers simple and multiple linear regression, estimation methods, inference, and model diagnostics.

## 14.1 Introduction to Regression

### Regression Function
The **regression function** describes the relationship between response Y and covariate X:
```
r(x) = E[Y|X = x] = ∫ y f(y|x) dy
```

**Goal:** Estimate r(x) from data (Y₁, X₁), ..., (Yₙ, Xₙ).

### Types of Regression
- **Parametric:** Assume specific functional form for r(x)
- **Nonparametric:** Estimate r(x) without assuming functional form
- **Semiparametric:** Mix of parametric and nonparametric components

## 14.2 Simple Linear Regression

### Model
```
Yᵢ = β₀ + β₁Xᵢ + εᵢ,  i = 1, ..., n
```

**Assumptions:**
1. **Linearity:** E[Y|X = x] = β₀ + β₁x
2. **Independence:** ε₁, ..., εₙ are independent
3. **Homoscedasticity:** Var(εᵢ) = σ² (constant)
4. **Normality:** εᵢ ~ N(0, σ²) (for inference)

### Matrix Form
```
Y = Xβ + ε
```

where:
```
Y = [Y₁]    X = [1  X₁]    β = [β₀]    ε = [ε₁]
    [⋮ ]        [⋮  ⋮ ]        [β₁]        [⋮ ]
    [Yₙ]        [1  Xₙ]                    [εₙ]
```

## 14.3 Least Squares Estimation

### Ordinary Least Squares (OLS)
Minimize the sum of squared residuals:
```
S(β₀, β₁) = Σᵢ (Yᵢ - β₀ - β₁Xᵢ)²
```

### Normal Equations
Taking derivatives and setting to zero:
```
∂S/∂β₀ = -2Σᵢ(Yᵢ - β₀ - β₁Xᵢ) = 0
∂S/∂β₁ = -2Σᵢ(Yᵢ - β₀ - β₁Xᵢ)Xᵢ = 0
```

### OLS Estimators
```
β̂₁ = Σᵢ(Xᵢ - X̄)(Yᵢ - Ȳ) / Σᵢ(Xᵢ - X̄)² = SₓY / SₓₓX̄)ȲSₓₓ

β̂₀ = Ȳ - β̂₁
```

where:
- SₓY = Σᵢ(Xᵢ - X̄)(Yᵢ - Ȳ) (sample covariance)
- Sₓₓ = Σᵢ(Xᵢ - X̄)² (sample variance of X)

### Matrix Form
```
β̂ = (XᵀX)⁻¹XᵀY
```

**Fitted values:** Ŷ = Xβ̂ = X(XᵀX)⁻¹XᵀY = HY

**Hat matrix:** H = X(XᵀX)⁻¹Xᵀ

**Residuals:** ê = Y - Ŷ = (I - H)Y

## 14.4 Properties of OLS Estimators

### Unbiasedness
Under assumptions 1-3:
```
E[β̂] = β
```

**Proof:** E[β̂] = E[(XᵀX)⁻¹XᵀY] = (XᵀX)⁻¹XᵀE[Y] = (XᵀX)⁻¹XᵀXβ = β

### Variance
```
Var(β̂) = σ²(XᵀX)⁻¹
```

**For simple regression:**
```
Var(β̂₁) = σ²/Sₓₓ
Var(β̂₀) = σ²(1/n + X̄²/Sₓₓ)
```

### Gauss-Markov Theorem
Under assumptions 1-3, OLS estimators are **BLUE** (Best Linear Unbiased Estimators):
- **Best:** Minimum variance among all linear unbiased estimators
- **Linear:** Linear in Y
- **Unbiased:** E[β̂] = β

## 14.5 Estimation of σ²

### Residual Sum of Squares
```
RSS = Σᵢ êᵢ² = êᵀê = (Y - Xβ̂)ᵀ(Y - Xβ̂)
```

### Unbiased Estimator
```
σ̂² = RSS/(n - p)
```

where p = number of parameters.

**Degrees of freedom:** n - p

**Proof of unbiasedness:**
```
E[RSS] = E[εᵀ(I - H)ε] = σ²tr(I - H) = σ²(n - p)
```

## 14.6 Inference for Regression Parameters

### Sampling Distributions
Under normality assumption:
```
β̂ ~ N(β, σ²(XᵀX)⁻¹)
(n - p)σ̂²/σ² ~ χ²ₙ₋ₚ
β̂ and σ̂² are independent
```

### t-tests for Individual Parameters
```
t = (β̂ⱼ - βⱼ)/ŝe(β̂ⱼ) ~ tₙ₋ₚ
```

where ŝe(β̂ⱼ) = σ̂√[(XᵀX)⁻¹]ⱼⱼ

**Confidence interval:**
```
β̂ⱼ ± tₙ₋ₚ,α/2 · ŝe(β̂ⱼ)
```

### F-test for Overall Significance
Testing H₀: β₁ = ... = βₚ₋₁ = 0

**Test statistic:**
```
F = (RSS₀ - RSS₁)/(p - 1) / (RSS₁/(n - p)) ~ Fₚ₋₁,ₙ₋ₚ
```

where RSS₀ = total sum of squares, RSS₁ = residual sum of squares.

**Equivalently:**
```
F = MSR/MSE = (TSS - RSS)/(p - 1) / (RSS/(n - p))
```

## 14.7 Analysis of Variance (ANOVA)

### ANOVA Table
| Source     | df    | Sum of Squares | Mean Square | F-ratio    |
|------------|-------|----------------|-------------|------------|
| Regression | p-1   | TSS - RSS      | MSR         | MSR/MSE    |
| Error      | n-p   | RSS            | MSE         |            |
| Total      | n-1   | TSS            |             |            |

where:
- TSS = Σᵢ(Yᵢ - Ȳ)² (Total Sum of Squares)
- RSS = Σᵢ(Yᵢ - Ŷᵢ)² (Residual Sum of Squares)
- MSR = (TSS - RSS)/(p-1) (Mean Square Regression)
- MSE = RSS/(n-p) (Mean Square Error)

### Coefficient of Determination
```
R² = (TSS - RSS)/TSS = 1 - RSS/TSS
```

**Interpretation:** Proportion of variance in Y explained by X.

**Adjusted R²:**
```
R²ₐⱼ = 1 - (RSS/(n-p))/(TSS/(n-1)) = 1 - (1-R²)(n-1)/(n-p)
```

## 14.8 Multiple Linear Regression

### Model
```
Yᵢ = β₀ + β₁X₁ᵢ + ... + βₚ₋₁Xₚ₋₁,ᵢ + εᵢ
```

### Matrix Formulation
```
Y = Xβ + ε
```

where X is n × p design matrix.

### Estimation
Same as simple regression:
```
β̂ = (XᵀX)⁻¹XᵀY
σ̂² = RSS/(n - p)
```

### Interpretation of Coefficients
β̂ⱼ = expected change in Y for one-unit increase in Xⱼ, **holding all other variables constant**.

## 14.9 Multicollinearity

### Definition
**Multicollinearity** occurs when predictor variables are highly correlated.

### Detection
1. **Correlation matrix:** High pairwise correlations
2. **Variance Inflation Factor (VIF):**
   ```
   VIFⱼ = 1/(1 - R²ⱼ)
   ```
   where R²ⱼ is from regressing Xⱼ on other predictors.

**Rule of thumb:** VIF > 10 indicates problematic multicollinearity.

### Consequences
- Large standard errors for β̂
- Unstable estimates
- Difficulty interpreting individual coefficients

### Solutions
- Remove redundant variables
- Principal component regression
- Ridge regression
- Collect more data

## 14.10 Model Diagnostics

### Residual Analysis

**Standardized residuals:**
```
rᵢ = êᵢ/σ̂
```

**Studentized residuals:**
```
tᵢ = êᵢ/(σ̂√(1 - hᵢᵢ))
```

where hᵢᵢ is the i-th diagonal element of H.

### Diagnostic Plots

1. **Residuals vs Fitted:** Check linearity and homoscedasticity
2. **Q-Q plot:** Check normality of residuals
3. **Scale-location plot:** Check homoscedasticity
4. **Residuals vs Leverage:** Identify influential observations

### Influential Observations

**Leverage:**
```
hᵢᵢ = xᵢᵀ(XᵀX)⁻¹xᵢ
```

**High leverage:** hᵢᵢ > 2p/n

**Cook's distance:**
```
Dᵢ = (1/p) · (rᵢ²/(1-hᵢᵢ)) · (hᵢᵢ/(1-hᵢᵢ))
```

**Influential:** Dᵢ > 1

## 14.11 Assumption Violations and Remedies

### Non-linearity
**Detection:** Residual plots, added variable plots
**Remedies:** 
- Transform variables
- Add polynomial terms
- Use nonlinear regression

### Heteroscedasticity
**Detection:** 
- Residual vs fitted plots
- Breusch-Pagan test
- White test

**Remedies:**
- Transform response variable
- Weighted least squares
- Robust standard errors

### Non-normality
**Detection:** Q-Q plots, Shapiro-Wilk test
**Remedies:**
- Transform variables
- Use robust methods
- Bootstrap inference

### Autocorrelation (Time Series)
**Detection:** Durbin-Watson test
**Remedies:**
- Add lagged variables
- Generalized least squares
- Time series methods

## 14.12 Variable Selection

### All Subsets Regression
Evaluate all 2ᵖ possible models using criteria like:
- R²ₐⱼ (maximize)
- AIC (minimize)
- BIC (minimize)
- Mallows' Cₚ (minimize)

### Stepwise Methods

**Forward selection:**
1. Start with null model
2. Add variable that most improves fit
3. Stop when no improvement

**Backward elimination:**
1. Start with full model
2. Remove least significant variable
3. Stop when all remaining variables significant

**Bidirectional:** Combine forward and backward

### Information Criteria

**AIC:**
```
AIC = n ln(RSS/n) + 2p
```

**BIC:**
```
BIC = n ln(RSS/n) + p ln(n)
```

**Mallows' Cₚ:**
```
Cₚ = RSS/σ̂² - n + 2p
```

## 14.13 Prediction

### Point Prediction
For new observation x₀:
```
Ŷ₀ = x₀ᵀβ̂
```

### Prediction Intervals

**Mean response at x₀:**
```
Var(Ŷ₀) = σ²x₀ᵀ(XᵀX)⁻¹x₀
```

**New observation at x₀:**
```
Var(Y₀ - Ŷ₀) = σ²(1 + x₀ᵀ(XᵀX)⁻¹x₀)
```

**Prediction interval:**
```
Ŷ₀ ± tₙ₋ₚ,α/2 · σ̂√(1 + x₀ᵀ(XᵀX)⁻¹x₀)
```

## 14.14 Regularization Methods

### Ridge Regression
Minimize:
```
RSS + λΣⱼβⱼ²
```

**Solution:**
```
β̂ᴿⁱᵈᵍᵉ = (XᵀX + λI)⁻¹XᵀY
```

**Properties:**
- Shrinks coefficients toward zero
- Handles multicollinearity
- All variables retained

### Lasso Regression
Minimize:
```
RSS + λΣⱼ|βⱼ|
```

**Properties:**
- Can set coefficients exactly to zero
- Performs variable selection
- Solution path is piecewise linear

### Elastic Net
Minimize:
```
RSS + λ₁Σⱼ|βⱼ| + λ₂Σⱼβⱼ²
```

Combines ridge and lasso penalties.

## 14.15 Cross-Validation

### k-fold Cross-Validation
1. Divide data into k folds
2. For each fold:
   - Train on k-1 folds
   - Test on remaining fold
3. Average prediction errors

### Leave-One-Out Cross-Validation
Special case with k = n.

**LOOCV for linear regression:**
```
CV = (1/n)Σᵢ(Yᵢ - Ŷ₍₋ᵢ₎)² = (1/n)Σᵢ(êᵢ/(1-hᵢᵢ))²
```

## 14.16 Logistic Regression

### Model
For binary response Y ∈ {0, 1}:
```
logit(p) = ln(p/(1-p)) = β₀ + β₁X₁ + ... + βₚ₋₁Xₚ₋₁
```

**Probability:**
```
p = exp(xᵀβ)/(1 + exp(xᵀβ))
```

### Estimation
Use maximum likelihood estimation (no closed form).

### Interpretation
- βⱼ = log odds ratio for one-unit increase in Xⱼ
- exp(βⱼ) = odds ratio

## 14.17 Generalized Linear Models

### Components
1. **Random component:** Y follows exponential family
2. **Systematic component:** η = Xβ (linear predictor)
3. **Link function:** g(μ) = η where μ = E[Y]

### Common GLMs
- **Normal:** Identity link, g(μ) = μ
- **Binomial:** Logit link, g(μ) = ln(μ/(1-μ))
- **Poisson:** Log link, g(μ) = ln(μ)
- **Gamma:** Inverse link, g(μ) = 1/μ

## 14.18 Model Comparison and Selection

### Nested Models
Use likelihood ratio test:
```
-2(ℓ₀ - ℓ₁) ~ χ²ₖ
```

where k = difference in number of parameters.

### Non-nested Models
Use information criteria (AIC, BIC) or cross-validation.

### Model Averaging
When uncertain about model choice:
```
Ŷ = Σⱼ wⱼŶⱼ
```

where wⱼ are model weights based on AIC, BIC, or Bayesian methods.

## 14.19 Robust Regression

### Outlier Problems
- **Outliers in Y:** High residuals
- **Outliers in X:** High leverage
- **Influential points:** Both high residual and leverage

### Robust Methods

**M-estimators:** Minimize Σᵢρ(rᵢ) where ρ is robust loss function.

**Huber loss:**
```
ρ(r) = {
  r²/2     if |r| ≤ c
  c|r| - c²/2  if |r| > c
}
```

**Least Absolute Deviations (LAD):** Minimize Σᵢ|rᵢ|

## 14.20 Practical Considerations

### Sample Size
**Rule of thumb:** Need at least 10-20 observations per parameter for stable estimates.

### Model Building Strategy
1. **Exploratory analysis:** Understand relationships
2. **Model fitting:** Start simple, add complexity
3. **Diagnostics:** Check assumptions
4. **Validation:** Test on new data

### Reporting Results
- Effect sizes and confidence intervals
- Model fit statistics (R², AIC, BIC)
- Assumption checks
- Sensitivity analysis

## Key Insights

1. **Linear Models:** Foundation for many statistical methods
2. **Least Squares:** Optimal under Gauss-Markov conditions
3. **Assumptions Matter:** Violations can invalidate inference
4. **Diagnostics:** Essential for model validation
5. **Regularization:** Important for high-dimensional problems

## Common Pitfalls

1. **Correlation ≠ Causation:** Regression shows association, not causation
2. **Extrapolation:** Predictions outside data range unreliable
3. **Model Selection:** Multiple testing issues with stepwise methods
4. **Overfitting:** Complex models may not generalize
5. **Assumption violations:** Can lead to biased or inefficient estimates

## Connections to Other Topics

- **Chapter 11:** Hypothesis testing for regression parameters
- **Chapter 12:** Bayesian regression methods
- **Chapter 23:** Relationship to classification methods
- **Machine Learning:** Foundation for many supervised learning methods

This chapter provides the foundation for most of applied statistics and serves as a gateway to more advanced modeling techniques.