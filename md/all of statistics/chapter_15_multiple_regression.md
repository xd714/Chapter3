# Chapter 15: Multiple Regression and Model Building - Mathematical Explanations

## Overview
Multiple regression extends simple linear regression to handle multiple predictor variables simultaneously. This chapter covers advanced regression techniques, model building strategies, variable selection methods, and modern approaches to handling high-dimensional data.

## 15.1 Multiple Linear Regression Model

### Model Specification
The multiple linear regression model is:
```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚX₁ + ε
```

**Matrix form:**
```
Y = Xβ + ε
```

where:
- Y is n×1 response vector
- X is n×(p+1) design matrix with first column of 1's
- β is (p+1)×1 parameter vector
- ε is n×1 error vector

### Model Assumptions
1. **Linearity:** E[Y|X] = Xβ
2. **Independence:** εᵢ are independent
3. **Homoscedasticity:** Var(εᵢ) = σ² (constant)
4. **Normality:** εᵢ ~ N(0, σ²) (for inference)
5. **Full rank:** rank(X) = p+1 (no perfect multicollinearity)

## 15.2 Least Squares Estimation

### Normal Equations
Minimize sum of squared residuals:
```
RSS(β) = (Y - Xβ)ᵀ(Y - Xβ)
```

**Solution:**
```
β̂ = (XᵀX)⁻¹XᵀY
```

### Fitted Values and Residuals
```
Ŷ = Xβ̂ = X(XᵀX)⁻¹XᵀY = HY
ê = Y - Ŷ = (I - H)Y
```

**Hat matrix:** H = X(XᵀX)⁻¹Xᵀ
- Projects Y onto column space of X
- H is idempotent: H² = H
- tr(H) = p+1

### Properties of OLS Estimators

**Unbiasedness:** E[β̂] = β

**Variance-covariance matrix:**
```
Var(β̂) = σ²(XᵀX)⁻¹
```

**Gauss-Markov theorem:** β̂ is BLUE (Best Linear Unbiased Estimator)

## 15.3 Statistical Inference

### Sampling Distributions
Under normality assumption:
```
β̂ ~ N(β, σ²(XᵀX)⁻¹)
(n-p-1)σ̂²/σ² ~ χ²ₙ₋ₚ₋₁
β̂ and σ̂² are independent
```

### Individual Parameter Tests
```
t = (β̂ⱼ - βⱼ)/se(β̂ⱼ) ~ tₙ₋ₚ₋₁
```

where se(β̂ⱼ) = σ̂√[(XᵀX)⁻¹]ⱼⱼ

**Confidence interval:**
```
β̂ⱼ ± tₙ₋ₚ₋₁,α/₂ · se(β̂ⱼ)
```

### Overall F-test
Testing H₀: β₁ = β₂ = ... = βₚ = 0

**Test statistic:**
```
F = (TSS - RSS)/p / (RSS/(n-p-1)) = MSR/MSE
```

Under H₀: F ~ Fₚ,ₙ₋ₚ₋₁

### Partial F-tests
Testing subset of parameters H₀: βⱼ₁ = ... = βⱼₖ = 0

**Test statistic:**
```
F = (RSS_reduced - RSS_full)/k / (RSS_full/(n-p-1))
```

## 15.4 Model Diagnostics

### Residual Analysis

**Standardized residuals:**
```
rᵢ = êᵢ/σ̂
```

**Studentized residuals:**
```
tᵢ = êᵢ/(σ̂√(1-hᵢᵢ))
```

**Externally studentized residuals:**
```
t*ᵢ = êᵢ/(σ̂₍ᵢ₎√(1-hᵢᵢ))
```

where σ̂₍ᵢ₎ is estimated without observation i.

### Diagnostic Plots

**Residuals vs fitted:** Check linearity and homoscedasticity
**Q-Q plot:** Check normality assumption
**Scale-location plot:** Check homoscedasticity
**Residuals vs leverage:** Identify influential points

### Influential Observations

**Leverage:** hᵢᵢ measures how far Xᵢ is from X̄
- High leverage: hᵢᵢ > 2(p+1)/n

**Cook's distance:**
```
Dᵢ = (rᵢ²/(p+1)) · (hᵢᵢ/(1-hᵢᵢ))
```
- Influential: Dᵢ > 1

**DFBETAS:** Change in β̂ⱼ when removing observation i
**DFFITS:** Change in fitted value when removing observation i

## 15.5 Multicollinearity

### Detection
**Correlation matrix:** High pairwise correlations (|r| > 0.9)

**Variance Inflation Factor:**
```
VIFⱼ = 1/(1 - R²ⱼ)
```

where R²ⱼ is from regressing Xⱼ on other predictors.
- Problematic: VIF > 10

**Condition number:** κ = √(λₘₐₓ/λₘᵢₙ) of XᵀX
- Problematic: κ > 30

### Consequences
- Large standard errors
- Unstable parameter estimates
- Sensitivity to small data changes
- Difficulty interpreting individual coefficients

### Solutions
1. **Remove redundant variables**
2. **Principal component regression**
3. **Ridge regression**
4. **Collect more data**
5. **Center variables**

## 15.6 Variable Selection

### Best Subset Selection
Evaluate all 2ᵖ possible models using criteria:

**R²ₐⱼ:** Adjusted R-squared
```
R²ₐⱼ = 1 - (RSS/(n-p-1))/(TSS/(n-1))
```

**AIC:** Akaike Information Criterion
```
AIC = n log(RSS/n) + 2(p+1)
```

**BIC:** Bayesian Information Criterion
```
BIC = n log(RSS/n) + (p+1)log(n)
```

**Mallows' Cₚ:**
```
Cₚ = RSS/σ̂² - n + 2(p+1)
```

### Stepwise Procedures

**Forward selection:**
1. Start with intercept-only model
2. Add variable that most improves fit
3. Stop when no improvement or criterion met

**Backward elimination:**
1. Start with full model
2. Remove least significant variable
3. Stop when all remaining significant

**Bidirectional:**
Combine forward and backward at each step

### Modern Selection Methods

**LASSO (Least Absolute Shrinkage and Selection Operator):**
```
min ||Y - Xβ||² + λ||β||₁
```

**Ridge regression:**
```
min ||Y - Xβ||² + λ||β||²
```

**Elastic Net:**
```
min ||Y - Xβ||² + λ₁||β||₁ + λ₂||β||²
```

## 15.7 Regularization Methods

### Ridge Regression
**Objective function:**
```
β̂ᴿⁱᵈᵍᵉ = argmin {||Y - Xβ||² + λ||β||²}
```

**Solution:**
```
β̂ᴿⁱᵈᵍᵉ = (XᵀX + λI)⁻¹XᵀY
```

**Properties:**
- Shrinks coefficients toward zero
- Handles multicollinearity
- All variables retained
- Continuous shrinkage

### LASSO Regression
**Objective function:**
```
β̂ᴸᴬˢˢᴼ = argmin {||Y - Xβ||² + λ||β||₁}
```

**Properties:**
- Performs variable selection
- Sets some coefficients exactly to zero
- Solution path is piecewise linear
- No closed-form solution

### Cross-Validation for λ Selection
**k-fold CV:**
1. Divide data into k folds
2. For each λ:
   - Fit model on k-1 folds
   - Predict on remaining fold
3. Choose λ minimizing CV error

## 15.8 Polynomial Regression

### Model
```
Y = β₀ + β₁X + β₂X² + ... + βₚXᵖ + ε
```

### Orthogonal Polynomials
Use orthogonal basis to reduce correlation:
```
Y = α₀P₀(X) + α₁P₁(X) + ... + αₚPₚ(X) + ε
```

where Pⱼ(X) are orthogonal polynomials.

### Model Selection
- Use cross-validation
- Information criteria
- Hypothesis testing for highest-order terms

## 15.9 Interaction Effects

### Two-way Interactions
```
Y = β₀ + β₁X₁ + β₂X₂ + β₃X₁X₂ + ε
```

**Interpretation:** Effect of X₁ depends on level of X₂:
- When X₂ = 0: ∂Y/∂X₁ = β₁
- When X₂ = c: ∂Y/∂X₁ = β₁ + β₃c

### Hierarchical Principle
If interaction X₁X₂ is in model, include main effects X₁ and X₂.

### Higher-order Interactions
```
Y = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + β₄X₁X₂ + β₅X₁X₃ + β₆X₂X₃ + β₇X₁X₂X₃ + ε
```

## 15.10 Categorical Predictors

### Dummy Variables
For categorical variable with k levels, use k-1 dummy variables:
```
D₁ = 1 if category 1, 0 otherwise
D₂ = 1 if category 2, 0 otherwise
...
Dₖ₋₁ = 1 if category k-1, 0 otherwise
```

Reference category: All dummies = 0

### Coding Schemes
**Treatment coding:** Compare each level to reference
**Sum coding:** Compare each level to overall mean  
**Helmert coding:** Compare each level to mean of previous levels

### ANOVA as Regression
One-way ANOVA equivalent to regression with dummy variables:
```
Yᵢⱼ = μ + αᵢ + εᵢⱼ
```

## 15.11 Model Building Strategy

### Exploratory Data Analysis
1. **Examine distributions** of variables
2. **Check for outliers** and unusual observations
3. **Explore relationships** with scatterplot matrix
4. **Assess collinearity** with correlation matrix

### Model Specification
1. **Theory-driven approach:** Based on subject matter knowledge
2. **Data-driven approach:** Let data guide model selection
3. **Hybrid approach:** Combine theory and data exploration

### Validation
**Training-validation split:** Fit on training, evaluate on validation
**Cross-validation:** k-fold or leave-one-out
**Bootstrap validation:** Resample to assess stability

### Model Comparison
1. **Nested models:** Use F-tests or likelihood ratio tests
2. **Non-nested models:** Use information criteria
3. **Prediction accuracy:** Cross-validation error
4. **Parsimony:** Prefer simpler models when performance similar

## 15.12 Nonlinear Regression

### Polynomial Models
```
Y = β₀ + β₁X + β₂X² + ... + βₚXᵖ + ε
```

### Spline Regression
**Piecewise polynomials** with continuity constraints at knots.

**Linear spline:**
```
Y = β₀ + β₁X + β₂(X-κ)₊ + ε
```

where (x)₊ = max(0, x) and κ is knot location.

### Smoothing Splines
Minimize:
```
∑(Yᵢ - g(Xᵢ))² + λ∫(g''(x))²dx
```

### Local Regression (LOESS)
Fit local weighted polynomials:
1. For each point, define neighborhood
2. Weight nearby points more heavily
3. Fit weighted polynomial
4. Predict at point of interest

## 15.13 Robust Regression

### M-estimators
Minimize:
```
∑ρ(rᵢ) = ∑ρ((Yᵢ - Xᵢᵀβ)/σ)
```

**Huber loss:**
```
ρ(r) = {
  r²/2     if |r| ≤ k
  k|r| - k²/2  if |r| > k
}
```

### Iteratively Reweighted Least Squares
1. Start with OLS estimates
2. Compute residuals and weights
3. Perform weighted least squares
4. Iterate until convergence

### Breakdown Point
- **OLS:** 0% (one outlier can have arbitrary effect)
- **M-estimators:** ~15% depending on loss function
- **Least median squares:** 50%

## 15.14 Weighted Least Squares

### Model
```
Y = Xβ + ε, Var(ε) = σ²W⁻¹
```

where W is diagonal weight matrix.

### Estimation
```
β̂ᵂᴸˢ = (XᵀWX)⁻¹XᵀWY
```

### Applications
- **Heteroscedasticity:** Weight by inverse variance
- **Grouped data:** Weight by group size
- **Measurement error:** Weight by measurement precision

## 15.15 Generalized Least Squares

### Model
```
Y = Xβ + ε, Var(ε) = σ²V
```

where V is general covariance matrix.

### Estimation
```
β̂ᴳᴸˢ = (XᵀV⁻¹X)⁻¹XᵀV⁻¹Y
```

### Feasible GLS
When V unknown:
1. Estimate V using residuals from OLS
2. Apply GLS with estimated V̂

## 15.16 High-Dimensional Regression

### Challenge
When p > n or p ≈ n:
- OLS doesn't exist or is unstable
- Traditional inference breaks down
- Need regularization

### Sparsity Assumption
Assume only s << p coefficients are non-zero.

### LASSO Theory
Under conditions, LASSO recovers true sparse model with high probability when:
```
n ≥ c·s·log(p)
```

### Other Methods
**Elastic Net:** Combines L1 and L2 penalties
**SCAD:** Smoothly clipped absolute deviation
**Group LASSO:** Selects groups of variables
**Adaptive LASSO:** Data-adaptive penalties

## 15.17 Machine Learning Connections

### Bias-Variance Decomposition
```
E[(Y - f̂(X))²] = Bias²[f̂(X)] + Var[f̂(X)] + σ²
```

### Regularization as Bias-Variance Trade-off
- **Low λ:** Low bias, high variance
- **High λ:** High bias, low variance
- **Optimal λ:** Minimizes total error

### Tree-based Methods
**Random Forest:** Ensemble of regression trees
**Gradient Boosting:** Sequential fitting of weak learners
**BART:** Bayesian Additive Regression Trees

### Neural Networks
**Universal approximation:** Can approximate any continuous function
**Regularization:** Dropout, weight decay, early stopping

## 15.18 Practical Considerations

### Sample Size Guidelines
**Rule of thumb:** Need at least 10-20 observations per parameter
**For inference:** Larger samples needed for reliable standard errors
**For prediction:** Depends on signal-to-noise ratio

### Computational Issues
**Numerical stability:** Use QR decomposition or SVD instead of (XᵀX)⁻¹
**Large datasets:** Stochastic gradient descent methods
**Missing data:** Multiple imputation or full information maximum likelihood

### Software Implementation
**R:** lm(), glmnet(), randomForest()
**Python:** sklearn, statsmodels
**Specialized:** SAS, SPSS, Stata

## Key Insights

1. **Model Building is Iterative:** Rarely get the right model on first try.

2. **Validation is Crucial:** Must assess out-of-sample performance.

3. **Parsimony Principle:** Simpler models often predict better.

4. **Assumptions Matter:** Violations can invalidate inference.

5. **Context is Key:** Statistical significance ≠ practical importance.

## Common Pitfalls

1. **Overfitting:** Including too many variables relative to sample size
2. **Data snooping:** Using same data for selection and inference
3. **Multicollinearity:** Ignoring correlation among predictors
4. **Assumption violations:** Not checking model assumptions
5. **Causal interpretation:** Confusing association with causation

## Connections to Other Chapters

### To Chapter 14 (Simple Linear Regression)
- Extension to multiple predictors
- Matrix formulation
- Similar diagnostic procedures

### To Chapter 5 (Inequalities)
- Concentration inequalities for high-dimensional settings
- Oracle inequalities for regularized methods

### To Chapter 11 (Hypothesis Testing)
- F-tests for nested models
- Multiple testing corrections

### To Chapter 23 (Classification)
- Logistic regression as extension
- Regularization methods
- Cross-validation for model selection

This chapter provides comprehensive coverage of multiple regression, from classical least squares to modern high-dimensional methods, essential for understanding contemporary statistical modeling.