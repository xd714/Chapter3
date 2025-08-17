# Chapter 16: Logistic Regression - Mathematical Explanations

## Overview
Logistic regression is the fundamental method for modeling binary and categorical outcomes. This chapter covers the mathematical foundations of logistic regression, parameter estimation, inference, model diagnostics, and extensions to multinomial outcomes.

## 16.1 Introduction to Logistic Regression

### Motivation
**Problem with linear regression for binary outcomes:**
- Predicted values can be outside [0,1]
- Error terms not normally distributed
- Variance not constant (heteroscedasticity)

### Logistic Function
The **logistic function** maps real line to (0,1):
```
p(x) = exp(β₀ + β₁x) / (1 + exp(β₀ + β₁x)) = 1 / (1 + exp(-(β₀ + β₁x)))
```

**Properties:**
- S-shaped curve
- Always between 0 and 1
- Smooth and differentiable
- lim_{x→-∞} p(x) = 0, lim_{x→∞} p(x) = 1

## 16.2 The Logistic Regression Model

### Binary Response Model
For binary outcome Y ∈ {0, 1} and covariates X:
```
P(Y = 1|X = x) = exp(β₀ + β₁x₁ + ... + βₚxₚ) / (1 + exp(β₀ + β₁x₁ + ... + βₚxₚ))
```

**Vector notation:**
```
P(Y = 1|X = x) = exp(xᵀβ) / (1 + exp(xᵀβ))
```

### Logit Transformation
The **logit** (log-odds) is:
```
logit(p) = log(p/(1-p)) = β₀ + β₁x₁ + ... + βₚxₚ = xᵀβ
```

**Key insight:** Logit is linear in parameters and covariates.

### Odds and Odds Ratios
**Odds:** ratio of probability of success to probability of failure
```
odds = p/(1-p) = exp(xᵀβ)
```

**Odds ratio:** For one-unit change in xⱼ:
```
OR = exp(βⱼ)
```

**Interpretation:**
- βⱼ > 0: Positive association (OR > 1)
- βⱼ < 0: Negative association (OR < 1)
- βⱼ = 0: No association (OR = 1)

## 16.3 Maximum Likelihood Estimation

### Likelihood Function
For observations (x₁, y₁), ..., (xₙ, yₙ):
```
L(β) = ∏ᵢ₌₁ⁿ pᵢʸⁱ(1-pᵢ)¹⁻ʸⁱ
```

where pᵢ = P(Yᵢ = 1|xᵢ) = exp(xᵢᵀβ)/(1 + exp(xᵢᵀβ))

### Log-likelihood
```
ℓ(β) = ∑ᵢ₌₁ⁿ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
      = ∑ᵢ₌₁ⁿ [yᵢxᵢᵀβ - log(1 + exp(xᵢᵀβ))]
```

### Score Function
```
S(β) = ∂ℓ/∂β = ∑ᵢ₌₁ⁿ (yᵢ - pᵢ)xᵢ = Xᵀ(y - p)
```

where X is design matrix, y is response vector, p is vector of fitted probabilities.

### Fisher Information Matrix
```
I(β) = E[-∂²ℓ/∂β∂βᵀ] = ∑ᵢ₌₁ⁿ pᵢ(1-pᵢ)xᵢxᵢᵀ = XᵀWX
```

where W = diag(p₁(1-p₁), ..., pₙ(1-pₙ))

### Newton-Raphson Algorithm
Iterative solution:
```
β⁽ᵏ⁺¹⁾ = β⁽ᵏ⁾ + I⁻¹(β⁽ᵏ⁾)S(β⁽ᵏ⁾)
       = β⁽ᵏ⁾ + (XᵀW⁽ᵏ⁾X)⁻¹Xᵀ(y - p⁽ᵏ⁾)
```

**Equivalent form:** Iteratively reweighted least squares (IRLS)
```
β⁽ᵏ⁺¹⁾ = (XᵀW⁽ᵏ⁾X)⁻¹XᵀW⁽ᵏ⁾z⁽ᵏ⁾
```

where z⁽ᵏ⁾ = Xβ⁽ᵏ⁾ + W⁻¹(y - p⁽ᵏ⁾) is working response.

## 16.4 Inference and Hypothesis Testing

### Asymptotic Distribution
```
√n(β̂ - β) ⇝ N(0, I⁻¹(β))
```

**Approximate distribution:**
```
β̂ ∼ N(β, I⁻¹(β̂))
```

### Wald Tests
For individual parameters:
```
z = β̂ⱼ/se(β̂ⱼ) ∼ N(0,1)
```

**Confidence interval:**
```
β̂ⱼ ± z_{α/2} · se(β̂ⱼ)
```

For multiple parameters H₀: Cβ = d:
```
W = (Cβ̂ - d)ᵀ[C·I⁻¹(β̂)·Cᵀ]⁻¹(Cβ̂ - d) ∼ χ²_q
```

where q = rank(C).

### Likelihood Ratio Tests
```
LRT = 2[ℓ(β̂) - ℓ(β̂₀)] ∼ χ²_q
```

where β̂₀ is MLE under null hypothesis.

### Score Tests
```
LM = S(β̂₀)ᵀI⁻¹(β̂₀)S(β̂₀) ∼ χ²_q
```

**Advantage:** Only requires fitting null model.

## 16.5 Model Diagnostics

### Goodness of Fit

**Deviance:**
```
D = 2[ℓ_saturated - ℓ(β̂)] = 2∑ᵢ[yᵢ log(yᵢ/p̂ᵢ) + (1-yᵢ) log((1-yᵢ)/(1-p̂ᵢ))]
```

**Pearson chi-square:**
```
X² = ∑ᵢ (yᵢ - p̂ᵢ)²/(p̂ᵢ(1-p̂ᵢ))
```

### Residuals

**Pearson residuals:**
```
rᵢᴾ = (yᵢ - p̂ᵢ)/√(p̂ᵢ(1-p̂ᵢ))
```

**Deviance residuals:**
```
rᵢᴰ = sign(yᵢ - p̂ᵢ)√[2yᵢ log(yᵢ/p̂ᵢ) + 2(1-yᵢ) log((1-yᵢ)/(1-p̂ᵢ))]
```

**Standardized residuals:**
```
rᵢˢ = rᵢᴾ/√(1-hᵢᵢ)
```

where hᵢᵢ are diagonal elements of hat matrix.

### Influential Observations

**Cook's distance:**
```
Dᵢ = (rᵢᴾ)²/(p+1) · hᵢᵢ/(1-hᵢᵢ)
```

**DFBETAS:** Change in β̂ⱼ when removing observation i

### Model Adequacy

**Hosmer-Lemeshow test:**
1. Group observations by predicted probabilities
2. Compare observed vs expected in each group
3. Test statistic follows χ² distribution

**Receiver Operating Characteristic (ROC) curve:**
- Plot sensitivity vs (1 - specificity)
- Area under curve (AUC) measures discriminatory ability
- AUC = 0.5: No discriminatory ability
- AUC = 1.0: Perfect discrimination

## 16.6 Model Selection and Regularization

### Information Criteria
**AIC:** AIC = -2ℓ(β̂) + 2p
**BIC:** BIC = -2ℓ(β̂) + p log(n)

### Stepwise Selection
Same procedures as linear regression:
- Forward selection
- Backward elimination  
- Bidirectional

### Regularized Logistic Regression

**Ridge logistic regression:**
```
β̂ᴿⁱᵈᵍᵉ = argmin{-ℓ(β) + λ∑ⱼ βⱼ²}
```

**LASSO logistic regression:**
```
β̂ᴸᴬˢˢᴼ = argmin{-ℓ(β) + λ∑ⱼ |βⱼ|}
```

**Elastic Net:**
```
β̂ᴱᴺ = argmin{-ℓ(β) + λ₁∑ⱼ |βⱼ| + λ₂∑ⱼ βⱼ²}
```

### Cross-Validation
Use k-fold CV to select tuning parameters:
1. Divide data into k folds
2. For each λ, fit model on k-1 folds
3. Predict on remaining fold
4. Choose λ minimizing CV deviance

## 16.7 Multinomial Logistic Regression

### Model for K Categories
For outcome Y ∈ {1, 2, ..., K}:
```
P(Y = j|x) = exp(xᵀβⱼ) / ∑ₖ₌₁ᴷ exp(xᵀβₖ)
```

**Identifiability constraint:** Set β₁ = 0 (reference category)

**Simplified form:**
```
P(Y = j|x) = exp(xᵀβⱼ) / (1 + ∑ₖ₌₂ᴷ exp(xᵀβₖ))  for j = 2, ..., K
P(Y = 1|x) = 1 / (1 + ∑ₖ₌₂ᴷ exp(xᵀβₖ))
```

### Log-odds Ratios
```
log(P(Y = j|x)/P(Y = 1|x)) = xᵀβⱼ  for j = 2, ..., K
```

### Maximum Likelihood Estimation
**Log-likelihood:**
```
ℓ(β) = ∑ᵢ