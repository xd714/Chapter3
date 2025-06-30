# ===============================================================================
# BAYESIAN SHRINKAGE vs RIDGE: SIMPLE COMPARISON
# Easy-to-understand explanation with R examples
# ===============================================================================

library(ggplot2)
library(glmnet)

set.seed(42)

cat("=== SIMPLE EXPLANATION ===\n\n")

cat("Think of shrinkage like 'pulling coefficients toward zero':\n\n")

cat("BAYESIAN RIDGE REGRESSION:\n")
cat("→ Pulls ALL coefficients equally toward zero\n")
cat("→ Like a gentle, uniform squeeze\n")
cat("→ Good when most features matter a little\n\n")

cat("BAYESIAN SHRINKAGE (other methods):\n")
cat("→ Can pull coefficients differently\n")
cat("→ Some methods can make coefficients exactly zero\n")
cat("→ Good when only some features really matter\n\n")

# ===============================================================================
# CREATE SIMPLE EXAMPLE DATA
# ===============================================================================

cat("=== CREATING SIMPLE EXAMPLE ===\n")

# Small, easy example
n <- 50  # 50 observations
p <- 8   # 8 features

# Create features
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- paste0("Feature_", 1:p)

# Create TRUE coefficients - some are zero, some are not
true_coef <- c(2, 0, -1.5, 0, 0, 1, 0, -0.8)
names(true_coef) <- colnames(X)

# Create response variable
y <- X %*% true_coef + rnorm(n, sd = 0.3)

cat("TRUE COEFFICIENTS:\n")
for(i in 1:length(true_coef)) {
  if(true_coef[i] == 0) {
    cat(sprintf("%s: %.1f (ZERO - doesn't matter)\n", names(true_coef)[i], true_coef[i]))
  } else {
    cat(sprintf("%s: %.1f (NON-ZERO - important!)\n", names(true_coef)[i], true_coef[i]))
  }
}

important_features <- sum(true_coef != 0)
cat(sprintf("\nSo %d out of %d features are actually important.\n\n", important_features, p)

# ===============================================================================
# METHOD 1: BAYESIAN RIDGE (Equal Shrinkage)
# ===============================================================================

cat("=== METHOD 1: BAYESIAN RIDGE ===\n")
cat("This method shrinks all coefficients equally toward zero.\n\n")

# Simple Bayesian Ridge implementation
simple_bayesian_ridge <- function(X, y) {
  X_scaled <- scale(X)
  n <- nrow(X_scaled)
  p <- ncol(X_scaled)
  
  # Add intercept
  X_full <- cbind(1, X_scaled)
  
  # Start with initial guess
  alpha <- 1.0  # How much to shrink
  
  # Iterate to find best shrinkage
  for(i in 1:50) {
    # Calculate coefficients with current shrinkage
    precision_matrix <- alpha * diag(p + 1) + t(X_full) %*% X_full
    coef_estimate <- solve(precision_matrix) %*% t(X_full) %*% y
    
    # Update shrinkage based on coefficient sizes
    alpha <- (p + 1) / sum(coef_estimate^2)
  }
  
  list(
    coefficients = as.vector(coef_estimate[-1]),  # Remove intercept
    shrinkage = alpha,
    method = "Bayesian Ridge"
  )
}

ridge_result <- simple_bayesian_ridge(X, y)

cat("Bayesian Ridge Results:\n")
cat(sprintf("Shrinkage parameter: %.3f\n", ridge_result$shrinkage))
cat("Estimated coefficients:\n")
for(i in 1:length(ridge_result$coefficients)) {
  cat(sprintf("%s: %.3f (true: %.1f)\n", 
              names(true_coef)[i], ridge_result$coefficients[i], true_coef[i]))
}

# ===============================================================================
# METHOD 2: BAYESIAN LASSO (Sparse Shrinkage)
# ===============================================================================

cat("\n=== METHOD 2: BAYESIAN LASSO ===\n")
cat("This method can shrink some coefficients to exactly ZERO.\n\n")

# Use glmnet for Lasso (it's easier and equivalent)
lasso_cv <- cv.glmnet(scale(X), y, alpha = 1)  # alpha=1 means Lasso
lasso_coef <- as.vector(coef(lasso_cv, s = "lambda.min"))[-1]  # Remove intercept

cat("Bayesian Lasso Results:\n")
cat(sprintf("Optimal lambda: %.4f\n", lasso_cv$lambda.min))
cat("Estimated coefficients:\n")
for(i in 1:length(lasso_coef)) {
  if(abs(lasso_coef[i]) < 0.001) {
    cat(sprintf("%s: %.3f (true: %.1f) ← SHRUNK TO ZERO!\n", 
                names(true_coef)[i], lasso_coef[i], true_coef[i]))
  } else {
    cat(sprintf("%s: %.3f (true: %.1f)\n", 
                names(true_coef)[i], lasso_coef[i], true_coef[i]))
  }
}

# ===============================================================================
# COMPARE THE METHODS
# ===============================================================================

cat("\n=== COMPARISON ===\n")

# Create comparison data frame
comparison <- data.frame(
  Feature = names(true_coef),
  True = true_coef,
  Bayesian_Ridge = ridge_result$coefficients,
  Bayesian_Lasso = lasso_coef,
  True_Important = ifelse(true_coef == 0, "No", "Yes")
)

print(round(comparison, 3))

# Calculate how many coefficients each method set to zero
ridge_zeros <- sum(abs(ridge_result$coefficients) < 0.01)
lasso_zeros <- sum(abs(lasso_coef) < 0.01)
true_zeros <- sum(true_coef == 0)

cat(sprintf("\nZERO COEFFICIENTS:\n"))
cat(sprintf("True number of zeros: %d\n", true_zeros))
cat(sprintf("Bayesian Ridge zeros: %d\n", ridge_zeros))
cat(sprintf("Bayesian Lasso zeros: %d\n", lasso_zeros))

# Calculate accuracy
ridge_error <- mean((true_coef - ridge_result$coefficients)^2)
lasso_error <- mean((true_coef - lasso_coef)^2)

cat(sprintf("\nACCURACY (Mean Squared Error):\n"))
cat(sprintf("Bayesian Ridge: %.4f\n", ridge_error))
cat(sprintf("Bayesian Lasso: %.4f\n", lasso_error))

if(lasso_error < ridge_error) {
  cat("→ Lasso is more accurate here!\n")
} else {
  cat("→ Ridge is more accurate here!\n")
}

# ===============================================================================
# VISUALIZATION
# ===============================================================================

cat("\n=== CREATING VISUAL COMPARISON ===\n")

# Prepare data for plotting
plot_data <- data.frame(
  Feature = rep(names(true_coef), 3),
  Coefficient = c(true_coef, ridge_result$coefficients, lasso_coef),
  Method = rep(c("True", "Bayesian Ridge", "Bayesian Lasso"), each = length(true_coef))
)

# Create the plot
p1 <- ggplot(plot_data, aes(x = Feature, y = Coefficient, fill = Method)) +
  geom_col(position = "dodge", alpha = 0.8) +
  labs(title = "Coefficient Comparison: Ridge vs Lasso Shrinkage",
       subtitle = "See how different methods shrink coefficients",
       x = "Features", y = "Coefficient Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  scale_fill_manual(values = c("True" = "darkgreen", 
                               "Bayesian Ridge" = "blue", 
                               "Bayesian Lasso" = "red"))

print(p1)

# Create shrinkage visualization
shrinkage_data <- data.frame(
  True_Abs = rep(abs(true_coef), 2),
  Estimated_Abs = c(abs(ridge_result$coefficients), abs(lasso_coef)),
  Method = rep(c("Bayesian Ridge", "Bayesian Lasso"), each = length(true_coef)),
  Important = rep(ifelse(true_coef == 0, "Unimportant", "Important"), 2)
)

p2 <- ggplot(shrinkage_data, aes(x = True_Abs, y = Estimated_Abs, color = Important)) +
  geom_point(size = 3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 0.5) +
  facet_wrap(~ Method) +
  labs(title = "Shrinkage Patterns",
       subtitle = "Points on diagonal = perfect estimation",
       x = "True |Coefficient|", y = "Estimated |Coefficient|",
       color = "Feature Type") +
  theme_minimal() +
  scale_color_manual(values = c("Important" = "red", "Unimportant" = "blue"))

print(p2)

# ===============================================================================
# WHEN TO USE WHICH METHOD
# ===============================================================================

cat("\n=== WHEN TO USE WHICH METHOD? ===\n\n")

cat("USE BAYESIAN RIDGE WHEN:\n")
cat("✓ You think most features have some small effect\n")
cat("✓ You don't want any features to be completely ignored\n")
cat("✓ You want smooth, gradual shrinkage\n")
cat("✓ Your data has multicollinearity (correlated features)\n\n")

cat("USE BAYESIAN LASSO (or other sparse methods) WHEN:\n")
cat("✓ You think only a few features really matter\n")
cat("✓ You want automatic feature selection\n")
cat("✓ You have many more features than observations\n")
cat("✓ You want interpretable models with fewer variables\n\n")

# ===============================================================================
# PRACTICAL EXAMPLE
# ===============================================================================

cat("=== PRACTICAL EXAMPLE ===\n\n")

cat("Imagine predicting house prices with features:\n")
cat("- Square footage, bedrooms, bathrooms (important)\n")
cat("- Garage size, garden area (somewhat important)\n")
cat("- Window color, door style, mailbox type (probably unimportant)\n\n")

cat("BAYESIAN RIDGE would:\n")
cat("→ Keep all features but shrink unimportant ones\n")
cat("→ Square footage: 50000 → 45000 (small shrinkage)\n")
cat("→ Window color: 100 → 20 (more shrinkage)\n")
cat("→ Mailbox type: 50 → 10 (more shrinkage)\n\n")

cat("BAYESIAN LASSO would:\n")
cat("→ Keep important features, eliminate unimportant ones\n")
cat("→ Square footage: 50000 → 48000 (small shrinkage)\n")
cat("→ Window color: 100 → 0 (eliminated!)\n")
cat("→ Mailbox type: 50 → 0 (eliminated!)\n\n")

# ===============================================================================
# DEMONSTRATE WITH DIFFERENT SCENARIOS
# ===============================================================================

cat("=== TESTING DIFFERENT SCENARIOS ===\n\n")

# Scenario 1: Many small effects
cat("SCENARIO 1: Many small effects (Ridge should win)\n")
test_scenario_1 <- function() {
  X_test <- matrix(rnorm(50 * 10), 50, 10)
  # All coefficients are small but non-zero
  true_coef_1 <- rep(0.3, 10)  # All features matter a little
  y_test <- X_test %*% true_coef_1 + rnorm(50, sd = 0.2)
  
  ridge_test <- simple_bayesian_ridge(X_test, y_test)
  lasso_test <- cv.glmnet(scale(X_test), y_test, alpha = 1)
  lasso_coef_test <- as.vector(coef(lasso_test, s = "lambda.min"))[-1]
  
  ridge_error_1 <- mean((true_coef_1 - ridge_test$coefficients)^2)
  lasso_error_1 <- mean((true_coef_1 - lasso_coef_test)^2)
  
  cat(sprintf("Ridge MSE: %.4f\n", ridge_error_1))
  cat(sprintf("Lasso MSE: %.4f\n", lasso_error_1))
  cat(sprintf("Winner: %s\n\n", ifelse(ridge_error_1 < lasso_error_1, "Ridge", "Lasso")))
}

test_scenario_1()

# Scenario 2: Sparse effects
cat("SCENARIO 2: Very sparse effects (Lasso should win)\n")
test_scenario_2 <- function() {
  X_test <- matrix(rnorm(50 * 10), 50, 10)
  # Only 2 out of 10 coefficients are non-zero
  true_coef_2 <- c(2, 0, 0, 0, 0, 0, 0, 0, 0, -1.5)
  y_test <- X_test %*% true_coef_2 + rnorm(50, sd = 0.2)
  
  ridge_test <- simple_bayesian_ridge(X_test, y_test)
  lasso_test <- cv.glmnet(scale(X_test), y_test, alpha = 1)
  lasso_coef_test <- as.vector(coef(lasso_test, s = "lambda.min"))[-1]
  
  ridge_error_2 <- mean((true_coef_2 - ridge_test$coefficients)^2)
  lasso_error_2 <- mean((true_coef_2 - lasso_coef_test)^2)
  
  cat(sprintf("Ridge MSE: %.4f\n", ridge_error_2))
  cat(sprintf("Lasso MSE: %.4f\n", lasso_error_2))
  cat(sprintf("Winner: %s\n\n", ifelse(ridge_error_2 < lasso_error_2, "Ridge", "Lasso")))
}

test_scenario_2()

# ===============================================================================
# UNCERTAINTY QUANTIFICATION EXAMPLE
# ===============================================================================

cat("=== UNCERTAINTY QUANTIFICATION ===\n\n")

cat("A big advantage of Bayesian methods: they tell you how confident they are!\n\n")

# Simple uncertainty for Bayesian Ridge
predict_with_uncertainty <- function(model_result, X_new) {
  X_new_scaled <- scale(X_new, center = attr(scale(X), "scaled:center"), 
                        scale = attr(scale(X), "scaled:scale"))
  
  # Point prediction
  prediction <- X_new_scaled %*% model_result$coefficients
  
  # Simple uncertainty estimate (in reality, this would be more complex)
  # Higher shrinkage = less confidence
  uncertainty <- sqrt(1/model_result$shrinkage + 0.1)
  
  list(prediction = as.vector(prediction), uncertainty = uncertainty)
}

# Test on new data
X_new <- matrix(rnorm(3 * p), 3, p)
pred_result <- predict_with_uncertainty(ridge_result, X_new)

cat("PREDICTIONS WITH UNCERTAINTY:\n")
for(i in 1:3) {
  lower <- pred_result$prediction[i] - 1.96 * pred_result$uncertainty
  upper <- pred_result$prediction[i] + 1.96 * pred_result$uncertainty
  cat(sprintf("Sample %d: %.2f (95%% CI: %.2f to %.2f)\n", 
              i, pred_result$prediction[i], lower, upper))
}

# ===============================================================================
# SUMMARY AND TAKEAWAYS
# ===============================================================================

cat("\n=== KEY TAKEAWAYS ===\n\n")

cat("1. TERMINOLOGY CLARIFICATION:\n")
cat("   • 'Bayesian Ridge' = specific type of Bayesian shrinkage\n")
cat("   • 'Bayesian Shrinkage' = broader category including Ridge, Lasso, Horseshoe, etc.\n\n")

cat("2. MAIN DIFFERENCES:\n")
cat("   • Ridge: Gentle, equal shrinkage for all coefficients\n")
cat("   • Lasso: Aggressive shrinkage, can eliminate features completely\n")
cat("   • Other methods: Various trade-offs between these extremes\n\n")

cat("3. WHEN TO USE:\n")
cat("   • Many relevant features → Bayesian Ridge\n")
cat("   • Few relevant features → Bayesian Lasso\n")
cat("   • Unsure about sparsity → Try both and compare!\n\n")

cat("4. BAYESIAN ADVANTAGES:\n")
cat("   • Automatic parameter tuning\n")
cat("   • Uncertainty quantification\n")
cat("   • Principled shrinkage based on data\n")
cat("   • Robustness to overfitting\n\n")

# ===============================================================================
# PRACTICAL WORKFLOW
# ===============================================================================

cat("=== PRACTICAL WORKFLOW FOR YOUR ANALYSIS ===\n\n")

cat("STEP 1: Explore your data\n")
cat("```r\n")
cat("# Look at correlation between features\n")
cat("cor(X)\n")
cat("# Check if you have more features than observations\n")
cat("cat('Features:', ncol(X), 'Observations:', nrow(X))\n")
cat("```\n\n")

cat("STEP 2: Try both methods\n")
cat("```r\n")
cat("# Fit both models\n")
cat("ridge_model <- simple_bayesian_ridge(X, y)\n")
cat("lasso_model <- cv.glmnet(scale(X), y, alpha = 1)\n")
cat("```\n\n")

cat("STEP 3: Compare performance\n")
cat("```r\n")
cat("# Use cross-validation or hold-out test set\n")
cat("# Compare prediction accuracy\n")
cat("# Look at coefficient patterns\n")
cat("```\n\n")

cat("STEP 4: Choose based on:\n")
cat("• Prediction accuracy\n")
cat("• Interpretability needs\n")
cat("• Domain knowledge about feature importance\n\n")

cat("STEP 5: Get uncertainty estimates\n")
cat("```r\n")
cat("# Bayesian methods give you confidence intervals\n")
cat("# Use these for decision making!\n")
cat("```\n\n")

cat("=== END OF TUTORIAL ===\n")
cat("You now understand the key differences between Bayesian Ridge and other Bayesian shrinkage methods!\n")