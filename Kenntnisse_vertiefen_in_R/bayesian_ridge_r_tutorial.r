# ===============================================================================
# BAYESIAN RIDGE REGRESSION TUTORIAL IN R
# A step-by-step learning guide
# ===============================================================================

# Load required libraries
library(ggplot2)
library(glmnet)  # for comparison

# ===============================================================================
# STEP 1: UNDERSTANDING THE CONCEPT
# ===============================================================================

cat("=== BAYESIAN RIDGE REGRESSION TUTORIAL ===\n\n")

cat("What is Bayesian Ridge Regression?\n")
cat("- It's ridge regression with automatic parameter tuning\n")
cat("- Uses Bayesian inference to estimate regularization strength\n")
cat("- Provides uncertainty estimates for predictions\n")
cat("- No need to manually choose the lambda parameter\n\n")

# ===============================================================================
# STEP 2: SIMPLE IMPLEMENTATION
# ===============================================================================

# Simple Bayesian Ridge function
simple_bayesian_ridge <- function(X, y, max_iter = 100) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Add intercept column
  X_full <- cbind(1, X)
  
  # Start with initial guesses
  alpha <- 1.0    # Controls how much we shrink coefficients
  lambda <- 1.0   # Controls noise level
  
  # Store history to see convergence
  alpha_history <- c()
  lambda_history <- c()
  
  cat("Starting iterations...\n")
  
  for (i in 1:max_iter) {
    # Step 1: Calculate coefficient estimates
    # This is the "Bayesian" part - we get a distribution, not just point estimates
    precision_matrix <- alpha * diag(p + 1) + lambda * t(X_full) %*% X_full
    covariance_matrix <- solve(precision_matrix)
    coefficients <- lambda * covariance_matrix %*% t(X_full) %*% y
    
    # Step 2: Update our guesses for alpha and lambda
    # Alpha controls regularization strength
    effective_params <- (p + 1) - alpha * sum(diag(covariance_matrix))
    alpha_new <- effective_params / sum(coefficients^2)
    
    # Lambda controls noise precision
    residuals <- y - X_full %*% coefficients
    lambda_new <- (n - effective_params) / sum(residuals^2)
    
    # Store for plotting
    alpha_history <- c(alpha_history, alpha_new)
    lambda_history <- c(lambda_history, lambda_new)
    
    # Check if we've converged (values stop changing much)
    if (i > 1 && abs(alpha_new - alpha) < 0.001 && abs(lambda_new - lambda) < 0.001) {
      cat("Converged at iteration", i, "\n")
      break
    }
    
    alpha <- alpha_new
    lambda <- lambda_new
  }
  
  # Return everything we need
  list(
    coefficients = as.vector(coefficients),
    intercept = coefficients[1],
    coef = coefficients[-1],
    alpha = alpha,
    lambda = lambda,
    covariance = covariance_matrix,
    alpha_history = alpha_history,
    lambda_history = lambda_history
  )
}

# ===============================================================================
# STEP 3: CREATE SAMPLE DATA
# ===============================================================================

cat("\n=== CREATING SAMPLE DATA ===\n")

set.seed(123)  # For reproducible results

# Create simple regression problem
n_samples <- 50
n_features <- 5

# Create feature matrix
X <- matrix(rnorm(n_samples * n_features), nrow = n_samples)
colnames(X) <- paste0("Feature_", 1:n_features)

# Create true coefficients (what we want to recover)
true_coefficients <- c(2, -1, 0.5, 0, -1.5)
names(true_coefficients) <- colnames(X)

# Create target variable with some noise
y <- X %*% true_coefficients + rnorm(n_samples, sd = 0.5)

cat("Data created:\n")
cat("- Samples:", n_samples, "\n")
cat("- Features:", n_features, "\n")
cat("- True coefficients:", paste(round(true_coefficients, 2), collapse = ", "), "\n\n")

# ===============================================================================
# STEP 4: FIT THE MODEL
# ===============================================================================

cat("=== FITTING BAYESIAN RIDGE MODEL ===\n")

# Standardize features (important for ridge regression)
X_scaled <- scale(X)

# Fit our Bayesian Ridge model
model <- simple_bayesian_ridge(X_scaled, y)

cat("Model fitted successfully!\n")
cat("Final alpha (regularization):", round(model$alpha, 4), "\n")
cat("Final lambda (noise precision):", round(model$lambda, 4), "\n\n")

# ===============================================================================
# STEP 5: EXAMINE RESULTS
# ===============================================================================

cat("=== COMPARING COEFFICIENTS ===\n")

# Compare true vs estimated coefficients
comparison <- data.frame(
  Feature = names(true_coefficients),
  True = true_coefficients,
  Estimated = model$coef,
  Difference = abs(true_coefficients - model$coef)
)

print(comparison)

# Calculate how well we recovered the coefficients
recovery_error <- mean(abs(true_coefficients - model$coef))
cat("\nMean absolute error in coefficient recovery:", round(recovery_error, 4), "\n\n")

# ===============================================================================
# STEP 6: MAKE PREDICTIONS WITH UNCERTAINTY
# ===============================================================================

cat("=== MAKING PREDICTIONS ===\n")

# Function to make predictions with uncertainty
predict_with_uncertainty <- function(model, X_new) {
  X_new_scaled <- scale(X_new, center = attr(X_scaled, "scaled:center"), 
                        scale = attr(X_scaled, "scaled:scale"))
  X_new_full <- cbind(1, X_new_scaled)
  
  # Point prediction
  y_pred <- X_new_full %*% model$coefficients
  
  # Uncertainty (standard deviation of prediction)
  pred_variance <- 1/model$lambda + diag(X_new_full %*% model$covariance %*% t(X_new_full))
  pred_std <- sqrt(pred_variance)
  
  list(prediction = as.vector(y_pred), uncertainty = pred_std)
}

# Test on some new data
X_test <- matrix(rnorm(5 * n_features), nrow = 5)
predictions <- predict_with_uncertainty(model, X_test)

cat("Test predictions:\n")
for (i in 1:5) {
  cat(sprintf("Sample %d: %.2f ± %.2f\n", i, 
              predictions$prediction[i], predictions$uncertainty[i]))
}

# ===============================================================================
# STEP 7: VISUALIZE CONVERGENCE
# ===============================================================================

cat("\n=== CREATING VISUALIZATIONS ===\n")

# Plot how alpha and lambda changed during fitting
convergence_data <- data.frame(
  Iteration = 1:length(model$alpha_history),
  Alpha = model$alpha_history,
  Lambda = model$lambda_history
)

# Create convergence plot
p1 <- ggplot(convergence_data) +
  geom_line(aes(x = Iteration, y = Alpha, color = "Alpha"), size = 1.2) +
  geom_line(aes(x = Iteration, y = Lambda, color = "Lambda"), size = 1.2) +
  labs(title = "How Parameters Converged",
       x = "Iteration", y = "Parameter Value",
       color = "Parameter") +
  theme_minimal() +
  theme(legend.position = "bottom")

print(p1)

# ===============================================================================
# STEP 8: COMPARE WITH REGULAR RIDGE
# ===============================================================================

cat("\n=== COMPARING WITH REGULAR RIDGE REGRESSION ===\n")

# Fit regular ridge with cross-validation
ridge_cv <- cv.glmnet(X_scaled, y, alpha = 0)  # alpha=0 means ridge
ridge_model <- glmnet(X_scaled, y, alpha = 0, lambda = ridge_cv$lambda.min)

cat("Regular Ridge optimal lambda:", round(ridge_cv$lambda.min, 4), "\n")
cat("Bayesian Ridge equivalent lambda: 1/alpha =", round(1/model$alpha, 4), "\n\n")

# Compare coefficients
coef_comparison <- data.frame(
  Feature = names(true_coefficients),
  True = true_coefficients,
  Bayesian_Ridge = model$coef,
  Regular_Ridge = as.vector(coef(ridge_model))[-1]  # exclude intercept
)

print(coef_comparison)

# ===============================================================================
# STEP 9: UNDERSTANDING THE BENEFITS
# ===============================================================================

cat("\n=== KEY BENEFITS OF BAYESIAN RIDGE ===\n")
cat("1. AUTOMATIC TUNING: No need to choose lambda manually\n")
cat("2. UNCERTAINTY: Tells you how confident predictions are\n")
cat("3. ADAPTIVE: Regularization strength adapts to your data\n")
cat("4. ROBUST: Less likely to overfit than regular regression\n\n")

# Demonstrate uncertainty benefit
cat("=== UNCERTAINTY EXAMPLE ===\n")
cat("Bayesian Ridge tells you prediction confidence:\n")
for (i in 1:3) {
  conf_level <- round(100 * (1 - 2 * pnorm(-1.96)), 1)  # ~95%
  lower <- predictions$prediction[i] - 1.96 * predictions$uncertainty[i]
  upper <- predictions$prediction[i] + 1.96 * predictions$uncertainty[i]
  cat(sprintf("Sample %d: %.2f (95%% CI: %.2f to %.2f)\n", 
              i, predictions$prediction[i], lower, upper))
}

# ===============================================================================
# STEP 10: PRACTICAL USAGE TEMPLATE
# ===============================================================================

cat("\n=== PRACTICAL USAGE TEMPLATE ===\n")
cat("# Here's how to use it in practice:\n\n")

cat("# 1. Prepare your data\n")
cat("X <- your_feature_matrix\n")
cat("y <- your_target_vector\n")
cat("X_scaled <- scale(X)  # Always scale for ridge regression!\n\n")

cat("# 2. Fit the model\n")
cat("model <- simple_bayesian_ridge(X_scaled, y)\n\n")

cat("# 3. Make predictions\n")
cat("predictions <- predict_with_uncertainty(model, X_new)\n\n")

cat("# 4. Check convergence\n")
cat("plot(model$alpha_history)  # Should flatten out\n\n")

cat("# 5. Examine coefficients\n")
cat("print(model$coef)\n")
cat("cat('Regularization strength:', model$alpha)\n\n")

# ===============================================================================
# SUMMARY
# ===============================================================================

cat("=== SUMMARY ===\n")
cat("You've learned:\n")
cat("✓ What Bayesian Ridge Regression is\n")
cat("✓ How it automatically tunes regularization\n")
cat("✓ How to implement it from scratch\n")
cat("✓ How to get uncertainty estimates\n")
cat("✓ How it compares to regular ridge regression\n")
cat("✓ When and why to use it\n\n")

cat("Next steps:\n")
cat("- Try it on your own data\n")
cat("- Experiment with different sample sizes\n")
cat("- Compare with other regularization methods\n")
cat("- Use the uncertainty estimates for decision making\n")