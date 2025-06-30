# ===============================================================================
# BAYESIAN SHRINKAGE VS BAYESIAN RIDGE ESTIMATION
# Complete comparison with R implementations
# ===============================================================================

library(ggplot2)
library(gridExtra)
library(glmnet)
library(MASS)

set.seed(123)

cat("=== UNDERSTANDING THE DIFFERENCES ===\n\n")

cat("BAYESIAN RIDGE REGRESSION:\n")
cat("- Uses Gaussian (Normal) prior: β ~ N(0, σ²/α)\n")
cat("- Shrinks ALL coefficients equally toward zero\n")
cat("- Good when many features are somewhat relevant\n")
cat("- Automatic tuning of regularization parameter α\n\n")

cat("BAYESIAN SHRINKAGE ESTIMATION (General):\n")
cat("- Can use different priors (Laplace, Horseshoe, etc.)\n")
cat("- May shrink coefficients differently\n")
cat("- Can achieve sparsity (some coefficients = 0)\n")
cat("- Includes methods like Bayesian Lasso, Horseshoe, etc.\n\n")

# ===============================================================================
# IMPLEMENTATION 1: BAYESIAN RIDGE (Gaussian Prior)
# ===============================================================================

bayesian_ridge <- function(X, y, max_iter = 100, tol = 1e-4) {
  n <- nrow(X)
  p <- ncol(X)
  X_full <- cbind(1, X)  # Add intercept
  
  # Initialize hyperparameters
  alpha <- 1.0  # precision of prior
  lambda <- 1.0  # precision of noise
  
  alpha_hist <- numeric(max_iter)
  lambda_hist <- numeric(max_iter)
  
  for (i in 1:max_iter) {
    # Posterior precision and covariance
    S_inv <- alpha * diag(p + 1) + lambda * t(X_full) %*% X_full
    S <- solve(S_inv)
    
    # Posterior mean
    mu <- lambda * S %*% t(X_full) %*% y
    
    # Update hyperparameters
    gamma <- (p + 1) - alpha * sum(diag(S))
    alpha_new <- gamma / sum(mu^2)
    
    residuals <- y - X_full %*% mu
    lambda_new <- (n - gamma) / sum(residuals^2)
    
    alpha_hist[i] <- alpha_new
    lambda_hist[i] <- lambda_new
    
    # Check convergence
    if (i > 1 && abs(alpha_new - alpha) < tol && abs(lambda_new - lambda) < tol) {
      alpha_hist <- alpha_hist[1:i]
      lambda_hist <- lambda_hist[1:i]
      break
    }
    
    alpha <- alpha_new
    lambda <- lambda_new
  }
  
  list(
    coefficients = as.vector(mu),
    coef = mu[-1],
    intercept = mu[1],
    alpha = alpha,
    lambda = lambda,
    covariance = S,
    alpha_history = alpha_hist,
    lambda_history = lambda_hist,
    shrinkage_type = "Gaussian (Ridge)"
  )
}

# ===============================================================================
# IMPLEMENTATION 2: BAYESIAN LASSO (Laplace Prior)
# ===============================================================================

bayesian_lasso <- function(X, y, max_iter = 1000, tol = 1e-4) {
  n <- nrow(X)
  p <- ncol(X)
  X_full <- cbind(1, X)
  
  # Initialize
  beta <- rep(0, p + 1)
  tau_sq <- rep(1, p + 1)  # Individual shrinkage parameters
  lambda_sq <- 1  # Global shrinkage
  sigma_sq <- 1   # Noise variance
  
  beta_history <- matrix(0, max_iter, p + 1)
  lambda_history <- numeric(max_iter)
  
  for (iter in 1:max_iter) {
    # Update beta (coefficients)
    D <- diag(tau_sq)
    V <- solve(t(X_full) %*% X_full / sigma_sq + D / lambda_sq)
    mu <- V %*% t(X_full) %*% y / sigma_sq
    
    # Sample from multivariate normal (or use mean for MAP estimate)
    beta <- as.vector(mu)  # Using MAP estimate
    
    # Update tau_sq (individual shrinkage parameters)
    for (j in 1:(p + 1)) {
      # This is a simplified update - full Gibbs would sample from inverse Gaussian
      tau_sq[j] <- sqrt(lambda_sq * sigma_sq) / abs(beta[j] + 1e-10)
    }
    
    # Update lambda_sq (global shrinkage)
    lambda_sq <- (p + 1) / sum(tau_sq * beta^2)
    
    # Update sigma_sq (noise variance)
    residuals <- y - X_full %*% beta
    sigma_sq <- sum(residuals^2) / (n - 2)
    
    beta_history[iter, ] <- beta
    lambda_history[iter] <- lambda_sq
    
    # Simple convergence check
    if (iter > 10 && max(abs(beta - beta_history[iter-1, ])) < tol) {
      beta_history <- beta_history[1:iter, ]
      lambda_history <- lambda_history[1:iter]
      break
    }
  }
  
  list(
    coefficients = beta,
    coef = beta[-1],
    intercept = beta[1],
    lambda = lambda_sq,
    sigma = sigma_sq,
    tau = tau_sq,
    beta_history = beta_history,
    lambda_history = lambda_history,
    shrinkage_type = "Laplace (Lasso)"
  )
}

# ===============================================================================
# IMPLEMENTATION 3: HORSESHOE PRIOR (Adaptive Shrinkage)
# ===============================================================================

horseshoe_estimator <- function(X, y, max_iter = 500) {
  n <- nrow(X)
  p <- ncol(X)
  X_full <- cbind(1, X)
  
  # Initialize
  beta <- rep(0, p + 1)
  lambda_j <- rep(1, p + 1)  # Local shrinkage parameters
  tau <- 1                   # Global shrinkage parameter
  sigma_sq <- 1
  
  beta_history <- matrix(0, max_iter, p + 1)
  tau_history <- numeric(max_iter)
  
  for (iter in 1:max_iter) {
    # Update beta
    D <- diag(lambda_j^2 * tau^2)
    V_inv <- t(X_full) %*% X_full / sigma_sq + solve(D)
    V <- solve(V_inv)
    mu <- V %*% t(X_full) %*% y / sigma_sq
    
    beta <- as.vector(mu)  # MAP estimate
    
    # Update local shrinkage parameters λⱼ
    # Simplified update (full would use inverse gamma distribution)
    lambda_j <- 1 / sqrt(1 + beta^2 / (tau^2 + 1e-10))
    
    # Update global shrinkage parameter τ
    # Simplified update
    tau <- sqrt(p) / sqrt(sum(beta^2 / lambda_j^2) + 1e-10)
    
    # Update noise variance
    residuals <- y - X_full %*% beta
    sigma_sq <- sum(residuals^2) / (n - 2)
    
    beta_history[iter, ] <- beta
    tau_history[iter] <- tau
    
    if (iter > 10 && max(abs(beta - beta_history[iter-1, ])) < 1e-4) {
      beta_history <- beta_history[1:iter, ]
      tau_history <- tau_history[1:iter]
      break
    }
  }
  
  list(
    coefficients = beta,
    coef = beta[-1],
    intercept = beta[1],
    tau = tau,
    lambda = lambda_j,
    sigma = sigma_sq,
    beta_history = beta_history,
    tau_history = tau_history,
    shrinkage_type = "Horseshoe (Adaptive)"
  )
}

# ===============================================================================
# GENERATE TEST DATA
# ===============================================================================

cat("=== GENERATING TEST DATA ===\n")

n <- 100
p <- 15

# Create features
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- paste0("X", 1:p)

# Create sparse true coefficients (only some are non-zero)
true_coef <- c(3, -2, 0, 0, 1.5, 0, 0, -1, 0, 0, 0, 2.5, 0, 0, -0.5)
names(true_coef) <- colnames(X)

# Generate response
y <- X %*% true_coef + rnorm(n, sd = 0.5)

# Standardize
X_scaled <- scale(X)

cat("True coefficients (sparse):\n")
print(round(true_coef, 2))
cat("\nNumber of non-zero coefficients:", sum(true_coef != 0), "out of", p, "\n\n")

# ===============================================================================
# FIT ALL MODELS
# ===============================================================================

cat("=== FITTING ALL MODELS ===\n")

# Fit all models
ridge_model <- bayesian_ridge(X_scaled, y)
lasso_model <- bayesian_lasso(X_scaled, y)
horseshoe_model <- horseshoe_estimator(X_scaled, y)

# Also fit classical versions for comparison
glmnet_ridge <- cv.glmnet(X_scaled, y, alpha = 0)
glmnet_lasso <- cv.glmnet(X_scaled, y, alpha = 1)

cat("All models fitted successfully!\n\n")

# ===============================================================================
# COMPARE COEFFICIENT ESTIMATES
# ===============================================================================

cat("=== COEFFICIENT COMPARISON ===\n")

# Combine all estimates
coef_comparison <- data.frame(
  Feature = names(true_coef),
  True = true_coef,
  Bayesian_Ridge = ridge_model$coef,
  Bayesian_Lasso = lasso_model$coef,
  Horseshoe = horseshoe_model$coef,
  GLMNet_Ridge = as.vector(coef(glmnet_ridge, s = "lambda.min"))[-1],
  GLMNet_Lasso = as.vector(coef(glmnet_lasso, s = "lambda.min"))[-1]
)

print(round(coef_comparison, 3))

# Calculate errors
errors <- data.frame(
  Method = c("Bayesian Ridge", "Bayesian Lasso", "Horseshoe", "GLMNet Ridge", "GLMNet Lasso"),
  MSE = c(
    mean((true_coef - ridge_model$coef)^2),
    mean((true_coef - lasso_model$coef)^2),
    mean((true_coef - horseshoe_model$coef)^2),
    mean((true_coef - as.vector(coef(glmnet_ridge, s = "lambda.min"))[-1])^2),
    mean((true_coef - as.vector(coef(glmnet_lasso, s = "lambda.min"))[-1])^2)
  ),
  Sparsity = c(
    sum(abs(ridge_model$coef) < 0.01),
    sum(abs(lasso_model$coef) < 0.01),
    sum(abs(horseshoe_model$coef) < 0.01),
    sum(abs(as.vector(coef(glmnet_ridge, s = "lambda.min"))[-1]) < 0.01),
    sum(abs(as.vector(coef(glmnet_lasso, s = "lambda.min"))[-1]) < 0.01)
  )
)

cat("\n=== PERFORMANCE COMPARISON ===\n")
print(errors)

# ===============================================================================
# VISUALIZATIONS
# ===============================================================================

cat("\n=== CREATING VISUALIZATIONS ===\n")

# Plot 1: Coefficient comparison
coef_long <- reshape2::melt(coef_comparison, id.vars = "Feature", 
                           variable.name = "Method", value.name = "Coefficient")

p1 <- ggplot(coef_long, aes(x = Feature, y = Coefficient, color = Method)) +
  geom_line(aes(group = Method), size = 1.2) +
  geom_point(size = 2) +
  labs(title = "Coefficient Estimates Comparison",
       x = "Feature", y = "Coefficient Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom") +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5)

# Plot 2: Shrinkage patterns
shrinkage_data <- data.frame(
  Feature = rep(names(true_coef), 4),
  True_Coef = rep(abs(true_coef), 4),
  Estimated_Coef = c(abs(ridge_model$coef), abs(lasso_model$coef), 
                     abs(horseshoe_model$coef), abs(as.vector(coef(glmnet_lasso, s = "lambda.min"))[-1])),
  Method = rep(c("Bayesian Ridge", "Bayesian Lasso", "Horseshoe", "GLMNet Lasso"), each = p)
)

p2 <- ggplot(shrinkage_data, aes(x = True_Coef, y = Estimated_Coef, color = Method)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  facet_wrap(~ Method, scales = "free") +
  labs(title = "Shrinkage Patterns: True vs Estimated Coefficients",
       x = "True |Coefficient|", y = "Estimated |Coefficient|") +
  theme_minimal() +
  theme(legend.position = "none")

# Plot 3: Convergence comparison
if (length(ridge_model$alpha_history) > 1) {
  conv_data <- data.frame(
    Iteration = c(1:length(ridge_model$alpha_history), 
                  1:length(lasso_model$lambda_history),
                  1:length(horseshoe_model$tau_history)),
    Value = c(ridge_model$alpha_history, 
              lasso_model$lambda_history,
              horseshoe_model$tau_history),
    Parameter = c(rep("Ridge Alpha", length(ridge_model$alpha_history)),
                  rep("Lasso Lambda", length(lasso_model$lambda_history)),
                  rep("Horseshoe Tau", length(horseshoe_model$tau_history)))
  )
  
  p3 <- ggplot(conv_data, aes(x = Iteration, y = Value, color = Parameter)) +
    geom_line(size = 1.2) +
    facet_wrap(~ Parameter, scales = "free_y") +
    labs(title = "Hyperparameter Convergence",
         x = "Iteration", y = "Parameter Value") +
    theme_minimal() +
    theme(legend.position = "none")
} else {
  p3 <- ggplot() + ggtitle("Convergence: Models converged in 1 iteration")
}

# Plot 4: Sparsity comparison
sparsity_data <- data.frame(
  Method = c("True", "Bayesian Ridge", "Bayesian Lasso", "Horseshoe", "GLMNet Lasso"),
  Zero_Coefficients = c(
    sum(true_coef == 0),
    sum(abs(ridge_model$coef) < 0.01),
    sum(abs(lasso_model$coef) < 0.01),
    sum(abs(horseshoe_model$coef) < 0.01),
    sum(abs(as.vector(coef(glmnet_lasso, s = "lambda.min"))[-1]) < 0.01)
  ),
  Total = p
)

p4 <- ggplot(sparsity_data, aes(x = Method, y = Zero_Coefficients)) +
  geom_col(fill = "steelblue", alpha = 0.7) +
  geom_text(aes(label = Zero_Coefficients), vjust = -0.5) +
  labs(title = "Sparsity: Number of Near-Zero Coefficients",
       x = "Method", y = "Number of Zero Coefficients") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ylim(0, max(sparsity_data$Zero_Coefficients) * 1.2)

# Combine plots
grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)

# ===============================================================================
# KEY DIFFERENCES SUMMARY
# ===============================================================================

cat("\n=== KEY DIFFERENCES SUMMARY ===\n\n")

cat("1. PRIOR DISTRIBUTIONS:\n")
cat("   - Ridge: Gaussian prior β ~ N(0, σ²/α)\n")
cat("   - Lasso: Laplace prior β ~ Laplace(0, λ)\n")
cat("   - Horseshoe: Heavy-tailed prior with adaptive shrinkage\n\n")

cat("2. SHRINKAGE BEHAVIOR:\n")
cat("   - Ridge: Shrinks all coefficients proportionally\n")
cat("   - Lasso: Can shrink coefficients to exactly zero (sparsity)\n")
cat("   - Horseshoe: Adaptive - little shrinkage for large coefficients\n\n")

cat("3. WHEN TO USE:\n")
cat("   - Ridge: Many small relevant effects\n")
cat("   - Lasso: Sparse problems with few important features\n")
cat("   - Horseshoe: Unknown sparsity, adaptive to signal strength\n\n")

cat("4. COMPUTATIONAL ASPECTS:\n")
cat("   - Ridge: Closed-form solution, fastest\n")
cat("   - Lasso: Iterative, moderate complexity\n")
cat("   - Horseshoe: Most complex, requires MCMC for full Bayesian\n\n")

# Performance summary
best_mse <- which.min(errors$MSE)
best_sparsity <- which.max(errors$Sparsity)

cat("5. RESULTS ON THIS DATA:\n")
cat(sprintf("   - Best coefficient recovery: %s (MSE = %.4f)\n", 
            errors$Method[best_mse], errors$MSE[best_mse]))
cat(sprintf("   - Most sparse solution: %s (%d zero coefficients)\n", 
            errors$Method[best_sparsity], errors$Sparsity[best_sparsity]))
cat(sprintf("   - True sparsity: %d zero coefficients\n", sum(true_coef == 0)))

cat("\n=== PRACTICAL RECOMMENDATIONS ===\n")
cat("- Use Bayesian Ridge when you expect many small effects\n")
cat("- Use Bayesian Lasso when you expect sparsity\n")
cat("- Use Horseshoe when sparsity is unknown and you want adaptivity\n")
cat("- All provide uncertainty quantification unlike classical methods\n")