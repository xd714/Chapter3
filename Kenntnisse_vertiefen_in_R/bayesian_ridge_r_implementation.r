# Bayesian Ridge Regression Implementation in R
# Load required libraries
library(ggplot2)
library(gridExtra)
library(dplyr)
library(MASS)  # for mvrnorm
library(glmnet)  # for comparison with ridge regression

# Set seed for reproducibility
set.seed(42)

# Function to generate synthetic regression data
generate_regression_data <- function(n_samples = 100, n_features = 10, noise = 0.1) {
  # Generate random features
  X <- matrix(rnorm(n_samples * n_features), nrow = n_samples, ncol = n_features)
  
  # Generate true coefficients
  true_coef <- rnorm(n_features, mean = 0, sd = 2)
  
  # Generate target variable
  y <- X %*% true_coef + rnorm(n_samples, mean = 0, sd = noise)
  
  return(list(X = X, y = as.vector(y), true_coef = true_coef))
}

# Bayesian Ridge Regression Class
BayesianRidge <- function() {
  # Create environment to store model parameters
  model <- new.env()
  
  # Initialize parameters
  model$max_iter <- 300
  model$tol <- 1e-3
  model$alpha_init <- 1.0
  model$lambda_init <- 1.0
  
  # Fit function
  model$fit <- function(X, y, max_iter = 300, tol = 1e-3, alpha_init = 1.0, lambda_init = 1.0) {
    n_samples <- nrow(X)
    n_features <- ncol(X)
    
    # Add intercept term
    X_with_intercept <- cbind(1, X)
    n_features_with_intercept <- n_features + 1
    
    # Initialize parameters
    alpha <- alpha_init  # precision of the prior
    lambda_param <- lambda_init  # precision of the noise
    
    # Initialize mean of posterior
    mean_beta <- rep(0, n_features_with_intercept)
    
    # Storage for convergence tracking
    alpha_history <- numeric(max_iter)
    lambda_history <- numeric(max_iter)
    
    for (iter in 1:max_iter) {
      # E-step: Update posterior distribution of beta
      # Covariance matrix of posterior
      S_inv <- alpha * diag(n_features_with_intercept) + lambda_param * t(X_with_intercept) %*% X_with_intercept
      S <- solve(S_inv)
      
      # Mean of posterior
      mean_beta <- lambda_param * S %*% t(X_with_intercept) %*% y
      
      # M-step: Update hyperparameters
      # Update alpha (precision of prior)
      gamma <- n_features_with_intercept - alpha * sum(diag(S))
      alpha <- gamma / sum(mean_beta^2)
      
      # Update lambda (precision of noise)
      residuals <- y - X_with_intercept %*% mean_beta
      lambda_param <- (n_samples - gamma) / sum(residuals^2)
      
      alpha_history[iter] <- alpha
      lambda_history[iter] <- lambda_param
      
      # Check convergence
      if (iter > 1) {
        if (abs(alpha_history[iter] - alpha_history[iter-1]) < tol &&
            abs(lambda_history[iter] - lambda_history[iter-1]) < tol) {
          alpha_history <- alpha_history[1:iter]
          lambda_history <- lambda_history[1:iter]
          break
        }
      }
    }
    
    # Store results
    model$coef <- as.vector(mean_beta[-1])  # exclude intercept
    model$intercept <- as.vector(mean_beta[1])
    model$alpha <- alpha
    model$lambda <- lambda_param
    model$sigma <- S  # posterior covariance
    model$alpha_history <- alpha_history
    model$lambda_history <- lambda_history
    model$fitted <- TRUE
    
    return(model)
  }
  
  # Predict function
  model$predict <- function(X, return_std = FALSE) {
    if (!model$fitted) {
      stop("Model must be fitted before making predictions")
    }
    
    n_samples <- nrow(X)
    X_with_intercept <- cbind(1, X)
    
    # Mean prediction
    y_pred <- X_with_intercept %*% c(model$intercept, model$coef)
    
    if (return_std) {
      # Predictive variance
      pred_var <- 1/model$lambda + diag(X_with_intercept %*% model$sigma %*% t(X_with_intercept))
      pred_std <- sqrt(pred_var)
      return(list(y_pred = as.vector(y_pred), y_std = pred_std))
    }
    
    return(as.vector(y_pred))
  }
  
  model$fitted <- FALSE
  return(model)
}

# Generate synthetic data
cat("=== Generating Synthetic Data ===\n")
data <- generate_regression_data(n_samples = 100, n_features = 10, noise = 0.5)
X <- data$X
y <- data$y
true_coef <- data$true_coef

# Split data into training and testing
train_indices <- sample(1:nrow(X), size = 0.8 * nrow(X))
X_train <- X[train_indices, ]
y_train <- y[train_indices]
X_test <- X[-train_indices, ]
y_test <- y[-train_indices]

# Standardize features
X_train_mean <- apply(X_train, 2, mean)
X_train_sd <- apply(X_train, 2, sd)

X_train_scaled <- scale(X_train)
X_test_scaled <- scale(X_test, center = X_train_mean, scale = X_train_sd)

# Fit Bayesian Ridge Regression
cat("=== Fitting Bayesian Ridge Regression ===\n")
br_model <- BayesianRidge()
br_model$fit(X_train_scaled, y_train)

# Make predictions
y_pred <- br_model$predict(X_test_scaled)
pred_with_std <- br_model$predict(X_test_scaled, return_std = TRUE)
y_pred_std <- pred_with_std$y_std

# Calculate performance metrics
mse_br <- mean((y_test - y_pred)^2)
r2_br <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)

cat(sprintf("Bayesian Ridge - MSE: %.4f, R²: %.4f\n", mse_br, r2_br))
cat(sprintf("Final hyperparameters: α = %.4f, λ = %.4f\n", br_model$alpha, br_model$lambda))

# Compare with regular Ridge Regression using cross-validation
cat("\n=== Comparing with Regular Ridge Regression ===\n")

# Fit ridge regression with cross-validation
ridge_cv <- cv.glmnet(X_train_scaled, y_train, alpha = 0, nfolds = 5)
ridge_model <- glmnet(X_train_scaled, y_train, alpha = 0, lambda = ridge_cv$lambda.min)
y_pred_ridge <- predict(ridge_model, X_test_scaled)

mse_ridge <- mean((y_test - y_pred_ridge)^2)
r2_ridge <- 1 - sum((y_test - y_pred_ridge)^2) / sum((y_test - mean(y_test))^2)

cat(sprintf("Regular Ridge - MSE: %.4f, R²: %.4f\n", mse_ridge, r2_ridge))
cat(sprintf("Optimal λ from CV: %.4f\n", ridge_cv$lambda.min))

# Create comprehensive visualizations
cat("\n=== Creating Visualizations ===\n")

# Plot 1: Hyperparameter convergence
convergence_data <- data.frame(
  iteration = rep(1:length(br_model$alpha_history), 2),
  value = c(br_model$alpha_history, br_model$lambda_history),
  parameter = rep(c("Alpha (prior precision)", "Lambda (noise precision)"), 
                  each = length(br_model$alpha_history))
)

p1 <- ggplot(convergence_data, aes(x = iteration, y = value, color = parameter)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  labs(title = "Hyperparameter Convergence",
       x = "Iteration", y = "Value",
       color = "Parameter") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Plot 2: Coefficient comparison
coef_comparison <- data.frame(
  feature = 1:length(true_coef),
  true_coef = true_coef,
  bayesian_ridge = br_model$coef,
  ridge = as.vector(coef(ridge_model))[-1]  # exclude intercept
)

coef_long <- reshape2::melt(coef_comparison, id.vars = "feature", 
                           variable.name = "method", value.name = "coefficient")

p2 <- ggplot(coef_long, aes(x = feature, y = coefficient, color = method)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  labs(title = "Coefficient Comparison",
       x = "Feature Index", y = "Coefficient Value",
       color = "Method") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Plot 3: Predictions with uncertainty
pred_data <- data.frame(
  index = 1:length(y_test),
  true_value = y_test,
  prediction = y_pred,
  std_error = y_pred_std,
  lower_bound = y_pred - 1.96 * y_pred_std,
  upper_bound = y_pred + 1.96 * y_pred_std
)

p3 <- ggplot(pred_data, aes(x = index)) +
  geom_ribbon(aes(ymin = lower_bound, ymax = upper_bound), 
              alpha = 0.3, fill = "red") +
  geom_point(aes(y = true_value), color = "blue", size = 2, alpha = 0.7) +
  geom_point(aes(y = prediction), color = "red", size = 2, alpha = 0.7) +
  labs(title = "Predictions with 95% Confidence Intervals",
       x = "Test Sample Index", y = "Target Value") +
  theme_minimal() +
  annotate("text", x = max(pred_data$index) * 0.7, y = max(pred_data$true_value) * 0.9,
           label = "Blue: True values\nRed: Predictions\nGray: 95% CI", 
           hjust = 0, size = 3)

# Plot 4: Residuals vs Uncertainty
residuals <- abs(y_test - y_pred)
residual_data <- data.frame(
  predicted_std = y_pred_std,
  absolute_residuals = residuals
)

correlation <- cor(y_pred_std, residuals)

p4 <- ggplot(residual_data, aes(x = predicted_std, y = absolute_residuals)) +
  geom_point(size = 2, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(title = "Residuals vs Uncertainty",
       x = "Predicted Standard Deviation", y = "Absolute Residuals") +
  theme_minimal() +
  annotate("text", x = max(y_pred_std) * 0.7, y = max(residuals) * 0.9,
           label = sprintf("Correlation: %.3f", correlation), 
           hjust = 0, size = 4)

# Combine all plots
combined_plot <- grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)

# Print comprehensive analysis
cat("\n=== Model Analysis ===\n")
cat("True coefficients:\n")
print(round(true_coef, 4))
cat("\nBayesian Ridge coefficients:\n")
print(round(br_model$coef, 4))
cat("\nRegular Ridge coefficients:\n")
print(round(as.vector(coef(ridge_model))[-1], 4))

# Calculate coefficient recovery metrics
coef_mse_br <- mean((true_coef - br_model$coef)^2)
coef_mse_ridge <- mean((true_coef - as.vector(coef(ridge_model))[-1])^2)

cat(sprintf("\nCoefficient Recovery MSE:\n"))
cat(sprintf("Bayesian Ridge: %.4f\n", coef_mse_br))
cat(sprintf("Regular Ridge: %.4f\n", coef_mse_ridge))

# Test with different noise levels
cat("\n=== Testing Robustness to Noise ===\n")
noise_levels <- c(0.1, 0.5, 1.0, 2.0)
noise_results <- data.frame()

for (noise in noise_levels) {
  # Generate data with different noise levels
  data_noise <- generate_regression_data(n_samples = 100, n_features = 10, noise = noise)
  X_noise <- scale(data_noise$X)
  y_noise <- data_noise$y
  
  # Split data
  train_idx <- sample(1:nrow(X_noise), size = 0.8 * nrow(X_noise))
  X_train_noise <- X_noise[train_idx, ]
  y_train_noise <- y_noise[train_idx]
  X_test_noise <- X_noise[-train_idx, ]
  y_test_noise <- y_noise[-train_idx]
  
  # Fit model
  br_noise <- BayesianRidge()
  br_noise$fit(X_train_noise, y_train_noise)
  y_pred_noise <- br_noise$predict(X_test_noise)
  
  # Calculate metrics
  mse_noise <- mean((y_test_noise - y_pred_noise)^2)
  r2_noise <- 1 - sum((y_test_noise - y_pred_noise)^2) / sum((y_test_noise - mean(y_test_noise))^2)
  
  noise_results <- rbind(noise_results, data.frame(
    noise_level = noise,
    mse = mse_noise,
    r2 = r2_noise,
    alpha = br_noise$alpha,
    lambda = br_noise$lambda
  ))
}

cat("Noise Level\tMSE\t\tR²\t\tα\t\tλ\n")
cat(paste(rep("-", 60), collapse = ""), "\n")
for (i in 1:nrow(noise_results)) {
  cat(sprintf("%.1f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n",
              noise_results$noise_level[i], noise_results$mse[i], 
              noise_results$r2[i], noise_results$alpha[i], noise_results$lambda[i]))
}

# Function to demonstrate usage
demonstrate_usage <- function() {
  cat("\n=== Usage Example ===\n")
  cat("# Create and fit a Bayesian Ridge model:\n")
  cat("model <- BayesianRidge()\n")
  cat("model$fit(X_train, y_train)\n")
  cat("\n# Make predictions:\n")
  cat("predictions <- model$predict(X_test)\n")
  cat("\n# Get predictions with uncertainty:\n")
  cat("pred_with_uncertainty <- model$predict(X_test, return_std = TRUE)\n")
  cat("\n# Access model parameters:\n")
  cat("cat('Alpha:', model$alpha, 'Lambda:', model$lambda)\n")
  cat("print(model$coef)  # coefficients\n")
  cat("print(model$intercept)  # intercept\n")
}

demonstrate_usage()

cat("\n=== Summary ===\n")
cat("Bayesian Ridge Regression provides:\n")
cat("1. Automatic regularization parameter selection\n")
cat("2. Uncertainty quantification for predictions\n")
cat("3. Robustness to overfitting\n")
cat("4. Better coefficient recovery compared to fixed-lambda ridge\n")
cat("\nThe model adapts the regularization strength based on the data,\n")
cat("making it more flexible than traditional ridge regression.\n")