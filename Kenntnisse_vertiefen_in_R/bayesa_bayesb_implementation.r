# ===============================================================================
# BAYESA AND BAYESB METHODS
# Complete implementation for genomic prediction and breeding applications
# ===============================================================================

library(ggplot2)
library(gridExtra)
library(MASS)  # for multivariate normal

set.seed(123)

cat("=== UNDERSTANDING BAYESA AND BAYESB ===\n\n")

cat("WHAT ARE BAYESA AND BAYESB?\n")
cat("These are Bayesian methods specifically designed for:\n")
cat("• Genomic prediction in plant/animal breeding\n")
cat("• Predicting traits from many genetic markers (SNPs)\n")
cat("• Handling situations where p >> n (more markers than individuals)\n\n")

cat("KEY DIFFERENCES:\n")
cat("BayesA: Assumes ALL markers have some effect (no sparsity)\n")
cat("BayesB: Assumes only SOME markers have effects (includes sparsity)\n\n")

cat("TYPICAL APPLICATIONS:\n")
cat("• Predicting milk yield from cow DNA\n")
cat("• Predicting crop yield from plant genetics\n")
cat("• Estimating disease resistance from genetic markers\n")
cat("• Breeding value estimation\n\n")

# ===============================================================================
# MATHEMATICAL BACKGROUND
# ===============================================================================

cat("=== MATHEMATICAL MODELS ===\n\n")

cat("BASIC MODEL (same for both):\n")
cat("y = Xβ + e\n")
cat("where:\n")
cat("y = phenotype (trait values, e.g., milk yield)\n")
cat("X = genotype matrix (genetic markers/SNPs)\n")
cat("β = marker effects\n")
cat("e = residual error\n\n")

cat("BAYESA PRIOR:\n")
cat("βⱼ ~ N(0, σ²ᵦⱼ)\n")
cat("σ²ᵦⱼ ~ InverseGamma(νᵦ/2, Sᵦνᵦ/2)\n")
cat("→ Each marker has its own variance, but all have some effect\n\n")

cat("BAYESB PRIOR:\n")
cat("βⱼ = δⱼ × αⱼ\n")
cat("δⱼ ~ Bernoulli(π) [0 = no effect, 1 = has effect]\n")
cat("αⱼ ~ N(0, σ²ᵦⱼ) [effect size if δⱼ = 1]\n")
cat("→ Only fraction π of markers have effects\n\n")

# ===============================================================================
# GENERATE REALISTIC GENOMIC DATA
# ===============================================================================

cat("=== GENERATING REALISTIC GENOMIC DATA ===\n")

# Simulate genomic prediction scenario
n_individuals <- 200    # Number of animals/plants
n_markers <- 1000      # Number of SNP markers
n_qtl <- 50           # Number of actual causal variants (for BayesB)

cat(sprintf("Simulating genomic data:\n"))
cat(sprintf("• %d individuals (animals/plants)\n", n_individuals))
cat(sprintf("• %d SNP markers\n", n_markers))
cat(sprintf("• %d truly causal variants\n", n_qtl))

# Generate SNP data (0, 1, 2 copies of allele)
generate_snp_data <- function(n_ind, n_snp, maf_range = c(0.05, 0.5)) {
  # Minor allele frequencies
  maf <- runif(n_snp, maf_range[1], maf_range[2])
  
  # Generate genotypes
  X <- matrix(0, n_ind, n_snp)
  for(j in 1:n_snp) {
    # Binomial sampling for each SNP
    X[, j] <- rbinom(n_ind, 2, maf[j])
  }
  
  colnames(X) <- paste0("SNP_", 1:n_snp)
  return(X)
}

# Generate the genomic data
X <- generate_snp_data(n_individuals, n_markers)

# Create true effects (sparse for realistic simulation)
true_effects <- rep(0, n_markers)
qtl_positions <- sample(1:n_markers, n_qtl)  # Random QTL positions
true_effects[qtl_positions] <- rnorm(n_qtl, 0, 0.5)  # Random effect sizes

# Generate phenotypes
h2 <- 0.6  # Heritability (60% genetic, 40% environmental)
genetic_value <- X %*% true_effects
var_genetic <- var(genetic_value)
var_error <- var_genetic * (1 - h2) / h2
environmental_effect <- rnorm(n_individuals, 0, sqrt(var_error))

y <- genetic_value + environmental_effect

cat(sprintf("• Heritability: %.1f%%\n", h2 * 100))
cat(sprintf("• Genetic variance: %.3f\n", var_genetic))
cat(sprintf("• Environmental variance: %.3f\n", var_error))
cat(sprintf("• Phenotypic variance: %.3f\n\n", var(y)))

# ===============================================================================
# BAYESA IMPLEMENTATION
# ===============================================================================

cat("=== BAYESA IMPLEMENTATION ===\n")

BayesA <- function(X, y, niter = 5000, burnin = 1000) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Initialize parameters
  beta <- rep(0, p)
  sigma2_e <- var(y)
  sigma2_beta <- rep(1, p)
  
  # Hyperparameters
  nu_beta <- 4
  S_beta <- 0.01
  
  # Storage
  beta_samples <- matrix(0, niter - burnin, p)
  sigma2_e_samples <- numeric(niter - burnin)
  
  cat("Running BayesA MCMC...\n")
  
  for(iter in 1:niter) {
    # Update beta (marker effects)
    for(j in 1:p) {
      # Remove effect of current marker
      y_corrected <- y - X[, -j] %*% beta[-j]
      
      # Posterior variance and mean
      var_post <- 1 / (sum(X[, j]^2) / sigma2_e + 1 / sigma2_beta[j])
      mean_post <- var_post * sum(X[, j] * y_corrected) / sigma2_e
      
      # Sample new value
      beta[j] <- rnorm(1, mean_post, sqrt(var_post))
    }
    
    # Update marker-specific variances
    for(j in 1:p) {
      sigma2_beta[j] <- 1 / rgamma(1, (nu_beta + 1) / 2, 
                                   (nu_beta * S_beta + beta[j]^2) / 2)
    }
    
    # Update residual variance
    residuals <- y - X %*% beta
    sigma2_e <- 1 / rgamma(1, n / 2, sum(residuals^2) / 2)
    
    # Store samples after burnin
    if(iter > burnin) {
      idx <- iter - burnin
      beta_samples[idx, ] <- beta
      sigma2_e_samples[idx] <- sigma2_e
    }
    
    if(iter %% 1000 == 0) cat(sprintf("Iteration %d/%d\n", iter, niter))
  }
  
  list(
    beta_mean = colMeans(beta_samples),
    beta_samples = beta_samples,
    sigma2_e_mean = mean(sigma2_e_samples),
    method = "BayesA"
  )
}

# ===============================================================================
# BAYESB IMPLEMENTATION
# ===============================================================================

cat("\n=== BAYESB IMPLEMENTATION ===\n")

BayesB <- function(X, y, pi = 0.95, niter = 5000, burnin = 1000) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Initialize parameters
  beta <- rep(0, p)
  delta <- rbinom(p, 1, 1 - pi)  # Inclusion indicators
  sigma2_e <- var(y)
  sigma2_beta <- rep(1, p)
  
  # Hyperparameters
  nu_beta <- 4
  S_beta <- 0.01
  
  # Storage
  beta_samples <- matrix(0, niter - burnin, p)
  delta_samples <- matrix(0, niter - burnin, p)
  sigma2_e_samples <- numeric(niter - burnin)
  
  cat(sprintf("Running BayesB MCMC (π = %.3f)...\n", pi))
  
  for(iter in 1:niter) {
    # Update beta and delta jointly
    for(j in 1:p) {
      # Remove effect of current marker
      y_corrected <- y - X[, -j] %*% beta[-j]
      
      # Calculate probabilities for delta = 0 and delta = 1
      # P(delta_j = 0)
      log_prob_0 <- log(pi)
      
      # P(delta_j = 1)
      var_post <- 1 / (sum(X[, j]^2) / sigma2_e + 1 / sigma2_beta[j])
      mean_post <- var_post * sum(X[, j] * y_corrected) / sigma2_e
      
      log_prob_1 <- log(1 - pi) + 0.5 * log(var_post) + 
                    0.5 * mean_post^2 / var_post
      
      # Sample delta
      prob_1 <- 1 / (1 + exp(log_prob_0 - log_prob_1))
      delta[j] <- rbinom(1, 1, prob_1)
      
      # Sample beta
      if(delta[j] == 1) {
        beta[j] <- rnorm(1, mean_post, sqrt(var_post))
      } else {
        beta[j] <- 0
      }
    }
    
    # Update marker-specific variances (only for included markers)
    for(j in 1:p) {
      if(delta[j] == 1) {
        sigma2_beta[j] <- 1 / rgamma(1, (nu_beta + 1) / 2, 
                                     (nu_beta * S_beta + beta[j]^2) / 2)
      }
    }
    
    # Update residual variance
    residuals <- y - X %*% beta
    sigma2_e <- 1 / rgamma(1, n / 2, sum(residuals^2) / 2)
    
    # Store samples after burnin
    if(iter > burnin) {
      idx <- iter - burnin
      beta_samples[idx, ] <- beta
      delta_samples[idx, ] <- delta
      sigma2_e_samples[idx] <- sigma2_e
    }
    
    if(iter %% 1000 == 0) cat(sprintf("Iteration %d/%d\n", iter, niter))
  }
  
  list(
    beta_mean = colMeans(beta_samples),
    delta_mean = colMeans(delta_samples),
    beta_samples = beta_samples,
    delta_samples = delta_samples,
    sigma2_e_mean = mean(sigma2_e_samples),
    method = "BayesB"
  )
}

# ===============================================================================
# FIT BOTH MODELS
# ===============================================================================

cat("\n=== FITTING BOTH MODELS ===\n")

# Split data for validation
train_idx <- sample(1:n_individuals, 0.7 * n_individuals)
test_idx <- setdiff(1:n_individuals, train_idx)

X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test <- X[test_idx, ]
y_test <- y[test_idx]

cat(sprintf("Training set: %d individuals\n", length(train_idx)))
cat(sprintf("Test set: %d individuals\n\n", length(test_idx)))

# Standardize markers
X_train_scaled <- scale(X_train)
X_test_scaled <- scale(X_test, center = attr(X_train_scaled, "scaled:center"),
                       scale = attr(X_train_scaled, "scaled:scale"))

# Fit models (using smaller niter for demo - increase for real analysis)
bayesA_result <- BayesA(X_train_scaled, y_train, niter = 2000, burnin = 500)
bayesB_result <- BayesB(X_train_scaled, y_train, pi = 0.95, niter = 2000, burnin = 500)

# ===============================================================================
# MODEL COMPARISON
# ===============================================================================

cat("\n=== MODEL COMPARISON ===\n")

# Predictions
pred_A <- X_test_scaled %*% bayesA_result$beta_mean
pred_B <- X_test_scaled %*% bayesB_result$beta_mean

# Accuracy metrics
cor_A <- cor(y_test, pred_A)
cor_B <- cor(y_test, pred_B)
mse_A <- mean((y_test - pred_A)^2)
mse_B <- mean((y_test - pred_B)^2)

cat("PREDICTION ACCURACY:\n")
cat(sprintf("BayesA - Correlation: %.3f, MSE: %.3f\n", cor_A, mse_A))
cat(sprintf("BayesB - Correlation: %.3f, MSE: %.3f\n", cor_B, mse_B))

# Effect size comparison
nonzero_A <- sum(abs(bayesA_result$beta_mean) > 0.01)
nonzero_B <- sum(abs(bayesB_result$beta_mean) > 0.01)
true_nonzero <- sum(abs(true_effects) > 0)

cat(sprintf("\nNUMBER OF NON-ZERO EFFECTS:\n"))
cat(sprintf("True: %d\n", true_nonzero))
cat(sprintf("BayesA: %d\n", nonzero_A))
cat(sprintf("BayesB: %d\n", nonzero_B))

# Variable selection accuracy (for BayesB)
if(exists("bayesB_result$delta_mean")) {
  true_included <- abs(true_effects) > 0
  pred_included_B <- bayesB_result$delta_mean > 0.5
  
  sensitivity <- sum(true_included & pred_included_B) / sum(true_included)
  specificity <- sum(!true_included & !pred_included_B) / sum(!true_included)
  
  cat(sprintf("\nVARIABLE SELECTION (BayesB):\n"))
  cat(sprintf("Sensitivity (true positives): %.3f\n", sensitivity))
  cat(sprintf("Specificity (true negatives): %.3f\n", specificity))
}

# ===============================================================================
# VISUALIZATIONS
# ===============================================================================

cat("\n=== CREATING VISUALIZATIONS ===\n")

# Plot 1: Effect size comparison
effect_comparison <- data.frame(
  Marker = 1:n_markers,
  True = true_effects,
  BayesA = bayesA_result$beta_mean,
  BayesB = bayesB_result$beta_mean,
  QTL = ifelse(abs(true_effects) > 0, "QTL", "Non-QTL")
)

effect_long <- reshape2::melt(effect_comparison[, -5], id.vars = "Marker", 
                             variable.name = "Method", value.name = "Effect")
effect_long$QTL <- rep(effect_comparison$QTL, 3)

p1 <- ggplot(effect_long, aes(x = Marker, y = Effect, color = Method)) +
  geom_point(alpha = 0.6, size = 0.8) +
  labs(title = "Estimated vs True Marker Effects",
       subtitle = "Red points = true QTL positions",
       x = "Marker Position", y = "Effect Size") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5)

# Highlight true QTLs
qtl_data <- effect_comparison[abs(effect_comparison$True) > 0, ]
p1 <- p1 + geom_vline(xintercept = qtl_data$Marker, color = "red", alpha = 0.3)

# Plot 2: Prediction accuracy
pred_data <- data.frame(
  Individual = rep(1:length(y_test), 3),
  Observed = rep(y_test, 3),
  Predicted = c(y_test, pred_A, pred_B),
  Method = rep(c("Perfect", "BayesA", "BayesB"), each = length(y_test))
)

p2 <- ggplot(pred_data[pred_data$Method != "Perfect", ], 
             aes(x = Observed, y = Predicted, color = Method)) +
  geom_point(alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  facet_wrap(~ Method) +
  labs(title = "Prediction Accuracy",
       subtitle = "Points closer to diagonal = better predictions",
       x = "Observed Phenotype", y = "Predicted Phenotype") +
  theme_minimal() +
  theme(legend.position = "none")

# Plot 3: Inclusion probabilities (BayesB only)
if(exists("bayesB_result$delta_mean")) {
  inclusion_data <- data.frame(
    Marker = 1:n_markers,
    Inclusion_Prob = bayesB_result$delta_mean,
    True_QTL = abs(true_effects) > 0
  )
  
  p3 <- ggplot(inclusion_data, aes(x = Marker, y = Inclusion_Prob, color = True_QTL)) +
    geom_point(alpha = 0.7) +
    labs(title = "BayesB: Marker Inclusion Probabilities",
         subtitle = "Red = true QTL, Blue = non-QTL",
         x = "Marker Position", y = "Inclusion Probability") +
    theme_minimal() +
    geom_hline(yintercept = 0.5, linetype = "dashed", alpha = 0.5) +
    scale_color_manual(values = c("FALSE" = "blue", "TRUE" = "red"),
                       name = "True QTL")
} else {
  p3 <- ggplot() + ggtitle("BayesB Inclusion Probabilities Not Available")
}

# Plot 4: MCMC trace plots
trace_data <- data.frame(
  Iteration = rep(1:nrow(bayesA_result$beta_samples), 2),
  Sigma2_e = c(rep(bayesA_result$sigma2_e_mean, nrow(bayesA_result$beta_samples)),
               rep(bayesB_result$sigma2_e_mean, nrow(bayesB_result$beta_samples))),
  Method = rep(c("BayesA", "BayesB"), each = nrow(bayesA_result$beta_samples))
)

p4 <- ggplot(trace_data, aes(x = Iteration, y = Sigma2_e, color = Method)) +
  geom_line(alpha = 0.7) +
  labs(title = "MCMC Convergence: Residual Variance",
       x = "MCMC Iteration", y = "σ²ₑ") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Combine plots
grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)

# ===============================================================================
# PRACTICAL APPLICATIONS
# ===============================================================================

cat("\n=== PRACTICAL APPLICATIONS ===\n\n")

cat("ANIMAL BREEDING:\n")
cat("• Predict milk yield in dairy cows\n")
cat("• Estimate breeding values for meat quality\n")
cat("• Select animals for disease resistance\n")
cat("• Optimize breeding programs\n\n")

cat("PLANT BREEDING:\n")
cat("• Predict crop yield before harvest\n")
cat("• Select for drought tolerance\n")
cat("• Improve nutritional content\n")
cat("• Accelerate variety development\n\n")

cat("HUMAN GENETICS:\n")
cat("• Polygenic risk scores for diseases\n")
cat("• Pharmacogenomics (drug response)\n")
cat("• Complex trait prediction\n")
cat("• Personalized medicine\n\n")

# ===============================================================================
# WHEN TO USE WHICH METHOD
# ===============================================================================

cat("=== WHEN TO USE BAYESA vs BAYESB ===\n\n")

cat("USE BAYESA WHEN:\n")
cat("✓ You believe many markers have small effects\n")
cat("✓ The trait is highly polygenic\n")
cat("✓ You want to capture all genetic variance\n")
cat("✓ Computational simplicity is preferred\n\n")

cat("USE BAYESB WHEN:\n")
cat("✓ You believe few markers have large effects\n")
cat("✓ You want variable selection\n")
cat("✓ The trait may be controlled by major genes\n")
cat("✓ You need to identify important markers\n\n")

cat("HYBRID APPROACHES:\n")
cat("• BayesC: Intermediate between A and B\n")
cat("• BayesR: Multiple variance classes\n")
cat("• Try multiple methods and compare!\n\n")

# ===============================================================================
# COMPUTATIONAL CONSIDERATIONS
# ===============================================================================

cat("=== COMPUTATIONAL TIPS ===\n\n")

cat("FOR REAL APPLICATIONS:\n")
cat("• Use more MCMC iterations (10,000-50,000)\n")
cat("• Longer burnin period (5,000-10,000)\n")
cat("• Monitor convergence diagnostics\n")
cat("• Consider parallel computing\n")
cat("• Use efficient software (BGLR, MTM, etc.)\n\n")

cat("AVAILABLE R PACKAGES:\n")
cat("• BGLR: Comprehensive Bayesian genomic models\n")
cat("• MTM: Multi-trait models\n")
cat("• rrBLUP: Ridge regression BLUP\n")
cat("• EMMREML: Efficient mixed models\n\n")

# ===============================================================================
# SUMMARY
# ===============================================================================

cat("=== SUMMARY ===\n\n")

cat("KEY TAKEAWAYS:\n")
cat("1. BayesA and BayesB are specialized for genomic prediction\n")
cat("2. Main difference: BayesB includes variable selection\n")
cat("3. Choice depends on genetic architecture assumptions\n")
cat("4. Both provide uncertainty quantification\n")
cat("5. Widely used in breeding and genetics research\n\n")

cat("RESULTS FROM THIS SIMULATION:\n")
cat(sprintf("• BayesA correlation: %.3f\n", cor_A))
cat(sprintf("• BayesB correlation: %.3f\n", cor_B))
cat(sprintf("• Better method: %s\n", ifelse(cor_A > cor_B, "BayesA", "BayesB")))
cat(sprintf("• BayesB identified %d/%d true QTLs\n", 
            ifelse(exists("sensitivity"), round(sensitivity * true_nonzero), "?"), 
            true_nonzero))

cat("\nYour turn: Try these methods on real genomic data!\n")
