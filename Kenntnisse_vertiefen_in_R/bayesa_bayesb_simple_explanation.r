# ===============================================================================
# BAYESA vs BAYESB: SIMPLE EXPLANATION
# Easy-to-understand comparison with practical example
# ===============================================================================

library(ggplot2)
library(glmnet)  # for comparison

set.seed(42)

cat("=== SIMPLE EXPLANATION: BAYESA vs BAYESB ===\n\n")

cat("IMAGINE YOU'RE A PLANT BREEDER:\n")
cat("You have 1000 genetic markers and want to predict crop yield\n")
cat("Question: Do ALL markers matter a little, or only SOME markers matter a lot?\n\n")

cat("BAYESA says: 'ALL markers have some effect'\n")
cat("→ Like saying every gene influences yield a tiny bit\n")
cat("→ Keeps all markers but shrinks their effects\n\n")

cat("BAYESB says: 'Only SOME markers have effects'\n")
cat("→ Like saying only certain genes really matter\n")
cat("→ Can turn off markers completely (set to zero)\n\n")

# ===============================================================================
# SIMPLE EXAMPLE WITH PLANT BREEDING
# ===============================================================================

cat("=== PLANT BREEDING EXAMPLE ===\n")

# Create simple breeding data
n_plants <- 100        # 100 plants in our field
n_markers <- 20        # 20 genetic markers
n_important <- 5       # Only 5 markers really affect yield

cat(sprintf("Our breeding experiment:\n"))
cat(sprintf("• %d plants\n", n_plants))
cat(sprintf("• %d genetic markers\n", n_markers))
cat(sprintf("• %d markers truly affect yield\n\n", n_important))

# Generate genetic markers (0, 1, or 2 copies of each gene variant)
X <- matrix(sample(0:2, n_plants * n_markers, replace = TRUE), 
            nrow = n_plants, ncol = n_markers)
colnames(X) <- paste0("Gene_", 1:n_markers)

# Create true effects - only some markers matter
true_effects <- rep(0, n_markers)
important_genes <- c(2, 5, 8, 12, 17)  # Which genes actually matter
true_effects[important_genes] <- c(0.8, -0.5, 0.6, -0.3, 0.7)  # Their effects

# Generate crop yield
genetic_effect <- X %*% true_effects
environmental_noise <- rnorm(n_plants, 0, 0.5)  # Weather, soil, etc.
yield <- 10 + genetic_effect + environmental_noise  # Base yield + genetics + environment

cat("TRUE GENE EFFECTS:\n")
for(i in 1:n_markers) {
  if(true_effects[i] != 0) {
    cat(sprintf("Gene_%d: %.1f (IMPORTANT!)\n", i, true_effects[i]))
  } else if(i <= 5) {  # Show first few
    cat(sprintf("Gene_%d: %.1f (no effect)\n", i, true_effects[i]))
  }
}
if(sum(true_effects == 0) > 5) {
  cat(sprintf("... and %d more genes with no effect\n\n", sum(true_effects == 0) - 5))
}

# ===============================================================================
# SIMPLE BAYESA IMPLEMENTATION
# ===============================================================================

simple_BayesA <- function(X, y, niter = 1000) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Start with small random effects for all markers
  beta <- rnorm(p, 0, 0.1)
  
  cat("BayesA thinking: 'Every gene matters at least a little'\n")
  
  # Simple iterative estimation
  for(iter in 1:niter) {
    for(j in 1:p) {
      # Update each gene effect
      y_without_j <- y - X[, -j] %*% beta[-j]
      
      # Estimate effect (with shrinkage toward zero)
      numerator <- sum(X[, j] * y_without_j)
      denominator <- sum(X[, j]^2) + 1  # +1 provides shrinkage
      beta[j] <- numerator / denominator
    }
  }
  
  return(list(effects = beta, method = "BayesA"))
}

# ===============================================================================
# SIMPLE BAYESB IMPLEMENTATION
# ===============================================================================

simple_BayesB <- function(X, y, pi = 0.8, niter = 1000) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Start with most effects = 0
  beta <- rep(0, p)
  included <- sample(c(TRUE, FALSE), p, replace = TRUE, prob = c(1-pi, pi))
  
  cat(sprintf("BayesB thinking: 'Only %.0f%% of genes matter'\n", (1-pi)*100))
  
  # Simple iterative estimation
  for(iter in 1:niter) {
    for(j in 1:p) {
      y_without_j <- y - X[, -j] %*% beta[-j]
      
      # Decide: include this gene or not?
      # Calculate evidence for including vs excluding
      if(included[j]) {
        # Currently included - calculate new effect
        numerator <- sum(X[, j] * y_without_j)
        denominator <- sum(X[, j]^2) + 1
        potential_effect <- numerator / denominator
        
        # Decide whether to keep it
        evidence <- abs(potential_effect)
        if(evidence > 0.1) {  # Threshold for keeping
          beta[j] <- potential_effect
        } else {
          beta[j] <- 0
          included[j] <- FALSE
        }
      } else {
        # Currently excluded - should we include it?
        numerator <- sum(X[, j] * y_without_j)
        denominator <- sum(X[, j]^2) + 1
        potential_effect <- numerator / denominator
        
        if(abs(potential_effect) > 0.2) {  # Higher threshold for inclusion
          beta[j] <- potential_effect
          included[j] <- TRUE
        }
      }
    }
  }
  
  return(list(effects = beta, included = included, method = "BayesB"))
}

# ===============================================================================
# FIT BOTH MODELS
# ===============================================================================

cat("\n=== FITTING BOTH MODELS ===\n")

# Standardize the genetic data
X_scaled <- scale(X)

# Fit both models
bayesA_result <- simple_BayesA(X_scaled, yield)
bayesB_result <- simple_BayesB(X_scaled, yield, pi = 0.8)

cat("\nBoth models fitted!\n\n")

# ===============================================================================
# COMPARE RESULTS
# ===============================================================================

cat("=== COMPARING RESULTS ===\n\n")

# Create comparison table
comparison <- data.frame(
  Gene = paste0("Gene_", 1:n_markers),
  True_Effect = true_effects,
  BayesA_Effect = bayesA_result$effects,
  BayesB_Effect = bayesB_result$effects,
  BayesB_Included = bayesB_result$included
)

# Show important genes
cat("IMPORTANT GENES (true effects ≠ 0):\n")
important_rows <- which(abs(true_effects) > 0)
print(round(comparison[important_rows, ], 3))

cat("\nFIRST FEW UNIMPORTANT GENES:\n")
unimportant_rows <- which(true_effects == 0)[1:5]
print(round(comparison[unimportant_rows, ], 3))

# Calculate accuracy
true_nonzero <- sum(abs(true_effects) > 0)
bayesA_nonzero <- sum(abs(bayesA_result$effects) > 0.01)
bayesB_nonzero <- sum(abs(bayesB_result$effects) > 0.01)

cat(sprintf("\nNUMBER OF GENES WITH EFFECTS:\n"))
cat(sprintf("True: %d genes\n", true_nonzero))
cat(sprintf("BayesA: %d genes (keeps all genes)\n", bayesA_nonzero))
cat(sprintf("BayesB: %d genes (selected important ones)\n", bayesB_nonzero))

# Prediction accuracy
pred_A <- X_scaled %*% bayesA_result$effects
pred_B <- X_scaled %*% bayesB_result$effects

cor_A <- cor(yield, pred_A)
cor_B <- cor(yield, pred_B)

cat(sprintf("\nPREDICTION ACCURACY:\n"))
cat(sprintf("BayesA correlation: %.3f\n", cor_A))
cat(sprintf("BayesB correlation: %.3f\n", cor_B))

if(cor_A > cor_B) {
  cat("→ BayesA is better for this data!\n\n")
} else {
  cat("→ BayesB is better for this data!\n\n")
}

# ===============================================================================
# VISUALIZATION
# ===============================================================================

cat("=== CREATING VISUALIZATIONS ===\n")

# Plot 1: Gene effects comparison
plot_data <- data.frame(
  Gene = rep(1:n_markers, 3),
  Effect = c(true_effects, bayesA_result$effects, bayesB_result$effects),
  Method = rep(c("True", "BayesA", "BayesB"), each = n_markers),
  Important = rep(abs(true_effects) > 0, 3)
)

p1 <- ggplot(plot_data, aes(x = Gene, y = Effect, color = Method, shape = Important)) +
  geom_point(size = 3, alpha = 0.8) +
  labs(title = "Gene Effects: True vs Estimated",
       subtitle = "Circles = unimportant genes, Triangles = important genes",
       x = "Gene Number", y = "Effect Size") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  scale_shape_manual(values = c(16, 17), name = "Important Gene")

print(p1)

# Plot 2: Predictions vs actual
pred_data <- data.frame(
  Actual = rep(yield, 2),
  Predicted = c(pred_A, pred_B),
  Method = rep(c("BayesA", "BayesB"), each = length(yield))
)

p2 <- ggplot(pred_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.7, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  facet_wrap(~ Method) +
  labs(title = "Prediction Accuracy",
       subtitle = "Points closer to red line = better predictions",
       x = "Actual Yield", y = "Predicted Yield") +
  theme_minimal()

print(p2)

# Plot 3: Gene selection (BayesB only)
selection_data <- data.frame(
  Gene = 1:n_markers,
  Selected = bayesB_result$included,
  True_Important = abs(true_effects) > 0,
  Effect_Size = abs(bayesB_result$effects)
)

p3 <- ggplot(selection_data, aes(x = Gene, y = Effect_Size, 
                                color = True_Important, shape = Selected)) +
  geom_point(size = 3, alpha = 0.8) +
  labs(title = "BayesB Gene Selection",
       subtitle = "Red = truly important, Blue = unimportant",
       x = "Gene Number", y = "Estimated Effect Size",
       color = "Truly Important", shape = "Selected by BayesB") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_color_manual(values = c("FALSE" = "blue", "TRUE" = "red")) +
  scale_shape_manual(values = c("FALSE" = 1, "TRUE" = 16))

print(p3)

# ===============================================================================
# REAL-WORLD APPLICATIONS
# ===============================================================================

cat("\n=== REAL-WORLD APPLICATIONS ===\n\n")

cat("🐄 DAIRY FARMING:\n")
cat("Problem: Predict milk yield from cow DNA\n")
cat("Data: 50,000 genetic markers, 1,000 cows\n")
cat("BayesA: Assumes all genes affect milk production\n")
cat("BayesB: Assumes only some genes are 'milk genes'\n\n")

cat("🌽 CROP BREEDING:\n")
cat("Problem: Develop drought-resistant corn\n")
cat("Data: 100,000 markers, 500 corn varieties\n")
cat("BayesA: All genes contribute to drought tolerance\n")
cat("BayesB: Only specific genes control drought response\n\n")

cat("🏥 HUMAN MEDICINE:\n")
cat("Problem: Predict disease risk from genetics\n")
cat("Data: 1,000,000 markers, 10,000 patients\n")
cat("BayesA: Many genes each add small disease risk\n")
cat("BayesB: Few genes have major effects on disease\n\n")

# ===============================================================================
# PRACTICAL DECISION GUIDE
# ===============================================================================

cat("=== WHICH METHOD TO USE? ===\n\n")

cat("USE BAYESA WHEN:\n")
cat("✅ You think the trait is 'polygenic' (many genes involved)\n")
cat("✅ Example: Height, weight, intelligence\n")
cat("✅ You want to capture all genetic effects\n")
cat("✅ You have limited computational resources\n\n")

cat("USE BAYESB WHEN:\n")
cat("✅ You think few genes have major effects\n")
cat("✅ Example: Single-gene diseases, major QTLs\n")
cat("✅ You want to identify important genes\n")
cat("✅ You have many more markers than samples\n\n")

cat("PRACTICAL TIP:\n")
cat("Try both methods and see which predicts better!\n")
cat("Real genetics often has both types of effects.\n\n")

# ===============================================================================
# SOFTWARE RECOMMENDATIONS
# ===============================================================================

cat("=== SOFTWARE FOR REAL ANALYSIS ===\n\n")

cat("R PACKAGES:\n")
cat("📦 BGLR - Most comprehensive, user-friendly\n")
cat("📦 MTM - Multi-trait models\n")
cat("📦 rrBLUP - Simpler ridge regression version\n")
cat("📦 EMMREML - Very fast for large datasets\n\n")

cat("EXAMPLE CODE FOR REAL DATA:\n")
cat("```r\n")
cat("library(BGLR)\n")
cat("# BayesA\n")
cat("model_A <- BGLR(y = yield, ETA = list(list(X = markers, model = 'BayesA')))\n")
cat("# BayesB\n")
cat("model_B <- BGLR(y = yield, ETA = list(list(X = markers, model = 'BayesB')))\n")
cat("```\n\n")

# ===============================================================================
# PERFORMANCE TIPS
# ===============================================================================

cat("=== PERFORMANCE TIPS ===\n\n")

cat("FOR LARGE DATASETS:\n")
cat("⚡ Use at least 10,000 MCMC iterations\n")
cat("⚡ Throw away first 5,000 (burn-in)\n")
cat("⚡ Check convergence diagnostics\n")
cat("⚡ Consider parallel computing\n")
cat("⚡ Start with smaller subset to test\n\n")

cat("MEMORY MANAGEMENT:\n")
cat("💾 Large marker matrices need lots of RAM\n")
cat("💾 Consider data compression\n")
cat("💾 Use efficient matrix storage\n")
cat("💾 Process in chunks if needed\n\n")

# ===============================================================================
# SUMMARY AND TAKEAWAYS
# ===============================================================================

cat("=== KEY TAKEAWAYS ===\n\n")

cat("UNDERSTANDING THE DIFFERENCE:\n")
cat("🔹 BayesA = Shrinks all effects toward zero equally\n")
cat("🔹 BayesB = Can set some effects to exactly zero\n")
cat("🔹 Both are Bayesian (provide uncertainty)\n")
cat("🔹 Both handle p >> n problems well\n\n")

cat("WHEN EACH WORKS BETTER:\n")
cat("🎯 BayesA: Polygenic traits (many small effects)\n")
cat("🎯 BayesB: Oligogenic traits (few large effects)\n")
cat("🎯 Reality: Often somewhere in between\n\n")

cat("PRACTICAL WORKFLOW:\n")
cat("1️⃣ Start with both methods\n")
cat("2️⃣ Compare prediction accuracy\n")
cat("3️⃣ Look at gene selection patterns\n")
cat("4️⃣ Consider biological knowledge\n")
cat("5️⃣ Validate on independent data\n\n")

# ===============================================================================
# RESULTS SUMMARY FOR THIS EXAMPLE
# ===============================================================================

cat("=== RESULTS FOR OUR PLANT BREEDING EXAMPLE ===\n\n")

# Calculate gene selection accuracy for BayesB
true_important <- abs(true_effects) > 0
pred_important <- bayesB_result$included

# True positives, false positives, etc.
tp <- sum(true_important & pred_important)
fp <- sum(!true_important & pred_important)
tn <- sum(!true_important & !pred_important)
fn <- sum(true_important & !pred_important)

sensitivity <- tp / sum(true_important)
specificity <- tn / sum(!true_important)
precision <- tp / (tp + fp)

cat(sprintf("BAYESB GENE SELECTION ACCURACY:\n"))
cat(sprintf("• Found %d/%d truly important genes (sensitivity: %.2f)\n", 
            tp, sum(true_important), sensitivity))
cat(sprintf("• Correctly ignored %d/%d unimportant genes (specificity: %.2f)\n", 
            tn, sum(!true_important), specificity))
cat(sprintf("• %d false discoveries out of %d selected\n", fp, sum(pred_important)))

cat(sprintf("\nPREDICTION PERFORMANCE:\n"))
cat(sprintf("• BayesA: r = %.3f (uses all %d genes)\n", cor_A, n_markers))
cat(sprintf("• BayesB: r = %.3f (uses %d selected genes)\n", cor_B, sum(pred_important)))

if(cor_B > cor_A) {
  cat("\n✅ BayesB wins! Gene selection improved prediction.\n")
} else {
  cat("\n✅ BayesA wins! Using all genes was better.\n")
}

cat("\nThis shows why you should try both methods on your real data!\n")