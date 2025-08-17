# Chapter 24 - Simulation Methods

## 24.1 Introduction

Simulation methods have become indispensable tools in modern statistics. Many statistical problems involve complex models where analytical solutions are intractable, confidence intervals are difficult to compute, or sampling distributions are unknown. In such cases, **simulation methods** provide powerful alternatives for:

- Computing integrals and expectations
- Generating samples from complex distributions
- Estimating sampling distributions
- Conducting statistical inference
- Validating theoretical results

This chapter covers the fundamental simulation techniques every statistician should know: basic random number generation, Monte Carlo integration, the bootstrap, and Markov Chain Monte Carlo (MCMC) methods.

## 24.2 Random Number Generation

### 24.2.1 Uniform Random Numbers

All simulation methods rely on generating **pseudo-random numbers** from the uniform distribution $\text{Uniform}(0,1)$. Modern computers use **linear congruential generators** or more sophisticated algorithms.

**Definition 24.1**. A sequence $U_1, U_2, \ldots$ is called a sequence of **pseudo-random numbers** if it appears to be an independent sequence from $\text{Uniform}(0,1)$.

In practice, we assume we have access to $U_1, U_2, \ldots \stackrel{\text{iid}}{\sim} \text{Uniform}(0,1)$.

### 24.2.2 Inverse Transform Method

The **inverse transform method** generates random variables from any distribution with known CDF.

**Theorem 24.2** (Inverse Transform). Let $F$ be a CDF and let $F^{-1}(u) = \inf\{x : F(x) \geq u\}$ be the **quantile function**. If $U \sim \text{Uniform}(0,1)$, then $X = F^{-1}(U)$ has CDF $F$.

**Example 24.1** (Exponential Distribution). To generate $X \sim \text{Exp}(\lambda)$:
1. Generate $U \sim \text{Uniform}(0,1)$
2. Return $X = -\frac{1}{\lambda} \log(1-U)$

```python
import numpy as np

def generate_exponential(lam, n):
    """Generate n samples from Exponential(lam)"""
    U = np.random.uniform(0, 1, n)
    return -np.log(1 - U) / lam

# Example
samples = generate_exponential(2.0, 1000)
```

### 24.2.3 Acceptance-Rejection Method

When the inverse of the CDF is not available or difficult to compute, we can use **acceptance-rejection**.

**Algorithm 24.3** (Acceptance-Rejection). To generate $X$ with density $f$:
1. Find a density $g$ (easy to sample from) and constant $c$ such that $f(x) \leq c g(x)$ for all $x$
2. Generate $Y$ from $g$ and $U \sim \text{Uniform}(0,1)$
3. If $U \leq \frac{f(Y)}{c g(Y)}$, return $X = Y$; otherwise go to step 2

**Example 24.2** (Beta Distribution). To generate $X \sim \text{Beta}(2,3)$:

```python
def generate_beta_23(n):
    """Generate n samples from Beta(2,3) using acceptance-rejection"""
    samples = []
    while len(samples) < n:
        # Use Uniform(0,1) as proposal
        Y = np.random.uniform(0, 1)
        U = np.random.uniform(0, 1)
        
        # Beta(2,3) density: f(x) = 12x(1-x)^2
        # Maximum is at x=1/3, f_max = 12*(1/3)*(2/3)^2 = 32/27
        c = 32/27
        
        if U <= 12 * Y * (1-Y)**2 / c:
            samples.append(Y)
    
    return np.array(samples)
```

## 24.3 Monte Carlo Integration

### 24.3.1 Basic Monte Carlo

**Monte Carlo integration** estimates integrals using simulation.

**Problem**: Estimate $\theta = \int g(x) f(x) dx = \mathbb{E}[g(X)]$ where $X$ has density $f$.

**Method**: Generate $X_1, \ldots, X_n \stackrel{\text{iid}}{\sim} f$ and use:

$$\hat{\theta}_n = \frac{1}{n} \sum_{i=1}^{n} g(X_i)$$

**Theorem 24.4**. $\hat{\theta}_n \to \theta$ almost surely and $\sqrt{n}(\hat{\theta}_n - \theta) \leadsto N(0, \sigma^2)$ where $\sigma^2 = \text{Var}(g(X))$.

The **Monte Carlo standard error** is $\text{se} = \sqrt{\hat{\sigma}^2/n}$ where $\hat{\sigma}^2 = \frac{1}{n-1} \sum_{i=1}^{n} (g(X_i) - \hat{\theta}_n)^2$.

**Example 24.3** (Estimating $\pi$). We can estimate $\pi$ by noting that:

$$\pi = 4 \int_0^1 \sqrt{1-x^2} dx = 4 \mathbb{E}[\sqrt{1-U^2}]$$

where $U \sim \text{Uniform}(0,1)$.

```python
def estimate_pi(n):
    """Estimate pi using Monte Carlo"""
    U = np.random.uniform(0, 1, n)
    theta_hat = 4 * np.mean(np.sqrt(1 - U**2))
    
    # Standard error
    g_values = 4 * np.sqrt(1 - U**2)
    se = np.std(g_values) / np.sqrt(n)
    
    return theta_hat, se

# Example
pi_est, se = estimate_pi(100000)
print(f"π estimate: {pi_est:.4f} ± {1.96*se:.4f}")
```

### 24.3.2 Importance Sampling

When $g(x)$ is large in regions where $f(x)$ is small, basic Monte Carlo can be inefficient. **Importance sampling** uses a different distribution for sampling.

**Method**: Choose a density $h$ and estimate:

$$\theta = \int g(x) f(x) dx = \int g(x) \frac{f(x)}{h(x)} h(x) dx = \mathbb{E}_h\left[g(X) \frac{f(X)}{h(X)}\right]$$

Generate $X_1, \ldots, X_n \stackrel{\text{iid}}{\sim} h$ and use:

$$\hat{\theta}_n = \frac{1}{n} \sum_{i=1}^{n} g(X_i) \frac{f(X_i)}{h(X_i)}$$

The ratio $w(x) = f(x)/h(x)$ is called the **importance weight**.

**Example 24.4** (Tail Probability). Estimate $\mathbb{P}(Z > 4)$ where $Z \sim N(0,1)$:

```python
def tail_probability_importance_sampling(n):
    """Estimate P(Z > 4) for Z ~ N(0,1) using importance sampling"""
    # Use exponential distribution shifted by 4 as importance function
    # h(x) = exp(-(x-4)) for x > 4
    Y = np.random.exponential(1, n) + 4
    
    # Importance weights: f(y)/h(y)
    f_y = (1/np.sqrt(2*np.pi)) * np.exp(-Y**2/2)  # Standard normal density
    h_y = np.exp(-(Y-4))  # Shifted exponential density
    weights = f_y / h_y
    
    theta_hat = np.mean(weights)
    se = np.std(weights) / np.sqrt(n)
    
    return theta_hat, se
```

## 24.4 The Bootstrap

The **bootstrap** is a simulation-based method for estimating the sampling distribution of a statistic.

### 24.4.1 The Bootstrap Principle

Given data $X_1, \ldots, X_n$ from an unknown distribution $F$, we want to estimate the distribution of a statistic $T_n = T(X_1, \ldots, X_n)$.

**Bootstrap Principle**: Replace the unknown $F$ with the **empirical distribution** $\hat{F}_n$ and simulate from $\hat{F}_n$.

**Algorithm 24.5** (Nonparametric Bootstrap):
1. Draw $X_1^*, \ldots, X_n^*$ from $\hat{F}_n$ (sample with replacement from $X_1, \ldots, X_n$)
2. Compute $T_n^* = T(X_1^*, \ldots, X_n^*)$
3. Repeat steps 1-2 $B$ times to get $T_1^*, \ldots, T_B^*$
4. Use $T_1^*, \ldots, T_B^*$ to approximate the distribution of $T_n$

### 24.4.2 Bootstrap Confidence Intervals

**Percentile Method**: A $1-\alpha$ confidence interval is:

$$[T_{(\alpha/2)}^*, T_{(1-\alpha/2)}^*]$$

where $T_{(\alpha)}^*$ is the $\alpha$-quantile of $T_1^*, \ldots, T_B^*$.

**Example 24.5** (Bootstrap for Sample Mean):

```python
def bootstrap_mean_ci(data, B=1000, alpha=0.05):
    """Bootstrap confidence interval for the mean"""
    n = len(data)
    bootstrap_means = []
    
    for _ in range(B):
        # Sample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Percentile method
    lower = np.percentile(bootstrap_means, 100 * alpha/2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
    
    return lower, upper, bootstrap_means

# Example
data = np.random.normal(5, 2, 50)
lower, upper, boot_means = bootstrap_mean_ci(data)
print(f"95% Bootstrap CI for mean: [{lower:.3f}, {upper:.3f}]")
```

### 24.4.3 Bootstrap Bias Correction

The bootstrap can estimate and correct for bias.

**Bias Estimate**: $\text{bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}^*] - \hat{\theta}$

**Bias-Corrected Estimator**: $\hat{\theta}_{\text{corrected}} = \hat{\theta} - \widehat{\text{bias}}(\hat{\theta})$

## 24.5 Markov Chain Monte Carlo (MCMC)

When we need to sample from complex, high-dimensional distributions, **Markov Chain Monte Carlo** methods are essential.

### 24.5.1 Markov Chains

**Definition 24.6**. A sequence $X_0, X_1, X_2, \ldots$ is a **Markov chain** if:

$$\mathbb{P}(X_{n+1} = x_{n+1} | X_0 = x_0, \ldots, X_n = x_n) = \mathbb{P}(X_{n+1} = x_{n+1} | X_n = x_n)$$

The **transition kernel** $P(x, y) = \mathbb{P}(X_{n+1} = y | X_n = x)$ characterizes the chain.

**Definition 24.7**. A distribution $\pi$ is **stationary** for a Markov chain if:

$$\pi(y) = \int \pi(x) P(x, y) dx$$

**Theorem 24.8**. Under regularity conditions, if a Markov chain has a unique stationary distribution $\pi$, then:

$$\frac{1}{n} \sum_{i=1}^{n} g(X_i) \to \mathbb{E}_\pi[g(X)]$$

as $n \to \infty$ for any function $g$ with $\mathbb{E}_\pi[|g(X)|] < \infty$.

### 24.5.2 Metropolis-Hastings Algorithm

The **Metropolis-Hastings algorithm** constructs a Markov chain with stationary distribution $\pi$.

**Algorithm 24.9** (Metropolis-Hastings):
1. Start with $X_0$
2. At step $n$, given $X_n = x$:
   - Generate $Y$ from proposal distribution $q(y|x)$
   - Compute acceptance probability: $\alpha(x,y) = \min\left(1, \frac{\pi(y)q(x|y)}{\pi(x)q(y|x)}\right)$
   - Set $X_{n+1} = Y$ with probability $\alpha(x,y)$, otherwise $X_{n+1} = x$

**Example 24.6** (Sampling from Gamma Distribution):

```python
def metropolis_hastings_gamma(alpha, beta, n_samples, sigma=0.5):
    """Sample from Gamma(alpha, beta) using Metropolis-Hastings"""
    # Target density (unnormalized)
    def log_target(x):
        if x <= 0:
            return -np.inf
        return (alpha - 1) * np.log(x) - beta * x
    
    samples = []
    current = alpha / beta  # Start at the mode
    
    for _ in range(n_samples):
        # Propose new state (random walk)
        proposal = current + np.random.normal(0, sigma)
        
        if proposal > 0:
            # Compute log acceptance probability
            log_alpha = log_target(proposal) - log_target(current)
            alpha_prob = min(1, np.exp(log_alpha))
            
            if np.random.uniform() < alpha_prob:
                current = proposal
        
        samples.append(current)
    
    return np.array(samples)

# Example
samples = metropolis_hastings_gamma(2, 1, 5000)
```

### 24.5.3 Gibbs Sampling

**Gibbs sampling** is used when the joint distribution is difficult to sample from but the conditional distributions are easy.

**Algorithm 24.10** (Gibbs Sampling). For $(X_1, X_2, \ldots, X_k) \sim \pi$:
1. Start with $(X_1^{(0)}, \ldots, X_k^{(0)})$
2. At iteration $t$:
   - Sample $X_1^{(t+1)} \sim \pi(X_1 | X_2^{(t)}, \ldots, X_k^{(t)})$
   - Sample $X_2^{(t+1)} \sim \pi(X_2 | X_1^{(t+1)}, X_3^{(t)}, \ldots, X_k^{(t)})$
   - $\vdots$
   - Sample $X_k^{(t+1)} \sim \pi(X_k | X_1^{(t+1)}, \ldots, X_{k-1}^{(t+1)})$

**Example 24.7** (Bivariate Normal):

```python
def gibbs_bivariate_normal(mu1, mu2, sigma1, sigma2, rho, n_samples):
    """Gibbs sampling for bivariate normal"""
    samples = np.zeros((n_samples, 2))
    x1, x2 = 0, 0  # Starting values
    
    for i in range(n_samples):
        # Sample x1 | x2
        mu_cond1 = mu1 + rho * (sigma1/sigma2) * (x2 - mu2)
        sigma_cond1 = sigma1 * np.sqrt(1 - rho**2)
        x1 = np.random.normal(mu_cond1, sigma_cond1)
        
        # Sample x2 | x1
        mu_cond2 = mu2 + rho * (sigma2/sigma1) * (x1 - mu1)
        sigma_cond2 = sigma2 * np.sqrt(1 - rho**2)
        x2 = np.random.normal(mu_cond2, sigma_cond2)
        
        samples[i] = [x1, x2]
    
    return samples
```

## 24.6 MCMC Diagnostics

### 24.6.1 Convergence Assessment

**Trace Plots**: Plot $X_t$ versus $t$ to visually assess mixing.

**Multiple Chains**: Run several chains from different starting points.

**Gelman-Rubin Statistic**: $\hat{R} = \sqrt{\frac{\hat{V}}{W}}$ where:
- $W$ = within-chain variance
- $\hat{V}$ = pooled variance estimate

Convergence is suggested when $\hat{R} \approx 1$.

### 24.6.2 Effective Sample Size

Due to autocorrelation, MCMC samples are not independent. The **effective sample size** is:

$$\text{ESS} = \frac{n}{1 + 2\sum_{k=1}^{\infty} \rho_k}$$

where $\rho_k$ is the lag-$k$ autocorrelation.

## 24.7 High-Dimensional Problems

### 24.7.1 Curse of Dimensionality

As dimension $d$ increases, simulation becomes more challenging:
- Volume concentrates in thin shells
- Typical distances become similar
- Random walks mix slowly

### 24.7.2 Hamiltonian Monte Carlo

**Hamiltonian Monte Carlo (HMC)** uses gradient information to propose better moves:

1. Introduce auxiliary momentum variables
2. Use Hamiltonian dynamics to propose large moves
3. Accept/reject using Metropolis criterion

HMC is particularly effective for smooth, high-dimensional distributions.

## 24.8 Applications

### 24.8.1 Bayesian Inference

MCMC is essential for Bayesian computation when posterior distributions are complex.

**Example**: For the model $Y_i | \theta \sim N(\theta, \sigma^2)$ with prior $\theta \sim N(\mu_0, \tau_0^2)$:

```python
def gibbs_normal_model(y, mu0, tau0_squared, sigma_squared, n_samples):
    """Gibbs sampling for normal model with normal prior"""
    n = len(y)
    samples = {'theta': [], 'sigma_squared': []}
    
    # Initialize
    theta = np.mean(y)
    
    for _ in range(n_samples):
        # Sample theta | sigma_squared, y
        precision = 1/tau0_squared + n/sigma_squared
        mean = (mu0/tau0_squared + n*np.mean(y)/sigma_squared) / precision
        variance = 1/precision
        theta = np.random.normal(mean, np.sqrt(variance))
        
        # Sample sigma_squared | theta, y (if using inverse-gamma prior)
        # This would require specifying a prior for sigma_squared
        
        samples['theta'].append(theta)
        samples['sigma_squared'].append(sigma_squared)
    
    return samples
```

### 24.8.2 Missing Data Imputation

MCMC can handle missing data through **data augmentation**:

1. Treat missing values as parameters
2. Sample missing values from their conditional distribution
3. Sample model parameters given complete data

## 24.9 Practical Considerations

### 24.9.1 Burn-in

Discard initial samples to reduce dependence on starting values.

### 24.9.2 Thinning

Keep every $k$-th sample to reduce autocorrelation (though this reduces effective sample size).

### 24.9.3 Computational Efficiency

- Use efficient proposal distributions
- Implement gradient-based methods when possible
- Consider parallel tempering for multimodal distributions
- Use adaptive algorithms that tune proposals automatically

## 24.10 Exercises

**Exercise 24.1**. Implement the Box-Muller transform to generate normal random variables from uniform random variables.

**Exercise 24.2**. Use Monte Carlo integration to estimate $\int_0^1 e^x dx$ and compare with the analytical answer. How does the standard error decrease with sample size?

**Exercise 24.3**. Generate data from a $t$-distribution with 3 degrees of freedom using acceptance-rejection with a Cauchy proposal distribution.

**Exercise 24.4**. Bootstrap the correlation coefficient. Generate bivariate normal data and construct a bootstrap confidence interval for the correlation.

**Exercise 24.5**. Implement a Metropolis-Hastings sampler for the posterior distribution in a logistic regression model.

**Exercise 24.6**. Use Gibbs sampling to estimate parameters in a normal mixture model with two components.

**Exercise 24.7**. Compare the efficiency of different proposal distributions in Metropolis-Hastings for sampling from a multivariate normal distribution.

**Exercise 24.8**. Implement MCMC diagnostics: trace plots, autocorrelation functions, and the Gelman-Rubin statistic.

## 24.11 Bibliographic Remarks

The foundations of Monte Carlo methods are covered in **Robert and Casella (2004)**: Monte Carlo Statistical Methods. Bootstrap methods were introduced by **Efron (1979)** and are comprehensively covered in **Efron and Tibshirani (1994)**: An Introduction to the Bootstrap.

MCMC methods originated with **Metropolis et al. (1953)** and **Hastings (1970)**. The Gibbs sampler was introduced to statistics by **Geman and Geman (1984)**. Modern developments including Hamiltonian Monte Carlo are covered in **Brooks et al. (2011)**: Handbook of Markov Chain Monte Carlo.

Computational aspects and software implementations are discussed in **Gentle (2009)**: Computational Statistics and **Rizzo (2019)**: Statistical Computing with R.
