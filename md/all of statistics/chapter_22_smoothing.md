# Chapter 22 - Smoothing Using Orthogonal Functions

## 22.1 Introduction

In Chapter 21, we studied nonparametric curve estimation using kernels and local polynomial methods. These approaches estimate functions by taking weighted averages of nearby observations. In this chapter, we explore an alternative approach: **smoothing using orthogonal functions**.

The key idea is to approximate an unknown function $f$ by projecting it onto a finite-dimensional space spanned by orthogonal basis functions. This approach includes several important methods:

- **Fourier series** and **trigonometric polynomials**
- **Wavelets**
- **Splines** (viewed as projections onto spline spaces)
- **Polynomial regression** (as a special case)

## 22.2 Orthogonal Functions and Projections

### 22.2.1 Basic Setup

Let $\mathcal{H}$ be a **Hilbert space** of functions with inner product $\langle f, g \rangle$. Suppose we have an orthonormal basis $\{\phi_1, \phi_2, \ldots\}$ such that:

$$\langle \phi_j, \phi_k \rangle = \begin{cases} 
1 & \text{if } j = k \\
0 & \text{if } j \neq k
\end{cases}$$

For any function $f \in \mathcal{H}$, we can write:

$$f = \sum_{j=1}^{\infty} \theta_j \phi_j$$

where $\theta_j = \langle f, \phi_j \rangle$ are the **Fourier coefficients**.

### 22.2.2 Finite-Dimensional Approximation

In practice, we approximate $f$ using only the first $J$ terms:

$$f_J(x) = \sum_{j=1}^{J} \theta_j \phi_j(x)$$

The **projection theorem** tells us that $f_J$ is the best approximation to $f$ in the span of $\{\phi_1, \ldots, \phi_J\}$ in the sense that:

$$f_J = \arg\min_{g \in \text{span}\{\phi_1, \ldots, \phi_J\}} \|f - g\|^2$$

## 22.3 Estimation

### 22.3.1 The Estimation Problem

Given observations $(X_1, Y_1), \ldots, (X_n, Y_n)$ where $Y_i = f(X_i) + \epsilon_i$ and $\mathbb{E}[\epsilon_i] = 0$, we want to estimate $f$.

The **method of moments estimator** for the coefficients is:

$$\hat{\theta}_j = \frac{1}{n} \sum_{i=1}^{n} Y_i \phi_j(X_i)$$

This gives us the estimator:

$$\hat{f}_J(x) = \sum_{j=1}^{J} \hat{\theta}_j \phi_j(x)$$

### 22.3.2 Properties

**Theorem 22.1** (Bias and Variance). Suppose $Y_i = f(X_i) + \epsilon_i$ where $\epsilon_i \sim N(0, \sigma^2)$ are independent, and $X_1, \ldots, X_n$ are fixed design points. Then:

1. **Bias**: $\mathbb{E}[\hat{f}_J(x)] - f(x) = f(x) - f_J(x)$ where $f_J$ is the best $J$-term approximation
2. **Variance**: $\text{Var}(\hat{f}_J(x)) = \frac{\sigma^2}{n} \sum_{j=1}^{J} \phi_j^2(x)$

The **Mean Squared Error** is:

$$\text{MSE}(x) = \text{Bias}^2 + \text{Variance} = |f(x) - f_J(x)|^2 + \frac{\sigma^2}{n} \sum_{j=1}^{J} \phi_j^2(x)$$

## 22.4 Choosing the Number of Terms

The choice of $J$ involves the usual **bias-variance tradeoff**:
- Small $J$: low variance, high bias
- Large $J$: high variance, low bias

### 22.4.1 Cross-Validation

We can use **cross-validation** to choose $J$:

$$\text{CV}(J) = \frac{1}{n} \sum_{i=1}^{n} \left(Y_i - \hat{f}_{J}^{(-i)}(X_i)\right)^2$$

where $\hat{f}_{J}^{(-i)}$ is the estimate computed without the $i$-th observation.

### 22.4.2 Generalized Cross-Validation

For orthogonal basis functions, we have the convenient **Generalized Cross-Validation** (GCV) criterion:

$$\text{GCV}(J) = \frac{n \sum_{i=1}^{n}(Y_i - \hat{f}_J(X_i))^2}{(n - J)^2}$$

## 22.5 Fourier Series

### 22.5.1 Trigonometric Basis

For functions on $[0, 2\pi]$, we can use the trigonometric basis:

$$\phi_0(x) = \frac{1}{\sqrt{2\pi}}, \quad \phi_{2j-1}(x) = \frac{\cos(jx)}{\sqrt{\pi}}, \quad \phi_{2j}(x) = \frac{\sin(jx)}{\sqrt{\pi}}$$

The Fourier series approximation is:

$$f_J(x) = \frac{a_0}{2} + \sum_{j=1}^{J/2} \left(a_j \cos(jx) + b_j \sin(jx)\right)$$

### 22.5.2 Discrete Fourier Transform

When the design points are equally spaced, we can use the **Fast Fourier Transform** (FFT) for efficient computation.

**Example 22.1** (Periodic Signal). Consider estimating a periodic function:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data
n = 100
x = np.linspace(0, 2*np.pi, n)
f_true = lambda t: np.sin(2*t) + 0.5*np.cos(5*t)
y = f_true(x) + 0.2*np.random.normal(size=n)

# Fourier approximation
def fourier_fit(x, y, J):
    n = len(x)
    a0 = 2*np.mean(y)
    
    coeffs = []
    for j in range(1, J+1):
        aj = 2*np.mean(y * np.cos(j*x))
        bj = 2*np.mean(y * np.sin(j*x))
        coeffs.append((aj, bj))
    
    def fitted_function(t):
        result = a0/2
        for j, (aj, bj) in enumerate(coeffs, 1):
            result += aj*np.cos(j*t) + bj*np.sin(j*t)
        return result
    
    return fitted_function

# Fit with different numbers of terms
x_fine = np.linspace(0, 2*np.pi, 200)
f_hat_5 = fourier_fit(x, y, 5)
f_hat_10 = fourier_fit(x, y, 10)
```

## 22.6 Wavelets

### 22.6.1 Wavelet Basis

**Wavelets** provide a basis that is localized in both time and frequency. The basic idea is to use:

- A **father wavelet** $\phi(x)$ (scaling function)
- A **mother wavelet** $\psi(x)$

The wavelet basis consists of:

$$\phi_{j,k}(x) = 2^{j/2} \phi(2^j x - k), \quad \psi_{j,k}(x) = 2^{j/2} \psi(2^j x - k)$$

where $j$ controls the **scale** and $k$ controls the **location**.

### 22.6.2 Multiresolution Analysis

Wavelets provide a **multiresolution analysis**:
- Low-resolution approximation using scaling functions
- Detail coefficients using wavelets at different scales

The estimator has the form:

$$\hat{f}(x) = \sum_{k} \alpha_{J,k} \phi_{J,k}(x) + \sum_{j=1}^{J} \sum_{k} \beta_{j,k} \psi_{j,k}(x)$$

### 22.6.3 Thresholding

A key advantage of wavelets is that we can perform **thresholding**:

$$\tilde{\beta}_{j,k} = \text{threshold}(\hat{\beta}_{j,k}, \lambda)$$

Common thresholding rules include:
- **Hard thresholding**: $\text{threshold}(x, \lambda) = x \cdot I(|x| > \lambda)$
- **Soft thresholding**: $\text{threshold}(x, \lambda) = \text{sign}(x)(|x| - \lambda)_+$

## 22.7 Splines as Projections

### 22.7.1 Spline Spaces

A **spline** of degree $d$ with knots $t_1 < \cdots < t_K$ is a piecewise polynomial that is $d-1$ times continuously differentiable.

The space of splines can be represented using a **B-spline basis** $\{B_1, \ldots, B_{K+d+1}\}$.

### 22.7.2 Smoothing Splines

The **smoothing spline** minimizes:

$$\sum_{i=1}^{n} (Y_i - f(X_i))^2 + \lambda \int (f''(x))^2 dx$$

This can be viewed as a projection onto an infinite-dimensional space of functions with finite second derivative.

## 22.8 Risk Properties

### 22.8.1 Minimax Rates

For functions in a **Sobolev space** $W^s$ (functions with $s$ bounded derivatives), the minimax risk is:

$$\inf_{\hat{f}} \sup_{f \in W^s} \mathbb{E}\|\hat{f} - f\|^2 \asymp n^{-2s/(2s+1)}$$

Both **wavelets** and **smoothing splines** achieve this optimal rate.

### 22.8.2 Adaptive Methods

**Adaptive methods** can achieve near-optimal rates without knowing the smoothness $s$ in advance. Examples include:
- Wavelet thresholding
- Model selection via cross-validation
- Penalized likelihood methods

## 22.9 Practical Considerations

### 22.9.1 Boundary Effects

Orthogonal function methods can suffer from **boundary effects** when the function domain is finite. Solutions include:
- Boundary correction methods
- Reflection/extension techniques
- Boundary-adapted bases

### 22.9.2 Computational Aspects

**Advantages**:
- FFT makes Fourier methods very fast
- Wavelets have fast transforms
- Splines lead to banded systems

**Disadvantages**:
- May require many terms for good approximation
- Choice of basis can be crucial

## 22.10 Exercises

**Exercise 22.1**. Show that for the Fourier basis on $[0, 2\pi]$, the functions $\{1, \cos(x), \sin(x), \cos(2x), \sin(2x), \ldots\}$ are orthogonal with respect to the inner product $\langle f, g \rangle = \int_0^{2\pi} f(x)g(x) dx$.

**Exercise 22.2**. Implement Fourier series estimation for the function $f(x) = x(2\pi - x)$ on $[0, 2\pi]$ with Gaussian noise. Plot the estimates for different values of $J$ and compute the MSE.

**Exercise 22.3**. Compare kernel smoothing (from Chapter 21) with Fourier series estimation on a periodic function. Which method performs better and why?

**Exercise 22.4**. Show that for orthogonal functions, the cross-validation score can be computed efficiently without refitting the model $n$ times.

**Exercise 22.5**. Implement soft thresholding for wavelet coefficients. Apply it to denoise a function with sharp discontinuities and compare the results to kernel smoothing.

**Exercise 22.6**. Prove that the smoothing spline minimizer of $\sum_{i=1}^{n} (Y_i - f(X_i))^2 + \lambda \int (f''(x))^2 dx$ is a natural cubic spline with knots at the observed $X_i$ values.

## 22.11 Bibliographic Remarks

Orthogonal function methods are covered in detail in:
- **Eubank (1999)**: Nonparametric Regression and Spline Smoothing
- **Vidakovic (1999)**: Statistical Modeling by Wavelets
- **Daubechies (1992)**: Ten Lectures on Wavelets

The theory of minimax rates was developed by **Pinsker (1980)** and **Nussbaum (1985)**. Adaptive wavelet methods were pioneered by **Donoho and Johnstone (1994, 1995)**.

Computational aspects of splines are covered in **de Boor (2001)**: A Practical Guide to Splines.
