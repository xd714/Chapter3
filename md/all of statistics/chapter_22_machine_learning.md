# Chapter 22: Machine Learning and Modern Methods - Mathematical Explanations (Completion)

## 22.14 Fairness in Machine Learning (Continued)

### Statistical Parity
**Demographic parity:** P(Ŷ = 1|A = 0) = P(Ŷ = 1|A = 1)
**Equalized odds:** P(Ŷ = 1|Y = y, A = a) same for all a, y
**Calibration:** P(Y = 1|Ŷ = 1, A = a) same for all a

### Impossibility Results
**Theorem:** It is generally impossible to satisfy all fairness criteria simultaneously.

**Trade-offs:**
- Demographic parity vs accuracy
- Individual fairness vs group fairness
- Short-term fairness vs long-term outcomes

### Algorithmic Interventions
**Pre-processing:** Modify training data to remove bias
**In-processing:** Modify algorithm to incorporate fairness constraints
**Post-processing:** Adjust model outputs to satisfy fairness criteria

**Fairness-aware algorithms:**
- Constrained optimization with fairness constraints
- Adversarial debiasing
- Fair representation learning

## 22.15 Interpretability and Explainability

### Local vs Global Explanations
**Local:** Explain individual predictions
**Global:** Explain overall model behavior

### Model-Agnostic Methods
**LIME (Local Interpretable Model-agnostic Explanations):**
1. Perturb input around instance of interest
2. Train interpretable model on perturbed data
3. Use interpretable model to explain prediction

**SHAP (SHapley Additive exPlanations):**
Based on cooperative game theory
```
φᵢ = ∑_{S⊆N\{i}} |S|!(|N|-|S|-1)!/|N|! [f(S∪{i}) - f(S)]
```

### Attention Mechanisms
**Attention weights:** Show which inputs model focuses on
**Self-attention:** Relate different positions within sequence
**Cross-attention:** Relate positions across sequences

## 22.16 Deep Learning Advanced Topics

### Convolutional Neural Networks
**Convolution operation:**
```
(f * g)(t) = ∑ₘ f(m)g(t-m)
```

**Key properties:**
- Translation invariance
- Parameter sharing
- Hierarchical feature learning

### Recurrent Neural Networks
**Basic RNN:**
```
hₜ = tanh(Wₓₓxₜ + Wₕₕhₜ₋₁ + bₕ)
yₜ = Wₕᵧhₜ + bᵧ
```

**LSTM (Long Short-Term Memory):**
- Forget gate: fₜ = σ(Wf[hₜ₋₁, xₜ] + bf)
- Input gate: iₜ = σ(Wi[hₜ₋₁, xₜ] + bi)
- Output gate: oₜ = σ(Wo[hₜ₋₁, xₜ] + bo)

### Transformers
**Self-attention mechanism:**
```
Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
```

**Multi-head attention:**
```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ)Wᴼ
where headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)
```

## 22.17 Advanced Statistical Learning Theory

### PAC-Bayes Bounds
**PAC-Bayes theorem:** For any prior π and posterior ρ:
```
P(KL(ρ||π) + ln(m/δ) ≥ m·KL(R̂ρ||Rρ)) ≤ δ
```

where R̂ρ is empirical risk under ρ, Rρ is true risk.

### Algorithmic Stability
**Uniform stability:** Algorithm is β-stable if:
```
sup_{z₁,...,zₘ,z'} |ℓ(A(z₁,...,zₘ), z') - ℓ(A(z₁,...,zᵢ₋₁,z',zᵢ₊₁,...,zₘ), z')| ≤ β
```

**Generalization bound:** For β-stable algorithm:
```
E[R(Â)] - R̂(Â) ≤ β
```

### Information-Theoretic Bounds
**Mutual information bound:**
```
E[R(Â) - R̂(Â)] ≤ √(I(A;S)/(2m))
```

where I(A;S) is mutual information between algorithm and training set.

## 22.18 Online Learning

### Regret Minimization
**Regret:** Difference between cumulative loss and best fixed strategy
```
Regret_T = ∑ₜ₌₁ᵀ ℓ(yₜ, ŷₜ) - min_y ∑ₜ₌₁ᵀ ℓ(yₜ, y)
```

### Online Gradient Descent
**Update rule:**
```
wₜ₊₁ = wₜ - ηₜ∇ℓ(yₜ, ⟨wₜ, xₜ⟩)
```

**Regret bound:** O(√T) for convex losses

### Multi-Armed Bandits
**Exploration vs exploitation trade-off**

**UCB (Upper Confidence Bound):**
```
aₜ = argmax_a [μ̂ₐ,ₜ + √(2ln t/nₐ,ₜ)]
```

**Thompson Sampling:** Sample from posterior distribution of rewards

## 22.19 Meta-Learning

### Learning to Learn
**Goal:** Use experience from multiple tasks to learn new tasks quickly

**Model-Agnostic Meta-Learning (MAML):**
```
θ* = argmin_θ ∑ᵢ L_Tᵢ(fθ - α∇_θL_Sᵢ(fθ))
```

### Few-Shot Learning
**Problem:** Learn from very few examples
**Approaches:**
- Metric learning
- Memory-augmented networks
- Gradient-based meta-learning

## 22.20 Computational Aspects

### Distributed Learning
**Data parallelism:** Distribute data across machines
**Model parallelism:** Distribute model across machines

**Federated learning:** Train on decentralized data without sharing

### Approximation Algorithms
**Sketching:** Reduce dimensionality while preserving key properties
**Sampling:** Use subset of data for computation
**Quantization:** Reduce precision of weights/activations

### Hardware Considerations
**GPU acceleration:** Parallel matrix operations
**TPU:** Specialized for tensor operations
**Neuromorphic computing:** Brain-inspired architectures

## 22.21 Evaluation and Validation

### Cross-Validation Variants
**Time series CV:** Respect temporal order
**Nested CV:** Unbiased estimate with hyperparameter tuning
**Group CV:** Account for clustering in data

### Model Comparison
**McNemar's test:** Compare classifiers on same test set
**Friedman test:** Compare multiple algorithms across datasets
**Bayesian model comparison:** Use posterior probabilities

### A/B Testing for ML
**Statistical significance:** Ensure sufficient power
**Multiple testing correction:** Control family-wise error rate
**Sequential testing:** Early stopping rules

## 22.22 Ethics and Societal Impact

### Algorithmic Bias
**Sources of bias:**
- Historical bias in training data
- Representation bias
- Measurement bias
- Evaluation bias

### Privacy-Preserving ML
**Differential privacy:** Add calibrated noise to protect individuals
**Homomorphic encryption:** Compute on encrypted data
**Secure multi-party computation:** Joint computation without sharing data

### Environmental Impact
**Carbon footprint:** Energy consumption of large models
**Green AI:** Efficiency-focused research
**Model compression:** Reduce computational requirements

## Summary

This chapter has covered the mathematical foundations and modern developments in machine learning, bridging traditional statistics with contemporary algorithmic approaches. Key themes include:

1. **Theoretical foundations:** Statistical learning theory, PAC learning, and generalization bounds
2. **Core algorithms:** From linear methods to deep learning architectures
3. **Practical considerations:** Optimization, evaluation, and deployment
4. **Societal aspects:** Fairness, interpretability, and ethical considerations
5. **Emerging directions:** Meta-learning, federated learning, and neuromorphic computing

The field continues to evolve rapidly, with new theoretical insights and practical innovations constantly emerging. The mathematical frameworks presented here provide the foundation for understanding both current methods and future developments in machine learning and artificial intelligence.

## Exercises

**22.1** Prove that the empirical risk minimization principle is equivalent to maximum likelihood estimation for exponential family distributions.

**22.2** Show that ridge regression is equivalent to MAP estimation with a Gaussian prior on the coefficients.

**22.3** Derive the dual formulation of the support vector machine optimization problem.

**22.4** Prove that the VC dimension of linear classifiers in ℝᵈ is d+1.

**22.5** Implement the backpropagation algorithm for a multi-layer perceptron and verify the gradient calculations numerically.

**22.6** Show that demographic parity and equalized odds cannot be satisfied simultaneously when base rates differ across groups.

**22.7** Derive the attention mechanism in transformers and explain its computational complexity.

**22.8** Prove the regret bound for online gradient descent in the convex case.

**22.9** Implement LIME for explaining a random forest classifier on a real dataset.

**22.10** Design a federated learning algorithm that preserves differential privacy.

## References

1. Vapnik, V. (1998). *Statistical Learning Theory*. Wiley.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
4. Shalev-Shwartz, S. & Ben-David, S. (2014). *Understanding Machine Learning*. Cambridge University Press.
5. Mohri, M., Rostamizadeh, A., & Talwalkar, A. (2018). *Foundations of Machine Learning*. MIT Press.
6. Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning*. fairmlbook.org.
7. Molnar, C. (2020). *Interpretable Machine Learning*. christophm.github.io/interpretable-ml-book.
8. Cesa-Bianchi, N. & Lugosi, G. (2006). *Prediction, Learning, and Games*. Cambridge University Press.