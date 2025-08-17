# Chapter 2: Probability - Mathematical Explanations

## Overview
This chapter introduces the fundamental concepts of probability theory, which forms the foundation for all statistical inference. The chapter covers probability spaces, basic probability rules, conditional probability, independence, and Bayes' theorem.

## 2.1 Sample Spaces and Events

### Sample Space (Ω)
The **sample space** Ω is the set of all possible outcomes of an experiment.

**Examples:**
- Coin flip: Ω = {H, T}
- Die roll: Ω = {1, 2, 3, 4, 5, 6}
- Continuous: Ω = [0, 1] for choosing a random number

### Events
An **event** A is a subset of the sample space: A ⊆ Ω

**Types of Events:**
- **Elementary event**: Single outcome {ω}
- **Null event**: ∅ (impossible event)
- **Certain event**: Ω (always occurs)

### Set Operations
- **Union**: A ∪ B (A or B occurs)
- **Intersection**: A ∩ B (both A and B occur)
- **Complement**: A^c (A does not occur)
- **Difference**: A \ B = A ∩ B^c

**De Morgan's Laws:**
- (A ∪ B)^c = A^c ∩ B^c
- (A ∩ B)^c = A^c ∪ B^c

## 2.2 Probability Functions

### Definition
A **probability function** P assigns numbers to events satisfying:

1. **Non-negativity**: P(A) ≥ 0 for all events A
2. **Normalization**: P(Ω) = 1
3. **Countable Additivity**: If A₁, A₂, ... are disjoint, then:
   P(⋃ᵢ Aᵢ) = ∑ᵢ P(Aᵢ)

### Basic Properties
From the axioms, we derive:

- P(∅) = 0
- P(A^c) = 1 - P(A)
- If A ⊆ B, then P(A) ≤ P(B)
- P(A ∪ B) = P(A) + P(B) - P(A ∩ B)

**Inclusion-Exclusion Principle:**
For n events A₁, ..., Aₙ:
```
P(A₁ ∪ ... ∪ Aₙ) = ∑ᵢ P(Aᵢ) - ∑ᵢ<ⱼ P(Aᵢ ∩ Aⱼ) + ∑ᵢ<ⱼ<ₖ P(Aᵢ ∩ Aⱼ ∩ Aₖ) - ... + (-1)^(n+1) P(A₁ ∩ ... ∩ Aₙ)
```

## 2.3 Conditional Probability

### Definition
The **conditional probability** of A given B is:
```
P(A|B) = P(A ∩ B) / P(B), provided P(B) > 0
```

**Interpretation:** P(A|B) is the probability of A occurring given that B has occurred.

### Properties
- 0 ≤ P(A|B) ≤ 1
- P(Ω|B) = 1
- P(A^c|B) = 1 - P(A|B)
- If A₁, A₂, ... are disjoint: P(⋃ᵢ Aᵢ|B) = ∑ᵢ P(Aᵢ|B)

### Multiplication Rule
P(A ∩ B) = P(A|B)P(B) = P(B|A)P(A)

**Chain Rule:** For events A₁, ..., Aₙ:
```
P(A₁ ∩ ... ∩ Aₙ) = P(A₁)P(A₂|A₁)P(A₃|A₁ ∩ A₂)...P(Aₙ|A₁ ∩ ... ∩ Aₙ₋₁)
```

## 2.4 Independence

### Definition
Events A and B are **independent** if:
```
P(A ∩ B) = P(A)P(B)
```

**Equivalent conditions** (when P(B) > 0):
- P(A|B) = P(A)
- P(A|B) = P(A|B^c)

### Properties
- If A and B are independent, then so are:
  - A and B^c
  - A^c and B
  - A^c and B^c

### Mutual Independence
Events A₁, ..., Aₙ are **mutually independent** if for every subset I ⊆ {1, ..., n}:
```
P(⋂ᵢ∈I Aᵢ) = ∏ᵢ∈I P(Aᵢ)
```

**Note:** Pairwise independence does not imply mutual independence.

## 2.5 Bayes' Theorem

### Law of Total Probability
If B₁, ..., Bₙ form a partition of Ω (disjoint and ⋃ᵢ Bᵢ = Ω), then:
```
P(A) = ∑ᵢ P(A|Bᵢ)P(Bᵢ)
```

### Bayes' Theorem
For events A and B with P(A) > 0 and P(B) > 0:
```
P(B|A) = P(A|B)P(B) / P(A)
```

**Extended form:** If B₁, ..., Bₙ partition Ω:
```
P(Bⱼ|A) = P(A|Bⱼ)P(Bⱼ) / ∑ᵢ P(A|Bᵢ)P(Bᵢ)
```

**Interpretation:**
- P(Bⱼ): **prior probability** of Bⱼ
- P(A|Bⱼ): **likelihood** of A given Bⱼ
- P(Bⱼ|A): **posterior probability** of Bⱼ given A

## 2.6 Discrete Probability Models

### Uniform Distribution
If Ω = {ω₁, ..., ωₙ} and all outcomes are equally likely:
```
P({ωᵢ}) = 1/n for all i
```

For any event A:
```
P(A) = |A| / |Ω|
```

### Counting Principles

**Multiplication Principle:** If task 1 can be done in n₁ ways and task 2 in n₂ ways, both tasks can be done in n₁ × n₂ ways.

**Permutations:** Number of ways to arrange n distinct objects:
```
n! = n × (n-1) × ... × 2 × 1
```

**Combinations:** Number of ways to choose k objects from n:
```
C(n,k) = (n choose k) = n! / (k!(n-k)!)
```

### Binomial Coefficient Properties
- (n choose 0) = (n choose n) = 1
- (n choose k) = (n choose n-k)
- (n+1 choose k) = (n choose k) + (n choose k-1)

**Binomial Theorem:**
```
(x + y)ⁿ = ∑ₖ₌₀ⁿ (n choose k) xᵏyⁿ⁻ᵏ
```

## 2.7 Computer Experiments and Simulation

### Law of Large Numbers (Intuitive)
As the number of trials increases, the relative frequency of an event approaches its theoretical probability.

For a sequence of independent trials with success probability p:
```
lim(n→∞) (Number of successes in n trials)/n = p
```

**Simulation Example:** Coin flipping experiments show convergence to theoretical probability p.

### Random Number Generation
- **Pseudo-random numbers**: Deterministic algorithms that produce sequences that appear random
- **Seed**: Initial value that determines the sequence
- **Uniform[0,1]**: Foundation for generating other distributions

## 2.8 Important Theorems and Results

### Bonferroni Inequality
For any events A₁, ..., Aₙ:
```
P(⋃ᵢ Aᵢ) ≤ ∑ᵢ P(Aᵢ)
```

**Union Bound:** This provides an upper bound for the probability of the union.

### Boole's Inequality
```
P(⋃ᵢ₌₁^∞ Aᵢ) ≤ ∑ᵢ₌₁^∞ P(Aᵢ)
```

## 2.9 Applications and Examples

### Medical Testing
Consider a diagnostic test with:
- **Sensitivity**: P(Test+|Disease+) = 0.95
- **Specificity**: P(Test-|Disease-) = 0.90
- **Prevalence**: P(Disease+) = 0.01

Using Bayes' theorem:
```
P(Disease+|Test+) = [P(Test+|Disease+) × P(Disease+)] / P(Test+)
```

Where P(Test+) = P(Test+|Disease+)P(Disease+) + P(Test+|Disease-)P(Disease-)

### Reliability Theory
For systems with components:
- **Series system**: Works if all components work
  P(System works) = ∏ᵢ P(Component i works)
- **Parallel system**: Works if at least one component works
  P(System works) = 1 - ∏ᵢ P(Component i fails)

## Key Insights and Connections

1. **Foundation for Statistics**: Probability theory provides the mathematical framework for statistical inference.

2. **Conditioning**: Understanding conditional probability is crucial for:
   - Bayesian statistics
   - Decision theory
   - Machine learning

3. **Independence**: A fundamental concept that simplifies calculations and appears throughout statistics.

4. **Simulation**: Computer experiments help verify theoretical results and provide intuition.

## Common Pitfalls

1. **Prosecutor's Fallacy**: Confusing P(A|B) with P(B|A)
2. **Base Rate Neglect**: Ignoring prior probabilities in Bayes' theorem
3. **Independence Assumption**: Assuming independence when it doesn't hold
4. **Simpson's Paradox**: Aggregate data can show opposite trends to subgroup data

## Further Reading

This chapter establishes the foundation for:
- Chapter 3: Random Variables
- Chapter 4: Expectation
- Chapter 12: Bayesian Inference
- All subsequent statistical inference topics