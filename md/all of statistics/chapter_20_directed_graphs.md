# Chapter 20: Directed Graphs and Causal Models - Mathematical Explanations

## Overview
Directed graphs provide a powerful framework for representing and reasoning about causal relationships in complex systems. This chapter covers directed acyclic graphs (DAGs), their connection to probability distributions, causal inference, and applications in understanding cause-and-effect relationships from observational and experimental data.

## 20.1 Introduction to Directed Graphs

### Basic Definitions
A **directed graph** G = (V, E) consists of:
- **Vertices (nodes)** V = {1, 2, ..., p}: Represent variables
- **Directed edges** E ⊆ V × V: Represent direct relationships

**Directed edge:** i → j means i directly influences j
**Parents:** pa(j) = {i : i → j} (direct causes of j)
**Children:** ch(i) = {j : i → j} (direct effects of i)
**Descendants:** de(i) = variables reachable from i following edges
**Ancestors:** an(j) = variables that can reach j following edges

### Directed Acyclic Graphs (DAGs)
A **DAG** is a directed graph with no directed cycles.

**Properties:**
- Can represent causal structures without feedback loops
- Allow for topological ordering of variables
- Enable factorization of joint distributions

### Paths and Connectivity
**Path:** Sequence of edges connecting two nodes
**Directed path:** All edges point in same direction
**Causal path:** Directed path from cause to effect

## 20.2 Probability and DAGs

### Markov Property for DAGs
A probability distribution P satisfies the **Markov property** relative to DAG G if:
```
X_i ⊥ X_{V\de(i)} | X_{pa(i)}
```

for all i ∈ V.

**Interpretation:** Each variable is independent of its non-descendants given its parents.

### Factorization
If P satisfies the Markov property for DAG G, then:
```
P(x₁, ..., x_p) = ∏ᵢ₌₁ᵖ P(xᵢ | x_{pa(i)})
```

**Fundamental result:** DAG structure determines factorization of joint distribution.

### Examples

**Chain:** X → Y → Z
```
P(X, Y, Z) = P(X)P(Y|X)P(Z|Y)
```

**Fork:** X ← Y → Z
```
P(X, Y, Z) = P(Y)P(X|Y)P(Z|Y)
```

**Collider:** X → Y ← Z
```
P(X, Y, Z) = P(X)P(Z)P(Y|X,Z)
```

## 20.3 d-separation

### Definition
Variables A and B are **d-separated** by set C in DAG G if every path between A and B is blocked by C.

**Path blocking rules:**
1. **Chain** i → m → j: Blocked if m ∈ C
2. **Fork** i ← m → j: Blocked if m ∈ C  
3. **Collider** i → m ← j: Blocked if m ∉ C and no descendant of m is in C

### Global Markov Property
If A and B are d-separated by C, then:
```
X_A ⊥ X_B | X_C
```

**Completeness:** All and only the conditional independence relations implied by d-separation hold in any distribution faithful to the DAG.

### Examples of d-separation

**Mediation:** X → M → Y with conditioning set C
- If M ∉ C: X and Y not d-separated (dependence through M)
- If M ∈ C: X and Y d-separated (M blocks path)

**Confounding:** X ← U → Y with U unobserved
- X and Y not d-separated by observed variables
- Association between X and Y even without causal relationship

**Collider bias:** X → C ← Y, conditioning on C
- Without conditioning: X ⊥ Y
- Conditioning on C: X ⫫ Y (creates dependence)

## 20.4 Causal Interpretation

### Causal vs Statistical Models
**Statistical model:** Describes joint distribution P(X₁, ..., X_p)
**Causal model:** Describes what happens under interventions

### Structural Causal Models
**Structural equations:**
```
Xᵢ = fᵢ(X_{pa(i)}, εᵢ), i = 1, ..., p
```

where εᵢ are independent noise terms.

**DAG representation:** Edge i → j if Xⱼ depends on Xᵢ in structural equation.

### Interventions
An **intervention** do(Xᵢ = x) sets Xᵢ = x and removes all incoming edges to Xᵢ.

**Manipulated graph:** G_x with edges into Xᵢ removed
**Post-intervention distribution:**
```
P(X_{V\{i}} | do(Xᵢ = x)) = ∏_{j≠i} P(Xⱼ | X_{pa(j)})
```

evaluated with Xᵢ = x.

## 20.5 Causal Effects

### Average Causal Effect
**Individual causal effect:** Yᵢ(1) - Yᵢ(0) (unobservable)
**Average causal effect:** E[Y(1) - Y(0)]

### Identifiability
A causal effect is **identifiable** if it can be computed from the observational distribution using the causal graph.

### Backdoor Criterion
Set Z satisfies the **backdoor criterion** for causal effect of X on Y if:
1. No node in Z is a descendant of X
2. Z blocks every path from X to Y that contains an arrow into X

**Backdoor adjustment:**
```
P(Y | do(X = x)) = ∑_z P(Y | X = x, Z = z)P(Z = z)
```

### Frontdoor Criterion
Set Z satisfies the **frontdoor criterion** for X → Y if:
1. Z intercepts all directed paths from X to Y
2. There is no backdoor path from X to Z
3. All backdoor paths from Z to Y are blocked by X

### Examples

**Confounding with observed confounder:**
- DAG: Z → X → Y, Z → Y
- Backdoor set: {Z}
- Adjustment: P(Y | do(X)) = ∑_z P(Y | X, Z = z)P(Z = z)

**Mediation:**
- DAG: X → M → Y
- Direct effect: P(Y | do(X = x, M = m))
- Indirect effect: Through mediator M

## 20.6 Causal Discovery

### Problem Statement
**Goal:** Learn causal graph structure from observational data.

**Challenges:**
- Correlation doesn't imply causation
- Multiple graphs can generate same distribution
- Confounding by unobserved variables

### Constraint-based Algorithms
**PC Algorithm:**
1. Start with complete graph
2. Remove edges based on conditional independence tests
3. Orient edges using d-separation rules

**IC Algorithm:** Handles latent confounders
**FCI Algorithm:** More robust to confounders

### Score-based Algorithms
**Approach:** Search over DAGs to maximize score
```
Score(G) = log P(Data | G) - λ·|G|
```

**Challenges:**
- Exponential search space
- Acyclicity constraint
- Score equivalence of different DAGs

### Structural Equation Models
**Assumptions:**
- Linear relationships
- Gaussian noise
- No confounders

**ICA-based methods:** Use non-Gaussianity to identify structure
**LiNGAM:** Linear non-Gaussian additive model

## 20.7 Instrumental Variables

### Definition
Variable Z is an **instrumental variable** for effect of X on Y if:
1. Z affects X (relevance)
2. Z affects Y only through X (exclusion restriction)
3. Z and Y share no confounders (exchangeability)

### Two-Stage Least Squares
**Stage 1:** Regress X on Z: X̂ = α + βZ
**Stage 2:** Regress Y on X̂: Y = γ + δX̂

**Causal effect:** δ estimates causal effect of X on Y

### Weak Instruments
**Problem:** Small correlation between Z and X leads to:
- Large standard errors
- Bias toward OLS estimate
- Poor finite-sample properties

**Solutions:**
- Test instrument strength
- Use robust inference methods
- Consider multiple instruments

## 20.8 Mediation Analysis

### Direct and Indirect Effects
**Total effect:** X → Y (all pathways)
**Direct effect:** X → Y controlling for mediators  
**Indirect effect:** X → M → Y through mediator M

### Mediation Formula
**Natural direct effect:**
```
NDE = E[Y(1, M(0)) - Y(0, M(0))]
```

**Natural indirect effect:**
```
NIE = E[Y(1, M(1)) - Y(1, M(0))]
```

**Total effect:** TE = NDE + NIE

### Identification Conditions
1. No confounding of X → Y
2. No confounding of M → Y  
3. No confounding of X → M
4. No confounder of M → Y affected by X

### Sequential Ignorability
**Assumption:** For mediation analysis:
```
(Y(x', m), M(x)) ⊥ X | C
{Y(x', m)} ⊥ M | X, C
```

## 20.9 Time-varying Confounding

### Longitudinal Data
Variables measured at multiple time points: X₁, L₁, X₂, L₂, ..., Y

**Time-varying confounding:** Lₜ affects both Xₜ₊₁ and Y

### G-methods
**G-computation:** Standardize over confounder distribution
**Inverse probability weighting:** Weight by treatment probability
**G-estimation:** Structural nested models

### Marginal Structural Models
**Model:** E[Y^{x̄}] = β₀ + β₁x̄

where Y^{x̄} is potential outcome under treatment sequence x̄.

**IPW estimator:**
```
∑ᵢ SW(x̄ᵢ)Yᵢ / ∑ᵢ SW(x̄ᵢ)
```

where SW(x̄ᵢ) is stabilized weight.

## 20.10 Selection Bias and Missing Data

### Selection Bias
**Problem:** Sample is not representative of target population

**Selection DAG:** Include selection variable S
**Bias:** Conditioning on S creates collider bias

### Missing Data Mechanisms
**MCAR:** Missing completely at random
**MAR:** Missing at random given observed variables
**MNAR:** Missing not at random

### Multiple Imputation
**Procedure:**
1. Create multiple imputations for missing values
2. Analyze each completed dataset
3. Combine results using Rubin's rules

## 20.11 Graphical Models for Networks

### Social Networks
**Peer effects:** Individual outcomes depend on network neighbors
**Identification challenges:**
- Homophily vs influence
- Correlated unobservables
- Network formation

### Interference
**SUTVA violation:** Treatment of one unit affects others
**Spillover effects:** Direct and indirect effects through network

**Design solutions:**
- Cluster randomization
- Network experiments
- Two-stage randomization

## 20.12 Machine Learning and Causal Inference

### Double Machine Learning
**Problem:** High-dimensional confounders
**Solution:** Use ML for nuisance functions, maintain root-n rates for causal parameters

**DML estimator:**
```
θ̂ = argmin_θ 𝔼ₙ[ψ(W; θ, η̂)]
```

where η̂ are ML-estimated nuisance parameters.

### Causal Forests
**Extension:** Random forests for heterogeneous treatment effects
**Honest trees:** Separate samples for splitting and estimation
**Local centering:** Reduce bias from confounding

### Deep Learning
**Representation learning:** Learn balanced representations
**TARNet:** Treatment-agnostic representation network
**GANITE:** GAN-based approach for causal inference

## 20.13 Experimental Design

### Randomized Controlled Trials
**Gold standard:** Random assignment eliminates confounding
**Limitations:**
- Ethical constraints
- External validity
- Compliance issues

### Natural Experiments
**Quasi-random assignment:** Leverage random or as-if random variation
**Examples:**
- Regression discontinuity
- Difference-in-differences  
- Instrumental variables

### A/B Testing
**Online experiments:** Randomize users to treatments
**Challenges:**
- Network effects
- Non-compliance
- Multiple testing

## 20.14 Sensitivity Analysis

### Unobserved Confounding
**Question:** How would conclusions change with unmeasured confounder?

**Rosenbaum bounds:** Assess sensitivity to hidden bias
**E-value:** Minimum strength of confounding to explain away effect

### Model Uncertainty
**Graph uncertainty:** Multiple plausible causal structures
**Specification uncertainty:** Functional form assumptions

**Robust methods:**
- Model averaging
- Worst-case bounds
- Intersection bounds

## 20.15 Software and Implementation

### R Packages
**dagitty:** Create and analyze DAGs
**pcalg:** Causal discovery algorithms
**mediation:** Mediation analysis
**ggdag:** Visualization of DAGs

### Python Libraries
**causalgraphicalmodels:** DAG manipulation
**causal-learn:** Causal discovery
**econml:** Econometric machine learning
**DoWhy:** Unified causal inference framework

### Workflow
1. **Specify DAG** based on domain knowledge
2. **Identify estimand** using causal criteria
3. **Estimate** using appropriate method
4. **Validate** with sensitivity analysis

## 20.16 Applications

### Epidemiology
**Exposure-disease relationships:** Account for confounding
**Time-varying treatments:** Dynamic treatment regimes
**Mediation:** Identify causal pathways

### Economics
**Policy evaluation:** Estimate treatment effects
**Natural experiments:** Leverage institutional variation
**Instrumental variables:** Address endogeneity

### Marketing
**Attribution:** Which ads drive conversions?
**Incrementality:** Lift due to advertising
**Personalization:** Heterogeneous treatment effects

### Technology
**A/B testing:** Product feature evaluation
**Recommendation systems:** Causal effects of recommendations
**Platform design:** Network effects and externalities

## Key Insights

1. **Graphs encode assumptions:** DAGs make causal assumptions explicit and testable.

2. **Identification conditions:** Specific criteria determine when causal effects are identifiable.

3. **No causation without manipulation:** Need potential outcomes framework for causal interpretation.

4. **Confounding is pervasive:** Most observational studies suffer from confounding bias.

5. **Design trumps analysis:** Good design reduces need for complex statistical methods.

## Common Pitfalls

1. **Correlation-causation confusion:** Statistical association ≠ causal relationship
2. **Post-treatment bias:** Controlling for variables affected by treatment
3. **Collider bias:** Conditioning on common effects creates spurious association
4. **Selection bias:** Non-representative samples lead to biased estimates
5. **Model dependence:** Results sensitive to untestable assumptions

## Best Practices

### DAG Construction
1. **Use domain knowledge:** Don't rely solely on data
2. **Include relevant confounders:** Consider all common causes
3. **Be explicit about assumptions:** State what you believe and why
4. **Consider temporal ordering:** Causes must precede effects
5. **Account for selection mechanisms:** Include selection variables

### Analysis Strategy
1. **Pre-specify analysis plan:** Avoid data-driven model selection
2. **Check identification conditions:** Verify backdoor/frontdoor criteria
3. **Perform sensitivity analysis:** Test robustness to assumptions
4. **Use multiple approaches:** Compare different identification strategies
5. **Report limitations clearly:** Acknowledge what cannot be identified

### Validation
1. **Placebo tests:** Use outcomes that shouldn't be affected
2. **Falsification tests:** Test implications of causal model
3. **Replication:** Verify results in different contexts
4. **External validation:** Test on independent datasets
5. **Expert review:** Have domain experts evaluate DAG

## 20.17 Philosophical Issues

### Causation vs Association
**Fundamental question:** What does it mean for X to cause Y?

**Manipulationist view:** X causes Y if manipulating X changes Y
**Counterfactual view:** X causes Y if Y would be different without X
**Mechanistic view:** X causes Y through identifiable physical process

### Levels of Causation
**Type causation:** Smoking causes cancer (general relationship)
**Token causation:** John's smoking caused his cancer (specific instance)
**Actual causation:** What actually happened
**Potential causation:** What could happen

### Scientific vs Legal Standards
**Scientific:** Establish general causal relationships
**Legal:** Determine specific causation in individual cases
**Policy:** Balance costs and benefits under uncertainty

## 20.18 Advanced Topics

### Nonlinear Causal Models
**Additive noise models:** Y = f(X) + ε with independent noise
**Post-nonlinear models:** Y = g(f(X) + ε)
**Identification:** Use non-Gaussianity or nonlinearity

### Cyclic Causal Models
**Feedback loops:** X → Y → X
**Equilibrium analysis:** Find stable solutions
**Dynamic systems:** Model evolution over time

### Latent Variable Models
**Hidden confounders:** Unmeasured common causes
**Measurement error:** Observed variables are proxies
**Factor models:** Multiple indicators of latent constructs

### Causal Discovery with Continuous Variables
**Score-based methods:** Optimize continuous scores
**Constraint-based methods:** Use partial correlations
**Hybrid approaches:** Combine both strategies

## 20.19 Modern Developments

### Deep Learning for Causal Discovery
**Neural causal discovery:** Use neural networks to learn causal structure
**NOTEARS:** Continuous optimization for acyclic graphs
**GraN-DAG:** Gradient-based neural DAG learning

### Causal Representation Learning
**Goal:** Learn causal variables from high-dimensional data
**Disentangled representations:** Separate causal factors
**Interventional data:** Use experimental data to identify representations

### Causal Reinforcement Learning
**Off-policy evaluation:** Estimate policy value from logged data
**Confounded bandits:** Multi-armed bandits with confounding
**Causal discovery in RL:** Learn environment structure

### Federated Causal Inference
**Distributed data:** Causal inference across multiple sites
**Privacy preservation:** Avoid sharing individual-level data
**Horizontal vs vertical federation:** Different data partitioning schemes

## 20.20 Connections to Other Fields

### Computer Science
**Artificial intelligence:** Causal reasoning in AI systems
**Database theory:** Query answering under uncertainty
**Programming languages:** Causal types and verification

### Philosophy
**Philosophy of science:** Nature of scientific explanation
**Epistemology:** How we gain causal knowledge
**Ethics:** Responsibility and moral causation

### Economics
**Econometrics:** Identification strategies
**Industrial organization:** Market structure and competition
**Public economics:** Policy evaluation and welfare analysis

### Biology
**Systems biology:** Regulatory networks
**Genetics:** Gene-environment interactions
**Evolution:** Causal mechanisms in natural selection

## 20.21 Future Directions

### Integration with Machine Learning
**Causal-aware ML:** Incorporate causal structure in learning
**Robust prediction:** Models that work under distribution shift
**Fairness:** Causal definitions of algorithmic fairness

### High-Dimensional Causal Inference
**Genomics:** Causal inference with millions of variables
**Network data:** Large-scale social and biological networks
**Time series:** Causal discovery in temporal data

### Experimental Design
**Adaptive experiments:** Use interim results to modify design
**Factorial designs:** Study multiple interventions simultaneously
**Bayesian optimal design:** Maximize expected information gain

### Causal Machine Learning
**Meta-learning:** Transfer causal knowledge across domains
**Few-shot causal learning:** Learn from limited data
**Continual learning:** Update causal models over time

## Practical Workflow

### Step 1: Problem Formulation
1. **Define research question:** What causal relationship to study?
2. **Specify estimand:** What quantity to estimate?
3. **Identify population:** Who does the inference apply to?
4. **Consider feasibility:** What data and methods are available?

### Step 2: DAG Construction
1. **Literature review:** What is known about the domain?
2. **Expert consultation:** Talk to domain experts
3. **Draw initial DAG:** Include all relevant variables
4. **Refine iteratively:** Update based on feedback and data

### Step 3: Identification
1. **Apply causal criteria:** Check backdoor/frontdoor conditions
2. **Consider alternatives:** Multiple identification strategies
3. **Assess assumptions:** What must be true for identification?
4. **Plan sensitivity analysis:** How to test key assumptions?

### Step 4: Estimation
1. **Choose estimator:** Based on identification strategy
2. **Handle confounders:** Adjustment, weighting, or matching
3. **Account for selection:** Include selection mechanisms
4. **Estimate uncertainties:** Confidence intervals and p-values

### Step 5: Validation
1. **Sensitivity analysis:** Vary key assumptions
2. **Placebo tests:** Check for spurious effects
3. **External validation:** Test in different contexts
4. **Robustness checks:** Try alternative methods

### Step 6: Interpretation
1. **Causal vs statistical:** Distinguish types of conclusions
2. **Effect sizes:** Practical significance
3. **Mechanisms:** How does causation work?
4. **Generalizability:** External validity concerns

## Key Mathematical Results

### d-separation Theorem
**Theorem:** In a DAG G, sets X and Y are d-separated by Z if and only if every path from X to Y is blocked by Z.

**Corollary:** If P is faithful to G, then X ⊥ Y | Z in P if and only if X and Y are d-separated by Z in G.

### Backdoor Adjustment Formula
**Theorem:** If Z satisfies the backdoor criterion for the effect of X on Y, then:
```
P(Y | do(X = x)) = ∑_z P(Y | X = x, Z = z)P(Z = z)
```

### Causal Markov Condition
**Theorem:** If the data-generating process follows the structural equations associated with DAG G, then the distribution satisfies the Markov condition relative to G.

### IC Algorithm Correctness
**Theorem:** Under faithfulness, the IC algorithm correctly identifies the Markov equivalence class of the true DAG.

## Connections to Other Chapters

### To Chapter 2 (Probability)
- Conditional independence and d-separation
- Factorization of joint distributions
- Bayes' theorem in causal inference

### To Chapter 7 (Statistical Inference)
- Identification vs estimation
- Confounding as source of bias
- Sensitivity analysis for robustness

### To Chapter 11 (Hypothesis Testing)
- Testing causal hypotheses
- Multiple testing in causal discovery
- Placebo tests and falsification

### To Chapter 12 (Bayesian Inference)
- Bayesian networks and causal models
- Prior specification for causal parameters
- Bayesian causal discovery

### To Chapter 13 (Decision Theory)
- Causal decision theory
- Value of information in causal systems
- Optimal interventions

### To Chapter 14 (Regression)
- Causal interpretation of regression
- Confounding and omitted variable bias
- Instrumental variables regression

This chapter provides a comprehensive introduction to causal inference using directed graphs, bridging the gap between statistical methods and causal reasoning in scientific applications.