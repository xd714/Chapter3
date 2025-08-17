# Chapter 2: Methodological Validation Through Simulation

## Technical Pipeline Phase 1: Method Testing Under Controlled Conditions

---

## CHAPTER 2: METHODOLOGICAL VALIDATION (Simulation Study)

### Principle: "Validate the Validators" Before Real Data Application

#### Scientific Principle

**"Validate the Validators"** - Before trusting validation methods on real data, test them under known conditions where truth is accessible.

#### Why Simulation First:

- **Access to Truth:** Can compare estimated vs. true reliabilities
- **Controlled Parameters:** Test different genetic scenarios systematically
- **Risk-Free Testing:** Identify method limitations before expensive real data application
- **Method Selection:** Determine which validation approaches work best for pig scenarios

---

## Technical Implementation

### **Step 1: Simulation Framework Setup**

```python
# QMSim Software Implementation
# Designed to mimic pig breeding structure

SIMULATION_PARAMETERS = {
    "software": "QMSim (Sargolzaei & Schenkel, 2009)",
    "purpose": "Test validation methods under controlled conditions",
    "advantage": "True breeding values known for comparison"
}

# Nine Genetic Scenarios
scenarios = {
    "heritabilities": [0.06, 0.25, 0.6],  # Low, medium, high
    "qtl_numbers": [10, 50, 500],         # Simple to complex genetic architecture
    "rationale": "Cover range typical of pig breeding traits",
    "replications": 30  # Per scenario for statistical robustness
}
```

### **Step 2: Population Structure Design**

```python
# Historic Population (Establish LD and mutation-drift equilibrium)
historic_population = {
    "initial_size": 2000,
    "sex_ratio": 0.5,
    "bottleneck": "Gen 950-980: 2000→1500→2000",  # Create long-range LD
    "mutation_rate": 5e-9,
    "generations": 1000,
    "mutation_distribution": "Poisson with mean μ = 2×10⁻⁶ × L"
}

# Breeding Population (Mimic pig breeding)
breeding_population = {
    "founders": {"males": 200, "females": 400},
    "selection_generations": 6,
    "litter_size": 9,
    "population_per_gen": 6610,
    "selection_method": "BLUP EBV-based",
    "selection_intensity": "Cull lowest EBV animals each generation"
}
```

### **Step 3: Genome Architecture**

```python
# Simplified but representative genome
genome_structure = {
    "chromosome_length": "400cM",  # Single chromosome (computational efficiency)
    "markers": 5000,               # Bi-allelic, random distribution
    "initial_allele_freq": 0.5,
    "qtl_variance_proportion": 0.9,  # 90% of heritability from QTL
    "polygenic_proportion": 0.1      # 10% polygenic background
}

# Trait Model (Sex-limited, typical for pig fertility traits)
trait_model = """
y_i = μ + a_i + q_i + e_i

where:
- μ = overall mean (constant)
- a_i = additive polygenic effect for animal i
- q_i = additive QTL effect for animal i  
- e_i = residual effect for animal i
- TBV_i = a_i + q_i (TRUE breeding value - our reference)

Variance Components:
Var(a_i) = (1-0.9) × h² × σ²_p = 0.1 × h² × σ²_p
Var(q_i) = 0.9 × h² × σ²_p  
Var(e_i) = (1-h²) × σ²_p
σ²_p = 1 (phenotypic variance set to 1 for simplicity)
"""
```

### **Step 4: Validation Method Testing Protocol**

**Response Variables Tested:**

```python
response_variables = {
    "TBV": "True Breeding Value (simulation reference)",
    "EBV": "Estimated Breeding Value (conventional BLUP)",
    "dEBV": "Deregressed Breeding Value (Garrick et al. 2009 method)"
}

# Deregression Mathematical Implementation (Garrick et al. 2009)
def deregression_process():
    """
    Remove parent average to avoid double-counting in genomic calibration
    """
    # Core deregression formula
    formula = "dEBV_i = (EBV_i - PA_i) / r²*_i"
    
    # Deregression weights
    weights = "w_i = r²*_i / h²"
    
    # Reliability calculation after removing PA
    reliability = "r²*_i = (r²_i - r²_PA) / (1 - r²_PA)"
    
    where = {
        "EBV_i": "Conventional breeding value for animal i",
        "PA_i": "Parent average for animal i", 
        "r²_i": "Original EBV reliability",
        "r²_PA": "Parent average reliability",
        "r²*_i": "Reliability after removing PA contribution",
        "h²": "Trait heritability"
    }
    
    return formula, weights, reliability, where
```

**Theoretical Reliability Calculation:**

```python
# From mixed model equations (Mrode, 2005)
theoretical_reliability = """
r²_i = 1 - [diag(C₂₂)_i × σ²_e × (1 + F_i)] / σ²_a

where:
- diag(C₂₂)_i = diagonal element for animal i in inverted coefficient matrix
- σ²_e = residual variance
- σ²_a = additive genetic variance
- F_i = inbreeding coefficient of animal i

Implementation:
C₂₂ obtained from inversion of mixed model equations:
[X'X  X'Z] [β̂]   [X'y]
[Z'X  Z'Z+A⁻¹λ] [û] = [Z'y]

where λ = σ²_e / σ²_a
"""
```

### **Step 5: Genomic BLUP Implementation**

```python
# gBLUP Model for DGV Estimation
gblup_model = {
    "software": "R package rrBLUP (Endelman, 2011)",
    "relationship_matrix": "VanRaden Method 1",
    "calibration_response": "dEBV (based on Step 4 results)",
    "calibration_size": 1500,  # Animals with highest deregression weights
    "validation_size": 200     # Generation 4 animals
}

# Mathematical Implementation - Genomic Relationship Matrix
genomic_relationship = """
G = WW' / c

where:
- W = M - 2(p - 0.5)  [centered genotype matrix]
- M = genotype matrix with elements {-1, 0, 1} for {AA, Aa, aa}
- p = vector of allele frequencies
- c = 2 Σ p_k(1 - p_k)  [scaling factor]
- k = 1 to m markers
"""

# gBLUP Mixed Model Equations
gblup_equations = """
[X'X    X'Z  ] [β̂]   [X'y]
[Z'X  Z'Z+G⁻¹λ] [û]  = [Z'y]

where:
- X = design matrix for fixed effects
- Z = design matrix for random effects  
- G = genomic relationship matrix
- λ = σ²_e / σ²_a (variance ratio)
- β̂ = fixed effect solutions
- û = random effect solutions (DGVs)
"""

# Actual Prediction Formula (rrBLUP implementation)
prediction_formula = """
ĝ = G* σ²_a Z' V⁻¹ (y - Xβ̂)

where:
- ĝ = vector of genomic breeding values
- G* = 0.9G + 0.1A (blended relationship matrix)
- A = numerator relationship matrix
- V = variance-covariance matrix = Z(G*σ²_a)Z' + Wσ²_e
- W = diagonal weight matrix from deregression
"""
```

### **Step 6: Validation Method Comparison**

```python
validation_methods = {
    "forward_prediction": {
        "principle": "Time-separated validation (cattle standard)",
        "datasets": ["t0: restricted (calibration only)", 
                    "t1: full (all phenotypes)"],
        "regression_model": """
        y = b₀ + b₁ × DGV + e
        
        where:
        - y = response variable (EBV or dEBV at t₁)
        - DGV = genomic breeding values at t₀
        - R² = reliability estimate
        - b₁ = prediction bias estimate
        """,
        "weighted_regression": """
        For dEBV as response variable:
        Var(e_i) = σ²_e / w_i
        
        where w_i = deregression weights
        """
    },
    
    "correction_method": {
        "principle": "Account for proxy variable reliability",
        "reliability_correction": """
        R²_corrected = R²_validation / r̄²_response
        
        where:
        - R²_validation = coefficient of determination from regression
        - r̄²_response = average reliability of response variable
        """,
        "purpose": "Adjust for imperfect response variable"
    },
    
    "theoretical_method": {
        "principle": "Direct calculation from mixed model",
        "reliability_formula": """
        r²_DGV = 1 - PEV_i / σ²_a
        
        where:
        - PEV_i = prediction error variance for animal i
        - σ²_a = additive genetic variance
        
        PEV obtained from diagonal of:
        C⁻¹ = [X'X    X'Z  ]⁻¹
              [Z'X  Z'Z+G⁻¹λ]
        """
    }
}
```

---

## Experimental Design and Analysis

### **Sample Selection Strategy:**

```python
sample_selection = {
    "calibration_strategy": "Select 1500 animals with highest deregression weights",
    "rationale": "Garrick et al. weights represent mendelian sampling information content",
    "validation_strategy": "Random 200 animals from generation 4",
    "time_separation": "Calibration: gen 0-3, Validation: gen 4"
}
```

### **Statistical Analysis Framework:**

```python
analysis_framework = """
Primary Metrics:
1. R²_True = R² from regression: TBV ~ DGV
2. R²_Estimated = R² from regression: Response_variable ~ DGV  
3. R²_Corrected = R²_Estimated / Average_reliability_response
4. Bias = |R²_Estimated - R²_True|
5. Prediction_bias = slope of regression (theoretical expectation = 1)

Evaluation Criteria:
- Best method: min(|R²_Estimated - R²_True|)
- Correction effectiveness: |R²_Corrected - R²_True| < |R²_Estimated - R²_True|
- Consistency: low variance across replications

Statistical Testing:
- 30 replications per scenario
- ANOVA for method comparison
- Regression analysis for reliability relationships
"""
```

---

## Key Findings from Chapter 2

```python
simulation_results = {
    "main_finding": "Forward prediction underestimates true reliability with low-reliability response variables",
    "best_response_variable": "dEBV with reliability correction",
    "reliability_correction": "Improves agreement but can overcorrect",
    "pig_breeding_implication": "Standard cattle methods need adaptation"
}

# Quantitative Results from Simulation
quantitative_findings = """
Validation Reliability vs True Reliability Comparison:

For h² = 0.06, 500 QTL scenario:
- True reliability (TBV): 0.244 ± 0.105
- EBV validation: 0.061 ± 0.052 (severely underestimated)
- dEBV validation: 0.046 ± 0.041 (severely underestimated)
- dEBV corrected: 0.264 ± 0.230 (close to true, high variance)

For h² = 0.6, 500 QTL scenario:
- True reliability (TBV): 0.420 ± 0.089  
- EBV validation: 0.235 ± 0.067 (underestimated)
- dEBV validation: 0.254 ± 0.073 (slightly better)
- dEBV corrected: 0.434 ± 0.126 (closest to true)

Key Mathematical Relationship:
Reliability_corrected = Reliability_validation / Average_response_reliability

Optimal correction achieved when:
Average_response_reliability >> Reliability_validation
"""
```

### **Method Performance Summary:**

```python
method_performance = """
Response Variable Performance Ranking:
1. dEBV with correction (best agreement with true reliability)
2. EBV with correction (good for high heritability)
3. dEBV without correction (better for calibration)
4. EBV without correction (severe underestimation)

Prediction Bias Results:
- True bias (b_True): Close to 1 for most scenarios
- EBV bias estimate: Consistently underestimated
- dEBV bias estimate: More accurate, especially high heritability

Correction Method Limitations:
- Overcorrection when: Average_reliability ≈ Validation_reliability
- High variance when: Average_reliability is low
- Failure mode: Reliability > 1 (impossible values)
"""
```

### **Critical Insights for Chapter 3:**

```python
chapter3_guidance = {
    "validation_method_selection": "Use theoretical reliabilities as gold standard",
    "forward_prediction_expectation": "Will underestimate - expect conservative estimates",
    "correction_application": "Apply with caution - monitor for overcorrection",
    "response_variable_choice": "Use dEBV for calibration and validation"
}
```

---

## Chapter 2 Conclusions and Transition

### **Scientific Contributions:**

1. **First systematic validation** of genomic validation methods under pig-like conditions
2. **Quantified reliability correction** effectiveness and limitations
3. **Identified best practices** for challenging population scenarios
4. **Established methodological foundation** for real data application

### **Methodological Recommendations:**

```python
recommendations = {
    "for_practitioners": "Use multiple validation methods and compare results",
    "for_researchers": "Report validation method used and its limitations",
    "for_pig_breeding": "Expect lower validation reliabilities than cattle",
    "for_interpretation": "Consider correction but monitor for overcorrection"
}
```

### **Transition to Chapter 3:**

Having established which validation methods work under controlled conditions and their limitations, Chapter 3 will apply these validated approaches to real Bavarian Herdbook data with informed expectations about method performance.

**Next Phase:** Real data implementation using validated methods with appropriate interpretation framework established in Chapter 2.





