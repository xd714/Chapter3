
# Chapter 3: Real Data Implementation and Validation

## Technical Pipeline Phase 2: Apply Validated Methods to Real Breeding Population

---

## CHAPTER 3: PRACTICAL IMPLEMENTATION (Real Data Application)

### Principle: "Informed Application" - Apply validated methods with realistic expectations

#### Scientific Principle

**"Informed Application"** - Apply genomic selection using methods validated in Chapter 2, with realistic expectations of validation method limitations.

#### Implementation Strategy:

- **Method Selection:** Based on Chapter 2 findings (dEBV + multiple validation approaches)
- **Expectation Setting:** Forward prediction will underestimate (as shown in simulation)
- **Validation Approach:** Use theoretical reliabilities as "gold standard"
- **Sample Strategy:** Include sows to achieve sufficient sample size

---

## Real Data Collection and Processing

### **Step 1: Sample Assembly Strategy**

```python
# Strategic Sample Composition
sample_design = {
    "total_animals": 2031,
    "boars": 337,     # All available (birth years 1995-2011)
    "sows": 1676,     # Strategic inclusion (birth years 2005-2011)
    "rationale": "Insufficient boars alone → include sows for 'critical mass'",
    "challenge": "Mixed population with heterogeneous reliabilities"
}

# Sample Structure Challenges Identified
challenges = {
    "temporal_stratification": "Old boars + young sows",
    "geographic_stratification": "Different AI stations, farmer preferences",
    "kinship_structure": "Suboptimal for genomic prediction",
    "reliability_heterogeneity": "Wide range of phenotype reliabilities"
}

# Strategic Justification
inclusion_rationale = """
Boar-Only Sample Limitations:
- n = 337 boars insufficient for reliable genomic prediction
- Limited genetic diversity within boar lines
- Economic constraints prevent additional boar genotyping

Sow Inclusion Benefits:
- Increased sample size (n = 2,013 total)
- Greater genetic diversity across farms
- Maternal trait information directly from dams
- Cost-effective way to achieve 'critical mass'

Trade-offs Accepted:
- Increased population stratification
- Heterogeneous reliability structure
- More complex validation interpretation
"""
```

### **Step 2: Genotyping and Quality Control**

```python
# High-Density SNP Array Processing
genotyping_protocol = {
    "platform": "Illumina PorcineSNP60 BeadChip (v1 and v2)",
    "initial_snps": 60000,
    "common_markers": 58931,  # Only markers present on both chip versions
    "annotation": "Sus scrofa genome build 10.2"
}

# Quality Control Pipeline Implementation
quality_control = """
Step 1: Hardy-Weinberg Equilibrium Filter
Remove SNPs with P < 10⁻⁵ for HWE test
Test statistic: χ² = (O_het - E_het)² / E_het
where E_het = 2pq for expected heterozygotes

Step 2: Call Rate Filter  
Remove SNPs with call rate < 0.95
Remove animals with call rate < 0.95
Call_rate = (total_genotypes - missing_genotypes) / total_genotypes

Step 3: Minor Allele Frequency Filter
Remove SNPs with MAF < 0.01
MAF = min(p, 1-p) where p = allele frequency

Final Result: 49,856-51,167 SNPs retained
"""

# Pedigree Verification (Critical for genomic accuracy)
pedigree_qc = {
    "method": "IBD coefficients (Wang, 2007)",
    "parent_threshold": 0.4,
    "grandparent_threshold": 0.1,
    "conflicts_resolution": "Set to missing",
    "final_sample": "2,013 animals (337 boars + 1,676 sows)"
}

# Mathematical IBD Verification
ibd_calculation = """
Identity-by-Descent Coefficient Calculation:
IBD = Σ(matching_alleles) / (2 × total_markers)

Parent-Offspring Relationship:
Expected IBD ≈ 0.5, Threshold = 0.4
If IBD < 0.4 → Set parent to missing

Grandparent-Offspring Relationship:  
Expected IBD ≈ 0.25, Threshold = 0.1
If IBD < 0.1 → Set grandparent to missing

Quality Control Impact:
Initial sample: 2,031 animals
After pedigree verification: 2,013 animals
Removal rate: 0.9% (18 animals with pedigree conflicts)
"""
```

### **Step 3: Phenotype Processing Pipeline**

```python
# Comprehensive Trait Analysis
trait_categories = {
    "fertility": ["NBA: Number born alive (h²=0.24)", 
                  "PW: Proportion weaned (h²=0.18)"],
    "performance": ["ADR: Average daily gain (h²=0.31)", 
                    "FCR: Feed conversion ratio (h²=0.48)"],
    "carcass": ["LP: Lean meat content (h²=0.66)", 
                "EMA: Eye muscle area (h²=0.72)", 
                "CL: Carcass length (h²=0.72)"],
    "quality": ["PH1: PH 45 min (h²=0.19)", 
                "IMF: Intramuscular fat (h²=0.70)"]
}

# EBV Estimation and Deregression
phenotype_processing = {
    "ebv_method": "Single-trait animal model BLUP",
    "deregression": "Garrick et al. (2009) method (validated in Chapter 2)",
    "rationale": "Chapter 2 showed dEBV superior for calibration"
}

# Mathematical EBV Calculation
ebv_model = """
Animal Model BLUP:
y = Xβ + Za + e

where:
- y = vector of phenotypic observations
- X = design matrix for fixed effects (farm, year, etc.)
- β = vector of fixed effects
- Z = incidence matrix relating animals to observations
- a ~ N(0, Aσ²_a) = additive genetic effects
- e ~ N(0, Iσ²_e) = residual effects
- A = numerator relationship matrix from pedigree

Mixed Model Equations:
[X'X  X'Z] [β̂]   [X'y]
[Z'X  Z'Z+A⁻¹λ] [â] = [Z'y]

where λ = σ²_e / σ²_a

EBV Reliability:
r²_i = 1 - PEV_i / σ²_a
where PEV_i = prediction error variance from MME inversion
"""

# Deregression Implementation (Following Chapter 2 Validation)
deregression_implementation = """
Based on Chapter 2 findings, apply Garrick et al. (2009) method:

Step 1: Calculate reliability after removing PA
r²*_i = (r²_i - r²_PA) / (1 - r²_PA)

Step 2: Deregress EBV
dEBV_i = (EBV_i - PA_i) / r²*_i

Step 3: Calculate weights for heterogeneous variances
w_i = r²*_i / h²

where:
- r²_i = original EBV reliability
- r²_PA = parent average reliability
- PA_i = (EBV_sire + EBV_dam) / 2
- h² = trait heritability

Purpose: Remove parent information to avoid double-counting in genomic calibration
"""
```





