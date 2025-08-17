# Blunk PhD Thesis: Complete Citation Guide by Argument

## Overview
This guide maps specific claims and arguments throughout Blunk's thesis to their supporting citations, organized by chapter and argument type. The thesis focuses on developing and validating statistical methods for detecting and analyzing parent-of-origin effects (genomic imprinting) in livestock.

---

## General Introduction: Foundational Arguments

### **Argument 1: Genomic Imprinting Definition and Mechanism**
- **Claim**: "Genomic imprinting is a phenomenon where the expression of genes is limited to one of the two inherited gametes"
- **Citations**: 
  - **O'Doherty et al. (2015)** - *Comprehensive review of imprinting mechanisms*
  - **Gould and Pfeifer (1998)** - *Developmental regulation of imprinting*
  - **Yu et al. (1998)** - *Tissue-specific imprinting patterns*
- **Use these for**: Establishing the biological foundation of genomic imprinting

### **Argument 2: Parent-of-Origin Effects Definition**
- **Claim**: "POEs are a comprehensive term for effects appearing as phenotypic differences between heterozygotes, depending on their parental origin"
- **Citation**: **Lawson et al. (2013)** - *Conceptual framework for parent-of-origin effects*
- **Use this for**: Distinguishing POEs from standard genetic effects

### **Argument 3: Imprinting Impact on Animal Breeding**
- **Claim**: "The neglect of genomic imprinting in animal breeding programs could bias breeding values and estimated genetic parameters"
- **Citation**: **Tier and Meyer (2012)** - *Quantitative POE analysis in beef cattle*
- **Use this for**: Justifying the practical importance of imprinting research

### **Argument 4: Epigenetic Mechanisms**
- **Claim**: "Imprinting relates to epigenetics through DNA methylation, histone modifications, and RNA-mediated effects"
- **Citations**:
  - **Li et al. (1993)** - *DNA methylation patterns*
  - **Weaver and Bartolomei (2014)** - *Chromatin regulation of imprinting*
  - **O'Doherty et al. (2015)** - *RNA-mediated imprinting effects*
- **Use these for**: Molecular mechanisms underlying imprinting

### **Argument 5: Chromosomal Organization**
- **Claim**: "Approximately 80% of all imprinted genes are physically organized in megabase-sized chromosomal clusters"
- **Citations**:
  - **Reik and Walter (2001)** - *Genomic imprinting organization*
  - **Wan and Bartolomei (2008)** - *Imprinting cluster regulation*
  - **Barlow (2011); Barlow and Bartolomei (2014)** - *Imprinting control elements*
- **Use these for**: Structural organization of imprinted loci

---

## Chapter 1: Theoretical Framework for Fluctuating Imprinting

### **Argument 6: Variance Component Theory**
- **Claim**: "Total imprinting variance has contributions from all imprinted loci with different expression patterns"
- **Citation**: **Neugebauer et al. (2010a;b)** - *Statistical framework for imprinting variance*
- **Use this for**: Theoretical foundation of imprinting variance models

### **Argument 7: Developmental and Tissue-Specific Imprinting**
- **Claim**: "Imprinting may be unstable over time (developmentally regulated) or between tissues (tissue specific)"
- **Citations**:
  - **Gould and Pfeifer (1998)** - *Developmental imprinting regulation*
  - **Yu et al. (1998)** - *Tissue-specific imprinting variance*
- **Use these for**: Supporting dynamic imprinting patterns

### **Argument 8: Gametic Relationship Matrix Applications**
- **Claim**: "Gametic relationship matrices enable proper modeling of parent-of-origin genetic effects"
- **Citation**: **Schaeffer et al. (1989)** - *Inverse gametic relationship matrix methodology*
- **Use this for**: Statistical implementation of imprinting models

---

## Chapter 2: Model Development for Brown Swiss Cattle

### **Argument 9: Direct Estimation of Imprinting Effects**
- **Claim**: "A new equivalent model facilitates direct estimation of imprinting effects instead of taking differences"
- **Citation**: **Henderson (1985)** - *Equivalent linear models theory*
- **Use this for**: Mathematical foundation for model equivalence

### **Argument 10: Prediction Error Variance Challenges**
- **Claim**: "Computation of imprinting effect PEVs requires laborious procedures in existing models"
- **Citation**: **Neugebauer et al. (2010a;b)** - *Original imprinting model limitations*
- **Use this for**: Justifying need for methodological improvements

### **Argument 11: Livestock Species Evidence**
- **Claim**: "Imprinting effects have been demonstrated in various agricultural species"
- **Citations**:
  - **Neugebauer et al. (2010a)** - *POE in pig performance traits (19 traits, 5-19% of genetic variance)*
  - **Neugebauer et al. (2010b)** - *POE in cattle beef traits (10 traits, 8-25% of genetic variance)*
  - **Tier and Meyer (2012)** - *28% average relative imprinting variance in cattle*
- **Use these for**: Evidence of imprinting relevance across species

### **Argument 12: Brown Swiss Slaughter Data Analysis**
- **Claim**: "Brown Swiss fattening bulls provide suitable data for imprinting analysis"
- **Statistical Framework**: 247,883 bulls (1994-2013), 428,710 animals in pedigree
- **Traits**: Net BW gain, carcass conformation, fatness, killing out percentage
- **Use this for**: Large-scale practical application example

### **Argument 13: Likelihood Ratio Testing**
- **Claim**: "REML likelihood ratio tests enable detection of significant imprinting variance"
- **Citations**:
  - **Self and Liang (1987)** - *Asymptotic properties of likelihood ratio tests*
  - **Neugebauer et al. (2010a;b)** - *Application to imprinting detection*
- **Use these for**: Statistical testing methodology for imprinting

---

## Chapter 3: Parsimonious Model for Large Datasets

### **Argument 14: Computational Efficiency Challenges**
- **Claim**: "Large volumes of commercial data require parsimonious models to avoid excessive computational burden"
- **Solution**: Maternal grandsire (MGS) model replacing dam effects
- **Citation**: **Henderson (1985)** - *Linear model equivalence principles*
- **Use this for**: Computational optimization in large-scale analyses

### **Argument 15: Dual-Purpose Simmental Application**
- **Claim**: "Dual-purpose Simmental provides extensive data for beef trait imprinting analysis"
- **Dataset**: 1,366,160 fattening bulls, 2,637,761 animals in pedigree
- **Traits**: Killing out percentage, net BW gain, carcass muscularity, fat score
- **Statistical Software**: **ASReml** (Gilmour et al., 2009)
- **Use this for**: Very large-scale implementation example

### **Argument 16: Model Equivalence Validation**
- **Claim**: "Equivalent models yield identical results after appropriate linear transformation"
- **Citations**:
  - **Henderson (1985)** - *Equivalence conditions*
  - **Blunk and Reinsch (2014)** - *Simulation verification*
- **Use these for**: Mathematical verification of model equivalence

### **Argument 17: Trait-Specific Imprinting Patterns**
- **Claim**: "Different beef traits show varying susceptibility to imprinting effects"
- **Evidence Pattern**: Carcass quality traits more consistent across breeds than performance traits
- **Literature Support**: **Engellandt and Tier (2002)** - *Breed-specific imprinting differences*
- **Use this for**: Trait-dependent imprinting interpretation

---

## Chapter 4: GWAS with Estimated Parent-of-Origin Effects

### **Argument 18: Traditional GWAS Limitations**
- **Claim**: "Standard GWAS requires phased genotypes and phenotyped animals, often unavailable in livestock"
- **Challenge**: Slaughtered offspring lack genotypes, phase uncertainty common
- **Innovation**: Use estimated POEs (ePOEs) as pseudo-phenotypes
- **Use this for**: Justifying novel GWAS approach

### **Argument 19: ePOE as Pseudo-Phenotypes**
- **Claim**: "ePOEs summarize all information on imprinted loci impact in progeny"
- **Mathematical Framework**: Regression of ePOEs on unphased parental genotypes
- **Citation**: **Garrick et al. (2009)** - *Deregression methodology adapted for ePOEs*
- **Use this for**: Novel methodological development

### **Argument 20: Deregression and Weighting Strategy**
- **Claim**: "De-regressed and weighted ePOEs avoid bias from parent-average effects"
- **Citations**:
  - **Garrick et al. (2009)** - *Deregression and weighting for genomic analysis*
  - **Ekine et al. (2014)** - *Problems with using breeding values directly in GWAS*
- **Use these for**: Technical implementation of ePOE correction

### **Argument 21: Brown Swiss GWAS Results**
- **Claim**: "ePOE-based GWAS successfully identifies candidate imprinted loci"
- **Key Findings**:
  - **Chromosome 11**: 5% genome-wide significant association (ARS-BFGL-NGS-101636)
  - **Gene Candidate**: REEP1 (Receptor Accessory Protein 1) for net BW gain
  - **Chromosome 24**: Additional associations for growth traits
  - **Chromosome 5**: Multiple associations with fatness traits
- **Use this for**: Successful application demonstration

### **Argument 22: Simulation Validation**
- **Claim**: "Simulation studies validate ePOE-based mapping approach"
- **Experimental Design**: 
  - 3-generation pedigree simulation
  - Known QTL effects for validation
  - Comparison of different ePOE treatments
- **Use this for**: Method validation and power assessment

---

## Advanced Technical Citations by Method

### **Statistical Model Development**
- **Linear Mixed Models**: **McCullagh and Nelder (1989)** - *Generalized linear models*
- **ASReml Implementation**: **Gilmour et al. (2009)** - *ASReml user guide release 3.0*
- **Variance Component Estimation**: **Mrode (2014)** - *Linear models for breeding values*

### **Genomic Data Analysis**
- **Linkage Disequilibrium**: **Browning and Browning (2009)** - *Genotype imputation and haplotype phasing*
- **Multiple Testing Correction**: **Benjamini and Hochberg (1995)** - *False discovery rate control*
- **QTL Mapping**: **Imumorin et al. (2011)** - *Parent-of-origin QTL effects in cattle*

### **Breeding Program Applications**
- **Sire Evaluation**: **VanRaden (1987)** - *Maternal grandsire evaluation methods*
- **Genetic Evaluation**: **Quaas and Pollak (1980; 1981)** - *Mixed model methodology*
- **Economic Impact**: **Fernando and Grossman (1990)** - *Genetic evaluation frameworks*

---

## Biological Validation Citations

### **IGF2 and Growth Traits**
- **Van Laere et al. (2003)** - *IGF2 regulatory mutation causing major QTL effect on muscle growth*
- **Jeon et al. (1999)** - *Paternally expressed QTL at IGF2 locus affecting muscle mass*
- **Dindot et al. (2004)** - *Conservation of genomic imprinting at IGF2 locus in bovine*

### **Cross-Species Imprinting Evidence**
- **Tuiskula-Haavisto et al. (2004)** - *Parent-of-origin QTL in chicken*
- **De Vries et al. (1994)** - *Gametic imprinting effects on pig growth*
- **Harlizius et al. (2000)** - *X-chromosome QTL for backfat and intramuscular fat*

### **Maternal Effects vs. Imprinting**
- **Hager et al. (2008)** - *Maternal effects as cause of parent-of-origin effects*
- **Wolf et al. (2008)** - *Genomic imprinting effects on complex traits*

---

## Software and Computational Citations

### **Primary Statistical Software**
- **ASReml**: **Gilmour et al. (2009)** - *Mixed model software for variance component estimation*
- **R Statistical Environment**: **R Core Team (2015)** - *Statistical computing platform*
- **Pedigree Analysis**: **Coster (2013)** - *Pedigree functions R package*

### **Specialized Genomic Tools**
- **Genome Assembly**: **Bos Taurus UMD3.1.1/bosTau8** - *UCSC Genome Browser reference*
- **QTL Database**: **Hu et al. (2016)** - *Animal QTLdb development and status*

---

## Citation Strategy Recommendations

### **For Different Argument Types:**

1. **Biological Foundation Claims**: Use O'Doherty et al. (2015) + Reik and Walter (2001) + mechanistic papers
2. **Statistical Method Claims**: Use Henderson (1985) for equivalence + Neugebauer et al. (2010a;b) for imprinting models
3. **Large-Scale Application Claims**: Use Gilmour et al. (2009) + specific dataset papers
4. **GWAS Innovation Claims**: Use Garrick et al. (2009) + Ekine et al. (2014) + simulation validation
5. **Cross-Species Evidence**: Use species-specific papers (Neugebauer for pigs/cattle, Van Laere for IGF2)

### **Combinations for Strong Arguments:**
- **Imprinting biological importance**: O'Doherty et al. (2015) + Tier and Meyer (2012) + species examples
- **Statistical model development**: Henderson (1985) + Neugebauer et al. (2010a;b) + ASReml implementation
- **GWAS methodology**: Garrick et al. (2009) + simulation validation + Brown Swiss results
- **Practical breeding applications**: Large dataset papers + economic impact citations

### **Novel Methodological Contributions:**
- **Equivalent Model**: First direct estimation of POEs with easy PEV calculation
- **Parsimonious MGS Model**: Computational efficiency for large commercial datasets
- **ePOE-based GWAS**: Novel approach for unphased genotypes in ungenotyped progeny
- **Multi-species validation**: Consistent methodology across cattle breeds and pig populations

---

## Critical Assessment Notes

**Strengths of Blunk's Citation Strategy:**
- Strong theoretical foundation with Henderson (1985) equivalence theory
- Comprehensive biological background from molecular to quantitative levels
- Appropriate use of simulation validation for novel methods
- Large-scale real data applications demonstrating practical utility

**Methodological Innovations:**
- Direct POE estimation avoiding computational challenges
- Parsimonious models enabling very large dataset analysis
- Novel GWAS approach using summary statistics (ePOEs)
- Cross-validation between multiple cattle breeds and datasets

**Potential Areas for Extension:**
- Integration with genomic prediction methods
- Single-step approaches combining pedigree and genomic information
- Multi-trait imprinting models
- Economic optimization of breeding programs considering imprinting

**Impact on Field:**
- Provided computationally feasible methods for large-scale imprinting analysis
- Enabled GWAS for imprinted loci without requiring expensive progeny genotyping
- Demonstrated significant imprinting effects across multiple economically important traits
- Established statistical framework for routine incorporation of imprinting in breeding programs