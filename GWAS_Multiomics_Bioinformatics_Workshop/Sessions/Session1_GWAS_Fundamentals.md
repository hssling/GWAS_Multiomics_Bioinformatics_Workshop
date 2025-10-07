# Session 1: Genome-Wide Association Studies Fundamentals

## Learning Objectives
By the end of this session, participants will be able to:
- Understand the basic principles of GWAS
- Describe the genetic basis of complex diseases
- Explain the concept of single nucleotide polymorphisms (SNPs)
- Understand GWAS study design and methodology

## Pre-Reading Materials
- Hirschhorn JN et al. (2005). Genomewide association studies and assessment of the risk of disease. *NEJM*.
- Visscher PM et al. (2017). 10 Years of GWAS Discovery: Biology, Function, and Translation. *AJHG*.

## Presentation Outline

### 1. Introduction to Genetic Variation (20 min)
#### 1.1 The Human Genome
- **Genome structure**: 23 chromosome pairs, ~3 billion base pairs
- **Genetic polymorphism**: Variations in DNA sequence between individuals
- **Types of variation**:
  - Single nucleotide polymorphisms (SNPs)
  - Insertions/deletions (indels)
  - Copy number variations (CNVs)
  - Structural variants

#### 1.2 SNP Characteristics
- **Definition**: Single base pair differences between individuals
- **Prevalence**: ~10-30 million SNPs per genome
- **Minor allele frequency (MAF)**: Frequency of the less common allele
- **Linkage disequilibrium (LD)**: Non-random association between SNPs

### 2. GWAS Principles and Design (30 min)
#### 2.1 Historical Context
- **Before GWAS**: Candidate gene studies limitations
- **The GWAS revolution**: Genome-wide interrogation
- **Technological enablers**:
  - SNP genotyping arrays
  - DNA sequencing technology
  - Statistical methodology

#### 2.2 Study Design Elements
- **Case-control design**:
  - Cases: Individuals with disease/phenotype
  - Controls: Individuals without disease/phenotype
  - Matching criteria
- **Cohort studies**: Prospective/retrospective designs
- **Sample size considerations**

#### 2.3 Quality Control Measures
- **Genotyping quality**: Call rate, Hardy-Weinberg equilibrium
- **Sample quality**: Contamination, relatedness, ancestry
- **Population stratification**: Confounding by ancestry
- **Batch effects**: Technical artifacts

### 3. Statistical Analysis in GWAS (25 min)
#### 3.1 Basic Statistical Tests
- **Chi-square test**: Allele frequency differences
- **Logistic regression**: Odds ratio estimation
- **Multiple testing correction**:
  - Bonferroni correction
  - False discovery rate (FDR)
  - q-value approach

#### 3.2 Significance Thresholds
- **Nominal p-value**: Standard 0.05 threshold
- **Genome-wide significance**: 5 × 10⁻⁸ (Bonferroni corrected)
- **Suggestive associations**: 1 × 10⁻⁵ to 5 × 10⁻⁸
- **Power considerations**: Effect size, MAF, sample size

#### 3.3 Effect Size Measures
- **Odds ratio (OR)**: Disease risk associated with variant
- **Beta coefficient**: Quantitative trait effect size
- **Confidence intervals**: Statistical precision

### 4. GWAS Results Interpretation (20 min)
#### 4.1 Manhattan Plots
- **X-axis**: Genomic position or SNP order
- **Y-axis**: -log₁₀(p-value)
- **Color coding**: Chromosomes or significance levels
- **Peaks**: Associated genomic regions

#### 4.2 Quantile-Quantile (QQ) Plots
- **Expected vs observed p-values**
- **Diagonal line**: Null hypothesis
- **Inflation factor (λ)**: Systematic bias indication
- **Deviations**: True associations vs artifacts

#### 4.3 Regional Association Plots
- **Fine-mapping**: Identifying causal variants
- **Recombination hotspots**: LD block breakdown
- **Functional annotation**: Gene and regulatory element mapping

## Interactive Exercises

### Exercise 1.1: Understanding SNP Nomenclature
**Objective**: Practice reading SNP IDs and understanding allele designations

**Data Provided**:
```
rs7412 on chromosome 19
Alleles: C/T
Genotype frequencies in population:
CC: 49%  CT: 42%  TT: 9%
```

**Questions**:
1. What is the reference allele?
2. What is the MAF in this population?
3. If someone has genotype CT, what is their predicted APOE genotype?

### Exercise 1.2: GWAS Power Calculations
**Objective**: Understand how sample size affects GWAS power

**Scenario**:
- Risk allele frequency: 0.3
- Genotypic relative risk: 1.5
- Required power: 80%
- Significance threshold: 5 × 10⁻⁸

**Questions**:
1. Calculate required sample size
2. How does power change with different effect sizes?
3. What happens if MAF is lower (e.g., 0.1)?

## Key Takeaways
- GWAS systematically scan the genome for disease associations
- SNPs are the primary markers used in most GWAS
- Multiple testing correction requires stringent significance thresholds
- Study design, quality control, and statistical analysis are critical for valid results
- Manhattan and QQ plots are essential visualization tools
- GWAS findings need independent replication and functional follow-up

## Further Reading
- McCarthy MI et al. (2008). Genome-wide association studies for complex traits: Consensus, uncertainty and challenges. *Nature Genetics*.
- Gibbs JR & Singleton A (2006). Application of genome-wide single nucleotide polymorphism typing: Simple association and beyond. *PLoS Genetics*.

## Session Assessment
**Pre/Post Test Questions**:
1. What is the primary statistical test used in case-control GWAS?
2. Why is genome-wide significance set at 5 × 10⁻⁸?
3. What does a Manhattan plot show?
4. What is linkage disequilibrium?

## Next Session Preview
**Session 2**: GWAS Data Analysis and Visualization - Advanced statistical methods, fine-mapping, and result interpretation.
