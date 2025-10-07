# Session 2: GWAS Data Analysis and Visualization

## Learning Objectives
By the end of this session, participants will be able to:
- Interpret Manhattan plots and identify genomic loci of interest
- Create and analyze quantile-quantile (QQ) plots for quality control
- Perform basic fine-mapping of GWAS associations
- Calculate and interpret polygenic risk scores (PRS)
- Understand GWAS meta-analysis approaches

## Pre-Reading Materials
- Purcell S et al. (2007). PLINK: A tool set for whole-genome association and population-based linkage analyses. *AJHG*.
- Bulik-Sullivan BK et al. (2015). LD Score regression distinguishes confounding from polygenicity in genome-wide association studies. *Nature Genetics*.

## Presentation Outline

### 1. GWAS Results Quality Assessment (20 min)
#### 1.1 Genome-Wide Inflation
- **λ (lambda) statistic**: Ratio of observed vs expected median χ²
- **Sources of inflation**: Population stratification, cryptic relatedness, unaccounted confounders
- **Correction methods**: Genomic control, principal components
- **QC thresholds**: λ < 1.05 for well-controlled studies

#### 1.2 Quantile-Quantile Plots
- **Construction**: Expected vs observed -log₁₀(p-values)
- **Expected distribution**: Uniform p-values under null hypothesis
- **Interpretation**: Deviation indicates systematic bias or true associations
- **Stratification analysis**: Subset-specific QQ plots

#### 1.3 Sample and SNP Quality Metrics
- **Call rate filtering**: Individual and SNP thresholds (typically >95%)
- **Heterozygosity outliers**: Detection of contaminated samples
- **Sex discrepancies**: X-chromosome heterozygosity checks
- **Population outliers**: Principal component analysis for ancestry

### 2. Manhattan Plots and Genomic Visualization (25 min)
#### 2.1 Manhattan Plot Construction
- **X-axis**: Genomic coordinates (chromosome, position)
- **Y-axis**: Statistical significance (-log₁₀(p-value))
- **Color coding**: Alternating chromosomes or significance levels
- **Significance thresholds**: Suggestive (p < 1×10⁻⁵) and genome-wide (p < 5×10⁻⁸)
- **Chromosomal boundaries**: Clear demarcation between chromosomes

#### 2.2 Identifying Association Signals
- **Peak height**: Strength of association signal
- **Peak width**: Extent of linkage disequilibrium
- **Multiple peaks**: Independent associations vs LD
- **Chromosomal anomalies**: Regions with unusual patterns

#### 2.3 Interactive Visualization Tools
- **Zoom functionality**: Detailed views of specific regions
- **Annotation overlay**: Gene names, regulatory elements
- **LD visualization**: Haplotype block structure
- **Functional tracks**: Conservation, chromatin state

### 3. Fine-Mapping and Functional Follow-up (25 min)
#### 3.1 Linkage Disequilibrium Analysis
- **LD measures**: D', r² correlation coefficients
- **Haplotype blocks**: Regions of high LD
- **Tag SNPs**: Representative variants in LD blocks
- **Population differences**: LD varies by ancestry

#### 3.2 Conditional Analysis
- **Stepwise conditional analysis**: Adding top SNPs as covariates
- **Joint modeling**: Multi-SNP conditional regression
- **Independence testing**: Distinguishing independent signals
- **Credible set construction**: 95% probability intervals

#### 3.3 Functional Annotation
- **eQTL analysis**: Expression quantitative trait loci
- **Chromatin state**: Open chromatin, histone modifications
- **Regulatory motifs**: Transcription factor binding sites
- **Conservation scores**: Evolutionary constraint

#### 3.4 Integration with Functional Genomics
- **ENCODE data**: Experimental annotations
- **Roadmap Epigenomics**: Tissue-specific regulatory elements
- **GTEx**: Tissue-specific gene expression
- **Variant prioritization**: Combined annotation approaches

### 4. Polygenic Risk Scores and Risk Prediction (20 min)
#### 4.1 PRS Construction
- **Clumping and thresholding**: LD-based SNP selection
- **Effect size weighting**: Odds ratios or beta coefficients
- **Risk allele identification**: Reference allele assignment
- **Score normalization**: Standardization across populations

#### 4.2 PRS Applications
- **Disease risk prediction**: Odds ratios for different quantiles
- **Genetic correlation**: Across traits and diseases
- **Stratified medicine**: Treatment response prediction
- **Population screening**: Public health applications

#### 4.3 PRS Evaluation
- **Discriminative ability**: AUC, explained variance
- **Calibration**: Observed vs expected risks
- **Robustness**: Cross-validation and out-of-sample testing
- **Potential biases**: Winner's curse, overfitting

## Interactive Exercises

### Exercise 2.1: GWAS Result Quality Assessment
**Objective**: Learn to assess GWAS quality and identify potential issues

**Dataset**: GWAS summary statistics with varying quality issues

**Tasks**:
1. Calculate genomic inflation factor (λ)
2. Create QQ plot and identify inflation sources
3. Assess sample and SNP quality metrics
4. Apply appropriate quality filters

**Questions**:
1. What does a λ > 1.05 indicate?
2. How might population stratification affect QQ plots?
3. What QC filters would you apply to this dataset?

### Exercise 2.2: Manhattan Plot Analysis
**Objective**: Practice GWAS result visualization and interpretation

**Dataset**: Simulated GWAS results with known causal variants

**Tasks**:
1. Create Manhattan plot from summary statistics
2. Identify top association signals
3. Zoom in on significant regions
4. Annotate peaks with nearest genes

**Questions**:
1. How many independent association signals do you observe?
2. What chromosome has the strongest association?
3. How would you prioritize regions for follow-up?

### Exercise 2.3: Fine-Mapping Analysis
**Objective**: Learn to fine-map GWAS associations using LD information

**Dataset**: Regional association data with LD information

**Tasks**:
1. Visualize LD structure in the region
2. Perform conditional analysis to identify independent signals
3. Construct credible sets for each association
4. Integrate functional annotations

**Questions**:
1. How many independent associations are there?
2. Which SNPs have the highest posterior probability?
3. What functional evidence supports the causal variants?

### Exercise 2.4: Polygenic Risk Score Calculation
**Objective**: Construct and evaluate polygenic risk scores

**Dataset**: GWAS discovery and target datasets for disease risk

**Tasks**:
1. Select SNPs for PRS using clumping and thresholding
2. Calculate PRS for target samples
3. Compare risk distribution by disease status
4. Evaluate PRS predictive performance

**Code Example**:
```python
import pandas as pd
import numpy as np
from scipy import stats

# Load GWAS summary statistics
gwas_data = pd.read_csv('gwas_summary.csv')

# Load target genotype data
genotypes = pd.read_csv('target_genotypes.csv')

# Calculate PRS using different p-value thresholds
def calculate_prs(genotypes, gwas_data, p_thresholds):
    prs_scores = {}

    for p_thresh in p_thresholds:
        # Select SNPs passing threshold
        selected_snps = gwas_data[gwas_data['P'] < p_thresh]

        # Weight by effect size (log OR)
        weighted_snps = selected_snps.copy()
        weighted_snps['weight'] = np.log(weighted_snps['OR'])

        # Calculate PRS
        prs = np.zeros(genotypes.shape[0])
        for _, snp in weighted_snps.iterrows():
            snp_col = snp['SNP']
            if snp_col in genotypes.columns:
                effect_allele = snp['A1']
                # Count effect alleles (simplified - real implementation more complex)
                effect_count = (genotypes[snp_col] == effect_allele).astype(int)
                prs += effect_count * snp['weight']

        prs_scores[p_thresh] = prs

    return prs_scores

# Calculate PRS for different thresholds
p_thresholds = [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]
prs_results = calculate_prs(genotypes, gwas_data, p_thresholds)

# Evaluate predictive performance
target_phenotypes = pd.read_csv('target_phenotypes.csv')
for p_thresh, scores in prs_results.items():
    auc = calculate_auc(target_phenotypes['disease'], scores)
    print(f"P-threshold: {p_thresh}, AUC: {auc:.3f}")
```

## Practical Demonstration

### Demo 1: Manhattan Plot with PLINK
```bash
# Create Manhattan plot from PLINK association results
# (Assuming association results in logistic.assoc.logistic file)

# Sort by chromosome and position
sort -k1,1n -k3,3n logistic.assoc.logistic > sorted_results.txt

# Use R to create Manhattan plot
Rscript -e "
library(qqman)
results <- read.table('sorted_results.txt', header=TRUE)
manhattan(results, chr='CHR', bp='BP', p='P', snp='SNP',
         suggestiveline = -log10(1e-5), genomewideline = -log10(5e-8))
"
```

### Demo 2: PRS Construction with PRSice
```bash
# Calculate PRS using PRSice
./PRSice_linux \
    --base discovery_gwas.tsv \
    --target target_genotypes \
    --binary-target T \
    --pheno target_phenotypes.phen \
    --cov target_covariates.cov \
    --out prs_results \
    --bar-levels 0.5,0.1,0.01,0.001,0.0001 \
    --fastscore
```

### Demo 3: Regional Visualization with LocusZoom
```bash
# Create regional association plot
# Requires: GWAS results, LD reference panel, gene annotation

locuszoom \
    --metal gwas_results.txt \
    --refgene gene_annotation.txt \
    --ld hapmap_ld_info.txt \
    --prefix region_plot \
    --chr 1 \
    --start 1000000 \
    --end 2000000 \
    --markercol SNP \
    --pvalcol P \
    --marker SNP_OF_INTEREST
```

## Advanced Topics

### Meta-Analysis Approaches
- **Fixed effects**: Inverse variance weighting
- **Random effects**: Account for heterogeneity
- **Heterogeneity testing**: Cochran's Q, I² statistic
- **Genome-wide meta-analysis**: Large-scale combination of studies

### Cross-Ancestry GWAS
- **Trans-ethnic meta-analysis**: Different ancestry groups
- **Ancestry-specific effects**: Populations differences
- **Admixed populations**: Local ancestry estimation
- **Polygenic scores**: Transferability across ancestries

### Functional GWAS Enrichment
- **gene-set analysis**: MAGMA, VEGAS
- **tissue-specific enrichment**: DEPICT, PASCAL
- **epigenetic annotation**: GARFIELD, PAINTOR
- **multi-trait analysis**: MTAG, GWAS-by-subtraction

## Key Takeaways
- GWAS visualization tools are essential for result interpretation
- Quality control metrics ensure reliable conclusions
- Fine-mapping identifies causal variants within associated regions
- Polygenic risk scores enable risk prediction applications
- Integration with functional genomics data enhances biological interpretation
- Meta-analysis increases statistical power for discovery
- Cross-validation and replication are crucial for validation

## Common GWAS Analysis Challenges

**False Positives**:
- Multiple testing correction insufficient
- Poor study design or data quality
- Population stratification artifacts
*Solution*: Stringent QC, replication in independent datasets

**False Negatives**:
- Underpowered studies
- Insufficient genomic coverage
- Rare variant effects missed
*Solution*: Larger sample sizes, careful phenotype definition

**Power Issues**:
- Small effect sizes require large samples
- Winner's curse inflates effect estimates
- Publication bias affects meta-analyses
*Solution*: Realistic power calculations, cross-validation

**Interpretation Problems**:
- Most SNPs in non-coding regions
- Weak biological insights without functional annotation
- Complex trait architecture poorly understood
*Solution*: Integration with multiomics data, pathway analysis

## Further Reading
- Dudbridge F & Gusnanto A (2008). Estimation of significance thresholds for genomewide association scans. *Genetic Epidemiology*.
- Skol AD et al. (2006). Joint analysis is more efficient than replication-based analysis for two-stage genome-wide association studies. *Nature Genetics*.
- Magi R & Morris AP (2017). GWAMA: Software for genome-wide association meta-analysis. *BMC Bioinformatics*.

## Session Assessment
**Pre/Post Test Questions**:
1. What does a λ value > 1.1 indicate in GWAS quality control?
2. How do you interpret peaks in a Manhattan plot?
3. What is the purpose of fine-mapping in GWAS?
4. How are polygenic risk scores calculated and validated?

## Next Session Preview
**Session 3**: Multiomics Integration Fundamentals - Combining genomics with transcriptomics, proteomics, and metabolomics data.
