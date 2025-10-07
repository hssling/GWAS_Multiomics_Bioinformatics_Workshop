# Workshop Knowledge Assessment

## Instructions
This quiz assesses your understanding of the key concepts covered in the GWAS, Multiomics Integration, and Bioinformatics Workshop. Each question is worth 2 points. Select the best answer for each question.

---

## Section 1: GWAS Fundamentals (Questions 1-5)

### Question 1
What is the primary goal of genome-wide association studies (GWAS)?
- A) Identify all genes in the human genome
- B) Find genetic variants associated with traits or diseases
- C) Determine the complete DNA sequence of individuals
- D) Predict protein structures from DNA sequences

### Question 2
Which of the following is the correct Bonferroni-corrected genome-wide significance threshold?
- A) p < 0.01
- B) p < 0.05
- C) p < 5 × 10⁻⁸
- D) p < 0.001

### Question 3
What does the genomic inflation factor (λ) > 1.05 indicate in GWAS quality control?
- A) Underpowered study
- B) Population stratification or other confounding
- C) Perfect quality control
- D) Too many SNPs genotyped

### Question 4
Which plot is used to visualize GWAS results across the entire genome?
- A) Volcano plot
- B) Manhattan plot
- C) QQ plot
- D) Box plot

### Question 5
What is linkage disequilibrium (LD)?
- A) Physical distance between genes
- B) Non-random association between nearby genetic variants
- C) The process of genetic recombination
- D) Variation in gene expression levels

---

## Section 2: Multiomics Integration (Questions 6-10)

### Question 6
Which of the following is NOT a common type of omics data?
- A) Genomics
- B) Transcriptomics
- C) Imaging-omics
- D) Radiomics

### Question 7
What is the main purpose of batch effect correction in multiomics studies?
- A) Remove technical variation between batches
- B) Introduce artificial variation
- C) Increase dataset size
- D) Reduce computational requirements

### Question 8
Which integration method uses probabilistic factor models to identify shared patterns across omics layers?
- A) Principal Component Analysis (PCA)
- B) Multi-Omics Factor Analysis (MOFA)
- C) Canonical Correlation Analysis (CCA)
- D) t-Distributed Stochastic Neighbor Embedding (t-SNE)

### Question 9
What does SNF stand for in multiomics integration?
- A) Single Nucleotide Fusion
- B) Similarity Network Fusion
- C) Systematic Network Filtering
- D) Statistical Noise Filtering

### Question 10
Which of the following is a challenge in multiomics data integration?
- A) Too few data types
- B) Data heterogeneity across platforms
- C) Lack of biological variation
- D) Perfect correlation between all omics layers

---

## Section 3: Bioinformatics Pipelines (Questions 11-15)

### Question 11
Which tool is commonly used for aligning sequencing reads to a reference genome?
- A) FastQC
- B) BWA
- C) MultiQC
- D) Cutadapt

### Question 12
What does FASTQ format contain?
- A) Aligned sequencing reads
- B) Base quality scores only
- C) Nucleotide sequences and quality scores
- D) Gene annotation information

### Question 13
Which of the following is used for differential expression analysis of RNA-seq data?
- A) BWA
- B) GATK
- C) DESeq2
- D) FastQC

### Question 14
What is the purpose of multiple testing correction in omics analysis?
- A) Increase false positive rate
- B) Control for false discoveries when testing many hypotheses
- C) Only apply to GWAS data
- D) Remove all p-values below 0.05

### Question 15
Which workflow management system uses Python-based rules to define analysis pipelines?
- A) Nextflow
- B) Docker
- C) Snakemake
- D) Galaxy

---

## Section 4: Precision Medicine (Questions 16-20)

### Question 16
What is precision medicine?
- A) A new type of medical imaging
- B) Treatment based on individual biological characteristics
- C) Using only genomic information for diagnosis
- D) Population-level health interventions only

### Question 17
What is a polygenic risk score (PRS)?
- A) A single genetic variant with large effect
- B) A score combining effects from multiple genetic variants
- C) A measure of environmental risk factors
- D) A type of medical imaging score

### Question 18
Which genetic variant is commonly associated with warfarin dosing?
- A) BRCA1/2
- B) CYP2C9
- C) HLA alleles
- D) APOE ε4

### Question 19
What is one major challenge in implementing precision medicine?
- A) Too much genomic data available
- B) Lack of technical knowledge
- C) Health equity and access to technology
- D) Overabundance of effective treatments

### Question 20
Which cancer type commonly uses multiomics profiling for treatment selection?
- A) All cancers equally
- B) Only blood cancers
- C) Non-small cell lung cancer (oncogene testing)
- D) Only skin cancers

---

## Answer Key

### Section 1: GWAS Fundamentals
1. **B** - Find genetic variants associated with traits or diseases
2. **C** - p < 5 × 10⁻⁸
3. **B** - Population stratification or other confounding
4. **B** - Manhattan plot
5. **B** - Non-random association between nearby genetic variants

### Section 2: Multiomics Integration
6. **D** - Radiomics (while related, not a primary omics type)
7. **A** - Remove technical variation between batches
8. **B** - Multi-Omics Factor Analysis (MOFA)
9. **B** - Similarity Network Fusion
10. **B** - Data heterogeneity across platforms

### Section 3: Bioinformatics Pipelines
11. **B** - BWA
12. **C** - Nucleotide sequences and quality scores
13. **C** - DESeq2
14. **B** - Control for false discoveries when testing many hypotheses
15. **C** - Snakemake

### Section 4: Precision Medicine
16. **B** - Treatment based on individual biological characteristics
17. **B** - A score combining effects from multiple genetic variants
18. **B** - CYP2C9
19. **C** - Health equity and access to technology
20. **C** - Non-small cell lung cancer (oncogene testing)

---

## Performance Rubric

- **36-40 points (90-100%)**: Excellent understanding of all workshop concepts
- **32-35 points (80-89%)**: Strong grasp of key concepts with minor gaps
- **28-31 points (70-79%)**: Good understanding of most concepts
- **24-27 points (60-69%)**: Basic understanding with some significant gaps
- **0-23 points (<60%)**: Needs review of fundamental concepts

---

## Learning Objectives Assessment

This quiz evaluates the following workshop learning objectives:

- ✅ **Identify genetic variants and understand GWAS principles**
- ✅ **Interpret and visualize GWAS results**
- ✅ **Apply multiomics data integration techniques**
- ✅ **Navigate bioinformatics analysis pipelines**
- ✅ **Understand precision medicine applications and challenges**
- ✅ **Critically evaluate bioinformatics evidence and limitations**

---

## Continuing Education

Based on your quiz performance:

### If you scored 90-100%:
- Congratulations! You have excellent mastery of the material
- Consider advanced topics in the Innovative_Methods directory
- Explore real-world applications in precision medicine

### If you scored 70-89%:
- Solid foundation in key concepts
- Review specific areas of difficulty
- Practice with the interactive tools in the workshop
- Apply concepts to real datasets

### If you scored below 70%:
- Focus on fundamental concepts in Sessions 1-3
- Work through the exercises in the Exercises directory
- Use the interactive Streamlit app for hands-on practice
- Consider retaking the quiz after additional study

### Additional Resources:
- 📚 [Nature Genetics GWAS review articles](https://www.nature.com/ng/)
- 🎓 [Online courses on Coursera/edX for genomics](https://www.coursera.org/)
- 🛠️ [Bioconductor packages for bioinformatics](https://www.bioconductor.org/)
- 📊 [GWAS Catalog for real data exploration](https://www.ebi.ac.uk/gwas/)
