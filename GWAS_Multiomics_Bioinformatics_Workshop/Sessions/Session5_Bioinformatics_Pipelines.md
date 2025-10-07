# Session 5: Bioinformatics Tools and Pipelines

## Learning Objectives
By the end of this session, participants will be able to:
- Understand the principles of bioinformatics workflow design
- Navigate common NGS data analysis pipelines
- Use command-line tools for sequence analysis
- Build reproducible bioinformatics pipelines
- Apply best practices for data management and documentation

## Pre-Reading Materials
- Goecks J et al. (2010). Galaxy: A comprehensive approach for supporting accessible, reproducible, and transparent computational research. *Genome Biology*.
- Köster J & Rahmann S (2012). Snakemake--a scalable bioinformatics workflow engine. *Bioinformatics*.

## Presentation Outline

### 1. Introduction to Bioinformatics Workflows (20 min)
#### 1.1 Workflow Definition and Purpose
- **Definition**: Automated series of computational steps
- **Purpose**: Reproducibility, scalability, error reduction
- **Components**: Input/output files, parameters, dependencies
- **Quality assurance**: Validation, testing, monitoring

#### 1.2 Workflow Management Systems
- **Snakemake**: Python-based, file-oriented workflow engine
- **Nextflow**: Domain-specific language for computational pipelines
- **Galaxy**: Web-based platform for reproducible research
- **CWL/WDL**: Standards for workflow description and execution

#### 1.3 Best Practices in Pipeline Development
- **Modular design**: Independent, reusable components
- **Version control**: Git for tracking changes
- **Documentation**: README files, inline comments
- **Testing**: Unit tests, integration tests, validation
- **Containerization**: Docker, Singularity for reproducibility

### 2. Next-Generation Sequencing (NGS) Data Analysis (30 min)
#### 2.1 Raw Data Processing
- **FASTQ format**: Quality scores and sequence data
- **Quality assessment**: FASTQC, MultiQC reports
- **Quality metrics**: Per-base sequence quality, GC content
- **Read length distribution, adapter content**

#### 2.2 Read Trimming and Filtering
- **Adapter removal**: Cutadapt, Trim Galore
- **Quality trimming**: Sliding window approach
- **Length filtering**: Minimum/maximum read length
- **Contamination screening**: Removal of unwanted sequences

#### 2.3 Alignment and Mapping
- **Reference genome preparation**: Indexing with BWA, bowtie2
- **Alignment algorithms**:
  - BWA-MEM: Burrows-Wheeler transform for DNA
  - STAR: Spliced alignment for RNA-seq
  - HISAT2: Graph-based alignment for RNA-seq
- **Alignment scoring**: Matches, mismatches, gaps, clipping

#### 2.4 File Formats and Compression
- **SAM/BAM**: Sequence Alignment/Map format
- **BED**: Browser Extensible Data format
- **VCF**: Variant Call Format
- **GTF/GFF**: Gene feature formats
- **CRAM**: Compressed BAM format

### 3. Variant Calling and Annotation (25 min)
#### 3.1 Variant Calling Pipeline
- **GATK Best Practices**: Standard for germline SNP/indel calling
- **HaplotypeCaller**: Accurate local de novo assembly
- **Joint calling**: Multi-sample variant discovery
- **VQSR**: Variant quality score recalibration

#### 3.2 Quality Control and Filtering
- **Quality metrics**: Depth (DP), quality score (QUAL), mapping quality (MQ)
- **Hardy-Weinberg equilibrium**: Population genetics filter
- **Missingness filters**: Genotype call rate
- **Population frequency**: Minor allele frequency thresholds

#### 3.3 Functional Annotation
- **ANNOVAR**: Efficient annotation of genetic variants
- **VEP (Variant Effect Predictor)**: Ensembl-based annotation
- **SnpEff/SnpSift**: Variant annotation and filtering
- **Functional consequence prediction**: Missense, nonsense, splicing

#### 3.4 Database Resources
- **dbSNP**: Reference SNP database
- **ClinVar**: Clinical variant database
- **OMIM**: Online Mendelian Inheritance in Man
- **HGMD**: Human Gene Mutation Database

### 4. RNA-seq Analysis Pipeline (25 min)
#### 4.1 Gene Expression Quantification
- **Alignment-based quantification**: HTSeq, featureCounts
- **Alignment-free quantification**: Salmon, kallisto
- **Transcript assembly**: StringTie, Cufflinks
- **Expression metrics**: FPKM, TPM, raw counts

#### 4.2 Differential Expression Analysis
- **DESeq2**: Negative binomial distribution modeling
- **edgeR**: Empirical Bayes estimation
- **limma-voom**: Linear modeling approach
- **Statistical testing**: Wald test, likelihood ratio test
- **Multiple testing correction**: Benjamini-Hochberg FDR

#### 4.3 Quality Assessment
- **Sample correlation**: Pearson correlation matrix
- **Principal component analysis**: Batch effect detection
- **Gene detection**: Proportion of expressed genes
- **Read distribution**: Intronic, exonic, intergenic reads

#### 4.4 Functional Enrichment Analysis
- **GO enrichment**: Gene Ontology biological processes
- **KEGG pathways**: Metabolic and signaling pathways
- **GSEA**: Gene set enrichment analysis
- **Reactome**: Curated pathway database

## Command Line Tools Demonstration

### Demo 1: Quality Assessment with FASTQC
```bash
# Quality control of FASTQ files
fastqc sample_R1.fastq.gz sample_R2.fastq.gz --outdir fastqc_output/

# Multi-sample quality report
multiqc fastqc_output/ --outdir multiqc_output/
```

### Demo 2: Read Alignment with BWA
```bash
# Index reference genome
bwa index reference_genome.fa

# Align paired-end reads
bwa mem reference_genome.fa sample_R1.fastq.gz sample_R2.fastq.gz > aligned.sam

# Convert SAM to BAM and sort
samtools view -bS aligned.sam | samtools sort -o aligned_sorted.bam
samtools index aligned_sorted.bam
```

### Demo 3: Variant Calling with GATK
```bash
# Mark duplicates
gatk MarkDuplicates -I aligned_sorted.bam -O marked_duplicates.bam -M marked_dup_metrics.txt

# Base quality score recalibration
gatk BaseRecalibrator -I marked_duplicates.bam -R reference.fa --known-sites dbsnp.vcf -O recal_data.table
gatk ApplyBQSR -I marked_duplicates.bam --bqsr-recal-file recal_data.table -O recalibrated.bam

# Call variants
gatk HaplotypeCaller -I recalibrated.bam -R reference.fa -O raw_variants.vcf
```

## Interactive Exercises

### Exercise 5.1: Quality Assessment and Trimming
**Objective**: Learn quality control and preprocessing of NGS data

**Dataset**: Example FASTQ files with quality issues

**Tasks**:
1. Run FASTQC on raw FASTQ files
2. Identify quality issues (adapter contamination, low quality regions)
3. Perform quality trimming with cutadapt
4. Compare quality metrics before and after trimming

**Expected Outcomes**:
- Generate MultiQC reports
- Understand quality metrics interpretation
- Apply appropriate filtering thresholds

### Exercise 5.2: Differential Expression Analysis
**Objective**: Perform RNA-seq differential expression analysis

**Dataset**: Simulated RNA-seq count matrix and sample metadata

**Tasks**:
1. Load count data and sample information
2. Normalize expression values
3. Perform differential expression testing
4. Apply multiple testing correction
5. Visualize results (volcano plot, MA plot)

**Code Framework**:
```python
import pandas as pd
from scipy import stats
import statsmodels.stats.multitest as smm

# Load data
counts = pd.read_csv('gene_counts.csv', index_col=0)
samples = pd.read_csv('sample_info.csv')

# Simple differential expression (tutorial purpose)
control_mask = samples['condition'] == 'control'
disease_mask = samples['condition'] == 'disease'

p_values = []
fold_changes = []

for gene in counts.index:
    control_expr = counts.loc[gene, control_mask]
    disease_expr = counts.loc[gene, disease_mask]

    # Fold change
    fc = disease_expr.mean() / control_expr.mean()
    fold_changes.append(fc)

    # t-test
    t_stat, p_val = stats.ttest_ind(disease_expr, control_expr)
    p_values.append(p_val)

# Multiple testing correction
rejected, p_adjusted, _, _ = smm.multipletests(p_values, method='fdr_bh')
```

## Building Reproducible Pipelines

### Snakemake Example
```python
# Snakefile for RNA-seq analysis
rule all:
    input:
        "results/differential_expression.csv"

rule fastqc:
    input:
        "data/{sample}_{read}.fastq.gz"
    output:
        "qc/{sample}_{read}_fastqc.html"
    shell:
        "fastqc {input} -o qc/"

rule trim_reads:
    input:
        r1="data/{sample}_R1.fastq.gz",
        r2="data/{sample}_R2.fastq.gz"
    output:
        r1="trimmed/{sample}_R1_trimmed.fastq.gz",
        r2="trimmed/{sample}_R2_trimmed.fastq.gz"
    shell:
        "cutadapt -a ADAPTER_SEQUENCE -A ADAPTER_SEQUENCE "
        "-o {output.r1} -p {output.r2} {input.r1} {input.r2}"

rule align_reads:
    input:
        r1="trimmed/{sample}_R1_trimmed.fastq.gz",
        r2="trimmed/{sample}_R2_trimmed.fastq.gz"
    output:
        "aligned/{sample}.bam"
    shell:
        "hisat2 -x genome_index -1 {input.r1} -2 {input.r2} | "
        "samtools sort -o {output}"

rule quantify_expression:
    input:
        "aligned/{sample}.bam"
    output:
        "quant/{sample}.counts"
    shell:
        "featureCounts -a annotation.gtf -o {output} {input}"
```

### Nextflow Pipeline Structure
```nextflow
#!/usr/bin/env nextflow

/*
RNA-seq Analysis Pipeline
*/

params.reads = "data/*_{R1,R2}_001.fastq.gz"
params.genome = "reference/genome.fa"
params.gtf = "reference/annotation.gtf"

Channel.fromFilePairs(params.reads, checkIfExists: true).set { read_pairs }

process QUALITY_CONTROL {
    input:
    tuple sampleId, file(reads) from read_pairs

    output:
    file("*_fastqc.html") into fastqc_reports

    script:
    """
    fastqc ${reads}
    """
}

process TRIM_READS {
    input:
    tuple sampleId, file(reads) from read_pairs

    output:
    tuple sampleId, file("*trimmed*") into trimmed_reads

    script:
    """
    cutadapt -q 20 -a AGATCGGAAGAG -A AGATCGGAAGAG \\
             -o ${sampleId}_R1_trimmed.fastq.gz \\
             -p ${sampleId}_R2_trimmed.fastq.gz \\
             ${reads[0]} ${reads[1]}
    """
}

workflow.onComplete {
    println "Pipeline completed at: $workflow.complete"
    println "Execution status: ${workflow.success ? 'OK' : 'FAILED'}"
}
```

## Key Takeaways
- Bioinformatics workflows automate complex analysis steps
- Quality control is essential for reliable NGS results
- Command-line tools provide power and flexibility
- Workflow management systems ensure reproducibility
- Documentation and version control are crucial
- Containerization enables platform-independent execution
- Testing and validation prevent analysis errors

## Common Challenges and Solutions
- **Large data volumes**: Use efficient file formats and compression
- **Computational requirements**: Utilize high-performance computing clusters
- **Pipeline complexity**: Modular design and extensive testing
- **Version management**: Environment management and containerization
- **Interpretation**: Biological validation and orthogonal confirmation

## Further Reading
- Ewels P et al. (2016). MultiQC: Summarize analysis results for multiple tools and samples in a single report. *Bioinformatics*.
- Di Tommaso P et al. (2017). Nextflow enables reproducible computational workflows. *Nature Biotechnology*.
- Faust GG & Hall IM (2014). SAMBLASTER: Fast duplicate marking and structural variant read extraction. *Bioinformatics*.

## Session Assessment
**Pre/Post Test Questions**:
1. What are the main components of a bioinformatics pipeline?
2. Why is quality control important in NGS data analysis?
3. What is the difference between SAM and BAM file formats?
4. How do workflow management systems improve bioinformatics analysis?

## Final Session Preview
**Session 6**: Precision Medicine Applications - Translating multiomics discoveries into clinical applications, case studies, and future directions.
