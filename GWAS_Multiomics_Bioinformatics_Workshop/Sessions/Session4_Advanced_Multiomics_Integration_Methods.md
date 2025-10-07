# Session 4: Advanced Multiomics Integration Methods

## Learning Objectives
By the end of this session, participants will be able to:
- Apply machine learning methods for multiomics integration
- Understand network-based integration approaches
- Implement joint dimensionality reduction techniques
- Evaluate integration method performance
- Select appropriate methods for different biological questions

## Pre-Reading Materials
- Argelaguet R et al. (2018). Multi-Omics Factor Analysis—a framework for unsupervised integration of multi-omics data sets. *Molecular Systems Biology*.
- Wang B et al. (2016). Similarity network fusion for aggregating data types on a genomic scale. *Nature Methods*.

## Presentation Outline

### 1. Machine Learning Approaches for Integration (25 min)
#### 1.1 Supervised Integration Methods
- **Multi-task learning**: Joint prediction across modalities
- **Deep learning integration**: Neural network-based approaches
  - Autoencoders for joint embedding
  - Multi-modal neural networks
  - Generative adversarial networks (GANs)
- **Ensemble methods**: Combining predictions across modalities

#### 1.2 Unsupervised Integration Methods
- **Multi-Omics Factor Analysis (MOFA)**:
  - Probabilistic factor model
  - Joint dimensionality reduction
  - Factor-specific variance explained
  - Model selection and interpretation
- **Joint Non-negative Matrix Factorization (jNMF)**:
  - Shared and modality-specific factors
  - Biological interpretation
  - Sparsity constraints

#### 1.3 Semi-supervised Approaches
- **Canonical Correlation Analysis (CCA)**: Optimal linear relationships
- **Sparse CCA**: Feature selection and regularization
- **Deep CCA**: Non-linear correlation learning
- **Partial Least Squares (PLS)**: Predictive dimensionality reduction

### 2. Network-Based Integration Methods (25 min)
#### 2.1 Similarity Network Fusion (SNF)
- **Algorithm overview**: Iterative fusion of similarity networks
- **Similarity matrix construction**: Distance-based similarities
- **Network fusion**: Patient similarity integration
- **Cluster identification**: Multi-scale clustering

#### 2.2 Graph-Based Integration
- **Multi-omics graphs**: Nodes as features, edges as relationships
- **Graph neural networks**: Learning on multi-omics graphs
- **Diffusion-based methods**: Information propagation
- **Community detection**: Identifying functional modules

#### 2.3 Biological Network Integration
- **Protein-protein interaction networks**: Functional relationships
- **Regulatory networks**: Transcription factor-gene interactions
- **Metabolic pathways**: Biochemical reaction networks
- **Disease networks**: Gene-disease associations

#### 2.4 Multi-Scale Network Analysis
- **Intra-modality networks**: Within-omics relationships
- **Inter-modality networks**: Cross-omics connections
- **Patient-specific networks**: Personalized network models
- **Temporal networks**: Longitudinal multiomics integration

### 3. Advanced Statistical Methods (20 min)
#### 3.1 Mixed Effects Models
- **Random effects integration**: Accounting for study heterogeneity
- **Meta-analysis frameworks**: Combining multiple studies
- **Hierarchical Bayesian models**: Probabilistic integration
- **Latent variable models**: Unobserved factor estimation

#### 3.2 Bayesian Integration Approaches
- **Bayesian networks**: Probabilistic graphical models
- **Markov chain Monte Carlo**: Posterior inference
- **Variational inference**: Approximate posterior computation
- **Hierarchical models**: Multi-level data integration

#### 3.3 Robust Statistical Methods
- **Robust regression**: Outlier-resistant integration
- **Surrogate variable analysis**: Latent factor identification
- **Combat-seq**: Batch effect correction for RNA-seq
- **RUV (Remove Unwanted Variation)**: Negative control-based correction

### 4. Integration Method Evaluation and Selection (20 min)
#### 4.1 Performance Metrics
- **Biological relevance**: Pathway enrichment, known associations
- **Clinical utility**: Disease classification, survival prediction
- **Stability**: Reproducibility across subsamples
- **Interpretability**: Biological meaning of integrated features

#### 4.2 Cross-Validation Strategies
- **Internal validation**: Train/test splits
- **External validation**: Independent datasets
- **Nested cross-validation**: Hyperparameter tuning
- **Stratified sampling**: Class balance preservation

#### 4.3 Comparative Analysis
- **Benchmarking frameworks**: Standardized evaluation
- **Method comparison**: ROC curves, precision-recall plots
- **Computational efficiency**: Time and memory requirements
- **Scalability assessment**: Large dataset performance

#### 4.4 Method Selection Guidelines
- **Data characteristics**: Sample size, feature dimension, noise level
- **Biological question**: Subtype discovery, biomarker identification
- **Computational resources**: Available hardware and time constraints
- **Interpretability requirements**: Mechanistic vs predictive focus

## Interactive Exercises

### Exercise 4.1: Multi-Omics Factor Analysis (MOFA)
**Objective**: Apply MOFA for unsupervised multiomics integration

**Dataset**: Simulated 5-omics dataset (transcripts, proteins, metabolites, DNA methylation, miRNA)

**Tasks**:
1. Install and configure MOFA2
2. Prepare multiomics data for integration
3. Train MOFA model with different factor numbers
4. Interpret factor loadings and variance explained
5. Visualize integrated latent space

**Code Example**:
```python
import pandas as pd
import numpy as np
import mofapy2

# Load multiomics data
transcript_data = pd.read_csv("transcripts.csv", index_col=0)
protein_data = pd.read_csv("proteins.csv", index_col=0)
metabolite_data = pd.read_csv("metabolites.csv", index_col=0)
methylation_data = pd.read_csv("methylation.csv", index_col=0)
mirna_data = pd.read_csv("mirna.csv", index_col=0)

# Create MOFA model
model = mofapy2.models.mofa_model()
model.set_data([transcript_data.values, protein_data.values,
               metabolite_data.values, methylation_data.values,
               mirna_data.values])

# Set model parameters
model.set_ard_weights()
model.set_covariates()  # If available
model.set_factors(10)   # Number of latent factors

# Train model
model.build()
model.run()

# Extract results
factors = model.get_factors()
weights = model.get_weights()
variance_explained = model.get_variance_explained()

# Visualize factor 1 vs 2
plt.scatter(factors[:, 0], factors[:, 1])
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.title('MOFA Integration: Factor 1 vs 2')
plt.show()

# Check variance explained by each factor
print("Variance explained per factor:")
for i, var in enumerate(variance_explained):
    print(f"Factor {i+1}: {var:.1f}%")
```

### Exercise 4.2: Similarity Network Fusion
**Objective**: Implement SNF for patient stratification across multiple omics

**Dataset**: Cancer multiomics data (mRNA expression, DNA methylation, miRNA)

**Tasks**:
1. Construct similarity networks for each modality
2. Apply SNF fusion algorithm
3. Perform multi-scale clustering
4. Compare with single-modality clustering
5. Evaluate clustering stability and biological relevance

**Code Structure**:
```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from snf import compute

# Load multiomics data (samples x features)
mrna_data = pd.read_csv("mrna_expression.csv")
methylation_data = pd.read_csv("dna_methylation.csv")
mirna_data = pd.read_csv("mirna_expression.csv")

# Function to create similarity network
def make_similarity_network(data, k=20, alpha=0.5):
    # Compute Euclidean distances
    distances = euclidean_distances(data)
    # Convert to similarity matrix
    sigma = np.mean(distances)
    similarity = np.exp(-distances**2 / (2*sigma**2))

    # Create k-nearest neighbors network
    # (Implementation details for SNF similarity matrix construction)
    return similarity_network

# Create similarity networks
mrna_similarity = make_similarity_network(mrna_data.values)
methyl_similarity = make_similarity_network(methylation_data.values)
mirna_similarity = make_similarity_network(mirna_data.values)

# Fuse networks using SNF
fused_network = compute.snf([mrna_similarity, methyl_similarity, mirna_similarity],
                          K=20, t=20)

# Perform multi-scale clustering
from sklearn.cluster import spectral_clustering
n_clusters = 3
cluster_labels = spectral_clustering(fused_network, n_clusters=n_clusters)

# Evaluate clustering quality
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(euclidean_distances(fused_network),
                                cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.3f}")
```

### Exercise 4.3: Deep Learning Integration
**Objective**: Implement a multi-modal autoencoder for integrative feature learning

**Dataset**: Paired genomics and transcriptomics data

**Tasks**:
1. Design multi-modal autoencoder architecture
2. Implement joint representation learning
3. Train on paired multiomics data
4. Extract integrated latent representations
5. Compare with single-modality features

**Architecture Outline**:
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Multi-modal autoencoder architecture
class MultiOmicsAutoencoder(keras.Model):
    def __init__(self, input_dims, latent_dim=50):
        super().__init__()
        self.input_dims = input_dims
        self.latent_dim = latent_dim

        # Encoder for each modality
        self.encoders = []
        for dim in input_dims:
            encoder = keras.Sequential([
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(256, activation='relu'),
                layers.Dense(latent_dim, activation='relu')
            ])
            self.encoders.append(encoder)

        # Joint encoder
        self.joint_encoder = keras.Sequential([
            layers.Dense(latent_dim, activation='relu'),
            layers.Dense(latent_dim, activation='relu')
        ])

        # Joint decoder
        self.joint_decoder = keras.Sequential([
            layers.Dense(latent_dim, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(sum(input_dims), activation='linear')
        ])

    def call(self, inputs):
        # Encode each modality
        encoded_modalities = []
        for i, encoder in enumerate(self.encoders):
            encoded = encoder(inputs[i])
            encoded_modalities.append(encoded)

        # Joint encoding
        joint_input = tf.concat(encoded_modalities, axis=1)
        latent_representation = self.joint_encoder(joint_input)

        # Joint decoding
        reconstruction = self.joint_decoder(latent_representation)

        # Split reconstruction back to modalities
        splits = np.cumsum(self.input_dims)[:-1]
        reconstructed_modalities = tf.split(reconstruction, splits, axis=1)

        return latent_representation, reconstructed_modalities

# Usage
genomics_dim, transcriptomics_dim = 10000, 20000
model = MultiOmicsAutoencoder([genomics_dim, transcriptomics_dim])

# Compile and train
model.compile(optimizer='adam', loss='mse')
# Train with paired data...

# Extract latent representations
latent_features = model.predict([genomics_data, transcriptomics_data])[0]
```

## Practical Demonstration

### Demo 1: MOFA Analysis with Real Data
**Example**: Integration of single-cell multiomics data

**Steps**:
1. Load single-cell RNA and ATAC-seq data from 10X Genomics
2. Apply MOFA for joint dimensionality reduction
3. Identify multi-omics factors explaining variance
4. Visualize factor loadings on genome browser
5. Interpret biological meaning of factors

### Demo 2: Network-Based Patient Stratification
**Example**: Cancer subtype discovery using SNF

**Implementation**:
```bash
# Using SNFtool in R
library(SNFtool)

# Load multiomics matrices
data_list <- list(mrna_data, methylation_data, cnvs_data, mirna_data)

# Normalize and create distance matrices
dist_list <- lapply(data_list, function(x) {
  x <- standardNormalization(x)
  dist_matrix <- dist2(as.matrix(x), as.matrix(x))
  return(dist_matrix)
})

# Apply SNF fusion
W <- SNF(dist_list, K=20, t=20)

# Multi-scale clustering
clustering_results <- spectralClustering(W, K=3)

# Group samples by subtype
subtype_assignments <- clustering_results$group
```

### Demo 3: DIABLO for Predictive Integration
**Example**: Multiomics prediction of treatment response

**R Implementation**:
```r
library(mixOmics)

# Load multiomics blocks
X_genomics <- read.csv("genomics_data.csv")
X_transcriptomics <- read.csv("transcriptomics_data.csv")
X_proteomics <- read.csv("proteomics_data.csv")

# Create block list
X <- list(genomics = X_genomics,
          transcriptomics = X_transcriptomics,
          proteomics = X_proteomics)

Y <- read.csv("treatment_response.csv")$response

# Apply DIABLO
diablo_result <- block.splsda(X, Y, ncomp=3)

# Plot correlation between components
plotDiablo(diablo_result, ncomp=1)

# Feature selection and interpretation
selected_features <- selectVar(diablo_result, comp=1)
```

## Case Studies and Applications

### Case Study 1: Cancer Subtype Classification
- **Data**: TCGA multiomics (mRNA, methylation, CNV, miRNA, proteins)
- **Methods**: MOFA + consensus clustering
- **Outcome**: Novel subtypes with therapeutic implications
- **Clinical Impact**: Improved diagnostic classification

### Case Study 2: Drug Response Prediction
- **Data**: Genomics, transcriptomics, drug sensitivity testing
- **Methods**: SNF + elastic net regression
- **Outcome**: Multiomics signatures predicting drug response
- **Clinical Impact**: Personalized treatment selection

### Case Study 3: Functional Genomics Integration
- **Data**: GWAS + eQTL + chromatin data
- **Methods**: Network-based integration
- **Outcome**: Prioritized causal variants and genes
- **Clinical Impact**: Target identification for drug development

## Key Takeaways
- Advanced integration methods leverage complex relationships across modalities
- MOFA, SNF, and deep learning approaches offer different integration perspectives
- Method selection depends on data characteristics and biological questions
- Evaluation requires multiple metrics including biological relevance
- Cross-validation ensures robust performance estimation
- Scalability and interpretability are key considerations
- Integration enhances both discovery and prediction capabilities

## Method Selection Guide

| **Scenario** | **Recommended Method** | **Why** |
|---|---|---|
| Unsupervised discovery | MOFA | Probabilistic, interpretable factors |
| Patient stratification | SNF | Robust to noise, finds subtypes |
| Large datasets | CCA variants | Scalable linear relationships |
| Deep phenotyping | Joint NMF | Identifies shared/decomposed factors |
| Predictive modeling | DIABLO | Supervised integration |
| Network biology | Graph-based methods | Biological network relationships |

## Common Implementation Issues

**Computational Complexity**:
- MOFA and deep learning require significant resources
- Network methods scale poorly with large datasets
*Solution*: Subsampling, distributed computing, efficient algorithms

**Missing Data**:
- Multiomics datasets often have incomplete samples
- Different missingness patterns across modalities
*Solution*: Imputation methods, robust algorithms, pairwise analyses

**Data Heterogeneity**:
- Different scales, distributions, and noise levels
- Batch effects and technical variability
*Solution*: Normalization, transformation, batch correction

**Interpretability Challenges**:
- Complex models may lack biological insight
- Latent factors may not map to known biology
*Solution*: Feature selection, pathway analysis, orthogonal validation

## Further Reading
- Rappoport N et al. (2019). MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data. *Genome Biology*.
- Nguyen H et al. (2020). A comprehensive survey of regularization strategies for cancer signatures. *Frontiers in Genetics*.
- Singh A et al. (2019). Deep learning in genomics and biomedicine. *Genomics*.

## Session Assessment
**Pre/Post Test Questions**:
1. How does MOFA differ from traditional PCA for multiomics data?
2. What are the advantages of SNF over simpler concatenation approaches?
3. How do you choose between different integration methods?
4. What evaluation metrics are most important for integration quality?

## Next Session Preview
**Session 5**: Bioinformatics Tools and Pipelines - Building reproducible workflows for genomics and multiomics analysis.
