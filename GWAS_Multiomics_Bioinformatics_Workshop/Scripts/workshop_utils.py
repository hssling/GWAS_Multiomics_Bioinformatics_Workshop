#!/usr/bin/env python3
"""
Workshop Utility Functions

A collection of utility functions for GWAS, Multiomics Integration,
and Bioinformatics Workshop analyses.

Author: Dr. Siddalingaiah H S
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_gwas_data(filepath=None):
    """
    Load GWAS summary statistics

    Parameters:
        filepath: Path to GWAS summary stats CSV file

    Returns:
        pd.DataFrame: GWAS summary statistics
    """
    if filepath is None:
        # Use sample data
        data_dir = Path(__file__).parent.parent / "Data"
        filepath = data_dir / "sample_gwas_summary_stats.csv"

    if not filepath.exists():
        raise FileNotFoundError(f"GWAS data file not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Loaded GWAS data: {len(df):,} SNPs from {len(df['CHR'].unique())} chromosomes")
    return df

def load_expression_data(filepath=None):
    """
    Load gene expression data

    Parameters:
        filepath: Path to expression CSV file

    Returns:
        pd.DataFrame: Gene expression data
    """
    if filepath is None:
        # Use sample data
        data_dir = Path(__file__).parent.parent / "Data"
        filepath = data_dir / "sample_gene_expression.csv"

    if not filepath.exists():
        raise FileNotFoundError(f"Expression data file not found: {filepath}")

    df = pd.read_csv(filepath, index_col=0)
    print(f"Loaded expression data: {df.shape[0]} genes × {df.shape[1]} samples")
    return df

def create_manhattan_plot(gwas_df, title="GWAS Manhattan Plot",
                         sig_threshold=5e-8, suggestive_threshold=1e-5,
                         save_path=None):
    """
    Create a Manhattan plot from GWAS data

    Parameters:
        gwas_df: DataFrame with CHR, BP, P columns
        title: Plot title
        sig_threshold: Genome-wide significance threshold
        suggestive_threshold: Suggestive significance threshold
        save_path: Path to save plot (optional)
    """
    # Prepare data for plotting
    plot_data = gwas_df.copy()

    # Calculate cumulative positions
    plot_data['cum_pos'] = 0
    chrom_starts = {}

    for chr_num in sorted(plot_data['CHR'].unique()):
        chrom_data = plot_data[plot_data['CHR'] == chr_num]
        chr_end = plot_data[plot_data['CHR'] < chr_num]['BP'].max() if chr_num > 1 else 0
        chrom_starts[chr_num] = chr_end
        plot_data.loc[plot_data['CHR'] == chr_num, 'cum_pos'] = chrom_data['BP'] + chr_end

        # Add gap between chromosomes
        if chr_num < plot_data['CHR'].max():
            gap = plot_data['BP'].max() * 0.02  # 2% gap
            plot_data.loc[plot_data['CHR'] > chr_num, 'cum_pos'] += gap

    # Create plot
    fig, ax = plt.subplots(figsize=(15, 8))

    # Color scheme
    colors = ['#1f77b4', '#ff7f0e'] * 11  # Alternate colors
    chr_colors = {chr_num: colors[chr_num-1] for chr_num in plot_data['CHR'].unique()}

    # Plot each chromosome
    for chr_num in sorted(plot_data['CHR'].unique()):
        chr_data = plot_data[plot_data['CHR'] == chr_num]
        ax.scatter(chr_data['cum_pos'], chr_data['P'],
                  c=chr_colors[chr_num], s=2, alpha=0.8,
                  label=f'Chr {chr_num}')

    # Significance lines
    ax.axhline(y=-np.log10(sig_threshold), color='red', linestyle='--', linewidth=1,
               label='.0f'               ax.axhline(y=-np.log10(suggestive_threshold), color='orange',
               linestyle='--', linewidth=1,
               label='.0f'
    # Chromosome labels (simplified)
    chrom_centers = []
    chrom_labels = []
    for chr_num in sorted(plot_data['CHR'].unique()):
        chr_data = plot_data[plot_data['CHR'] == chr_num]
        center_pos = chr_data['cum_pos'].mean()
        chrom_centers.append(center_pos)
        chrom_labels.append(f'{chr_num}')

    ax.set_xticks(chrom_centers)
    ax.set_xticklabels(chrom_labels, rotation=45)

    ax.set_xlabel('Chromosome', fontsize=12)
    ax.set_ylabel('-log₁₀(p-value)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig, ax

def calculate_qq_plot(gwas_df, save_path=None):
    """
    Create QQ plot and calculate inflation factor

    Parameters:
        gwas_df: DataFrame with P column
        save_path: Path to save plot (optional)

    Returns:
        float: Genomic inflation factor λ
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Observed p-values (sorted)
    observed = np.sort(gwas_df['P'])[::-1]

    # Expected p-values under null
    n_snps = len(gwas_df)
    expected = -np.log10(np.arange(1, n_snps + 1) / (n_snps + 1))

    # Plot
    ax.scatter(expected, observed, alpha=0.5, s=2, color='blue')

    # Diagonal line
    max_val = max(max(observed), max(expected))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1, label='Expected')

    # Calculate λ (inflation factor)
    lambda_gc = np.median(10 ** -observed) / np.median(10 ** -expected)

    # Labels and title
    ax.set_xlabel('Expected -log₁₀(p-value)', fontsize=12)
    ax.set_ylabel('Observed -log₁₀(p-value)', fontsize=12)
    ax.set_title('.3f', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"QQ plot saved to: {save_path}")

    return lambda_gc

def perform_differential_expression(expression_df, sample_metadata,
                                  group_col='condition', group1='Control', group2='Disease',
                                  alpha=0.05, save_results=True):
    """
    Perform differential expression analysis

    Parameters:
        expression_df: Gene expression DataFrame
        sample_metadata: Sample metadata DataFrame
        group_col: Column name for grouping variable
        group1, group2: Names of the two groups
        alpha: Significance threshold
        save_results: Whether to save results to CSV

    Returns:
        DataFrame: Differential expression results
    """
    print(f"Performing differential expression analysis: {group1} vs {group2}")

    # Get samples for each group
    group1_samples = sample_metadata[sample_metadata[group_col] == group1]['sample_id'].values
    group2_samples = sample_metadata[sample_metadata[group_col] == group2]['sample_id'].values

    print(f"Group 1 ({group1}): {len(group1_samples)} samples")
    print(f"Group 2 ({group2}): {len(group2_samples)} samples")

    results = []

    # Analyze each gene
    for gene in expression_df.index:
        try:
            group1_expr = expression_df.loc[gene, group1_samples].values
            group2_expr = expression_df.loc[gene, group2_samples].values

            # Calculate fold change
            fold_change = np.mean(group2_expr) / np.mean(group1_expr)
            log2_fc = np.log2(fold_change)

            # t-test
            t_stat, p_value = stats.ttest_ind(group2_expr, group1_expr)

            results.append({
                'gene': gene,
                'fold_change': fold_change,
                'log2_fold_change': log2_fc,
                'p_value': p_value,
                't_statistic': t_stat,
                f'mean_{group1}': np.mean(group1_expr),
                f'mean_{group2}': np.mean(group2_expr),
                f'std_{group1}': np.std(group1_expr),
                f'std_{group2}': np.std(group2_expr)
            })

        except Exception as e:
            print(f"Warning: Could not analyze {gene}: {e}")
            continue

    # Create results DataFrame
    de_results = pd.DataFrame(results)

    # Apply multiple testing correction (FDR)
    from statsmodels.stats.multitest import multipletests
    _, p_adjusted, _, _ = multipletests(de_results['p_value'], method='fdr_bh', alpha=alpha)
    de_results['adjusted_p_value'] = p_adjusted
    de_results['significant'] = p_adjusted < alpha

    # Sort by adjusted p-value
    de_results = de_results.sort_values('adjusted_p_value')

    print("
Analysis complete:")
    n_sig = sum(de_results['significant'])
    print(f"  - Total genes analyzed: {len(de_results)}")
    print(".2f"    print(f"  - Significant genes (FDR < {alpha}): {n_sig}")
    print(".1f"
    if save_results:
        output_path = "differential_expression_results.csv"
        de_results.to_csv(output_path, index=False)
        print(f"  - Results saved to: {output_path}")

    return de_results

def perform_pca_analysis(data_matrix, n_components=10, standardize=True):
    """
    Perform PCA analysis on omics data

    Parameters:
        data_matrix: Data matrix (samples × features)
        n_components: Number of principal components
        standardize: Whether to z-score standardize

    Returns:
        dict: PCA results (pca_object, scores, loadings, explained_variance)
    """
    print(f"Performing PCA analysis on {data_matrix.shape[0]} samples × {data_matrix.shape[1]} features")

    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_matrix)
    else:
        data_scaled = data_matrix.copy()

    # Perform PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(data_scaled)

    # Get loadings (principal component coefficients)
    loadings = pca.components_.T

    # Explained variance
    explained_variance = pca.explained_variance_ratio_ * 100

    results = {
        'pca': pca,
        'scores': scores,
        'loadings': loadings,
        'explained_variance': explained_variance,
        'cumulative_variance': np.cumsum(explained_variance)
    }

    print(".1f")
    print(".1f")

    return results

def create_pca_plot(pca_results, sample_labels=None, color_labels=None,
                   title="PCA Analysis", save_path=None):
    """
    Create PCA scatter plot

    Parameters:
        pca_results: Results from perform_pca_analysis
        sample_labels: Labels for samples
        color_labels: Labels for coloring points
        title: Plot title
        save_path: Path to save plot
    """
    scores = pca_results['scores']
    explained_var = pca_results['explained_variance']

    fig, ax = plt.subplots(figsize=(10, 8))

    if color_labels is not None:
        # Color by category
        unique_labels = np.unique(color_labels)
        colors = sns.color_palette('husl', len(unique_labels))
        color_map = dict(zip(unique_labels, colors))

        for label in unique_labels:
            mask = color_labels == label
            ax.scatter(scores[mask, 0], scores[mask, 1],
                      c=[color_map[label]], label=label, alpha=0.7, s=50)

        ax.legend()
    else:
        # Simple scatter plot
        ax.scatter(scores[:, 0], scores[:, 1], alpha=0.7, s=50)

    ax.set_xlabel('.1f'    ax.set_ylabel('.1f'    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA plot saved to: {save_path}")

    return fig, ax

def calculate_power_analysis(maf, odds_ratio, sample_size, alpha=5e-8):
    """
    Calculate statistical power for GWAS

    Parameters:
        maf: Minor allele frequency
        odds_ratio: Genetic effect size
        sample_size: Total sample size (cases + controls)
        alpha: Significance threshold

    Returns:
        dict: Power analysis results
    """
    # Simplified power calculation (approximation)
    # In practice, use tools like Genetic Power Calculator

    # Calculate detectable effect size
    detectable_or = 1.0  # Placeholder - would need proper calculation

    # Estimate power (rough approximation)
    effect_size = abs(np.log(odds_ratio) / np.log(1.5))  # Normalized effect
    base_power = min(0.8, effect_size * maf * np.sqrt(sample_size / 10000))

    # Adjust for significance threshold
    threshold_factor = np.log10(alpha) / np.log10(0.05)
    power = min(0.95, base_power * abs(threshold_factor))

    results = {
        'estimated_power': power,
        'maf': maf,
        'odds_ratio': odds_ratio,
        'sample_size': sample_size,
        'alpha': alpha,
        'detectable_effect': detectable_or
    }

    return results

# Workshop metadata
__version__ = "1.0.0"
__author__ = "Dr. Siddalingaiah H S"
__description__ = "Utility functions for GWAS, Multiomics Integration, and Bioinformatics Workshop"

if __name__ == "__main__":
    print("GWAS & Multiomics Bioinformatics Workshop Utilities")
    print("=" * 50)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Description: {__description__}")
    print()
    print("Available functions:")
    functions = [name for name in dir() if not name.startswith('_')]
    for func in functions:
        if callable(eval(func)):
            print(f"  - {func}")
    print()
    print("For usage examples, see workshop materials!")
