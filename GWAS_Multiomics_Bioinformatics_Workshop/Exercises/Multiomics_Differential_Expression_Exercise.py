#!/usr/bin/env python3
"""
Multiomics Differential Expression Analysis Exercise

This script demonstrates how to perform differential expression analysis
on multiomics data, including gene expression, proteomics, and metabolomics.

Learning Objectives:
- Load and preprocess multiomics data
- Perform statistical testing for differential expression
- Apply multiple testing correction
- Create visualization plots (volcano plots, heatmaps)
- Interpret biological significance
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.multitest as smm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def load_sample_data():
    """
    Load sample gene expression data and create sample metadata

    Returns:
        tuple: (expression_data, sample_metadata)
    """
    # Load expression data
    expr_df = pd.read_csv('../Data/sample_gene_expression.csv', index_col=0)

    # Create sample metadata (50 control, 50 disease)
    n_samples = expr_df.shape[1]
    conditions = ['Control'] * (n_samples // 2) + ['Disease'] * (n_samples // 2)

    metadata = pd.DataFrame({
        'sample_id': expr_df.columns,
        'condition': conditions,
        'age': np.random.normal(50, 10, n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples)
    })

    return expr_df, metadata

def perform_differential_expression(expression_df, metadata, alpha=0.05):
    """
    Perform differential expression analysis using t-tests

    Parameters:
        expression_df: DataFrame with genes as rows, samples as columns
        metadata: DataFrame with sample information
        alpha: Significance threshold

    Returns:
        DataFrame with differential expression results
    """
    results = []

    # Get sample groups
    control_samples = metadata[metadata['condition'] == 'Control']['sample_id'].values
    disease_samples = metadata[metadata['condition'] == 'Disease']['sample_id'].values

    print(f"Control samples: {len(control_samples)}, Disease samples: {len(disease_samples)}")

    for gene in expression_df.index:
        control_expr = expression_df.loc[gene, control_samples].values
        disease_expr = expression_df.loc[gene, disease_samples].values

        # Calculate fold change
        fold_change = np.mean(disease_expr) / np.mean(control_expr)

        # Perform t-test
        try:
            t_stat, p_value = stats.ttest_ind(disease_expr, control_expr)
        except:
            p_value = 1.0
            t_stat = 0.0

        # Store results
        results.append({
            'gene': gene,
            'fold_change': fold_change,
            'log2_fold_change': np.log2(fold_change),
            'p_value': p_value,
            't_statistic': t_stat,
            'mean_control': np.mean(control_expr),
            'mean_disease': np.mean(disease_expr)
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Apply multiple testing correction
    rejected, p_adjusted, _, _ = smm.multipletests(results_df['p_value'],
                                                  method='fdr_bh',
                                                  alpha=alpha)

    results_df['adjusted_p_value'] = p_adjusted
    results_df['significant'] = rejected
    results_df['neg_log10_p'] = -np.log10(results_df['p_value'] + 1e-300)  # Avoid log(0)
    results_df['neg_log10_adj_p'] = -np.log10(results_df['adjusted_p_value'] + 1e-300)

    return results_df

def create_volcano_plot(results_df, title="Volcano Plot - Differential Expression"):
    """
    Create a volcano plot showing fold change vs statistical significance
    """
    # Color points by significance
    colors = ['red' if sig else 'blue' for sig in results_df['significant']]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all points
    ax.scatter(results_df['log2_fold_change'], results_df['neg_log10_p'],
               c=colors, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

    # Add horizontal line for significance threshold
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7,
               label=f'p = 0.05 (raw)')

    ax.axhline(y=-np.log10(results_df['adjusted_p_value'].max()[
               results_df['significant']]),
               color='orange', linestyle='--', alpha=0.7,
               label=f'FDR = 0.05')

    # Add vertical lines for fold change
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-1, color='gray', linestyle='--', alpha=0.5)

    # Label significant genes
    significant_genes = results_df[results_df['significant']].head(10)
    for _, gene in significant_genes.iterrows():
        ax.annotate(gene['gene'],
                   xy=(gene['log2_fold_change'], gene['neg_log10_p']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)

    ax.set_xlabel('log₂(Fold Change)', fontsize=12)
    ax.set_ylabel('-log₁₀(p-value)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig, ax

def create_heatmap(expression_df, metadata, top_n=50):
    """
    Create a heatmap showing expression patterns for top differentially expressed genes
    """
    # Get top differentially expressed genes (this would come from DE analysis)
    # For demo, just take first 50 genes
    top_genes = expression_df.index[:top_n]

    # Subset expression data
    plot_data = expression_df.loc[top_genes]

    # Sort samples by condition
    sorted_samples = metadata.sort_values(['condition', 'sample_id'])['sample_id'].values
    plot_data = plot_data[sorted_samples]

    # Create annotations
    col_annotations = ['red' if cond == 'Disease' else 'blue'
                      for cond in metadata.set_index('sample_id').loc[sorted_samples]['condition']]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(15, 10))

    # Center the data
    data_centered = plot_data - plot_data.mean(axis=1).values.reshape(-1, 1)

    im = ax.imshow(data_centered.values, cmap='RdBu_r', aspect='auto',
                   vmin=-2, vmax=2)  # Z-score like scaling

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Expression (centered)', rotation=270, labelpad=20)

    # Labels
    ax.set_xticks([])
    ax.set_yticks(range(0, len(top_genes), 10))
    ax.set_yticklabels([f'{gene}' for gene in top_genes[::10]])
    ax.set_xlabel('Samples')
    ax.set_ylabel('Genes')
    ax.set_title(f'Gene Expression Heatmap (Top {top_n} genes)\\nRed=Disease, Blue=Control',
                 fontsize=14, fontweight='bold')

    # Add condition legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                  markersize=10, label='Disease'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
                  markersize=10, label='Control')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    return fig, ax

def analyze_pathway_enrichment(de_results, significant_threshold=0.01):
    """
    Simulate pathway enrichment analysis
    (In real analysis, this would use databases like KEGG, Reactome, GO)
    """
    # Get significant genes
    sig_genes = de_results[de_results['adjusted_p_value'] < significant_threshold]

    # Simulate pathway categories (in practice, use gene set databases)
    pathways = {
        'Immune Response': ['Gene_1', 'Gene_3', 'Gene_5', 'Gene_7', 'Gene_9'],
        'Cell Cycle': ['Gene_0', 'Gene_2', 'Gene_4', 'Gene_6', 'Gene_8'],
        'Metabolism': ['Gene_10', 'Gene_15', 'Gene_20', 'Gene_25'],
        'Signal Transduction': ['Gene_11', 'Gene_13', 'Gene_16', 'Gene_18']
    }

    # Calculate enrichment (simplified Fisher's exact test simulation)
    enrichment_results = []

    total_sig = len(sig_genes)
    total_nonsig = len(de_results) - total_sig

    for pathway_name, pathway_genes in pathways.items():
        pathway_sig = len(set(sig_genes['gene']) & set(pathway_genes))
        pathway_total = len(pathway_genes)

        if pathway_sig > 0:
            # Simple overrepresentation test (hypergeometric-like)
            expected = (total_sig / len(de_results)) * pathway_total
            enrichment = pathway_sig / expected if expected > 0 else 1

            # Calculate p-value (simplified)
            p_value = 0.1 / enrichment  # Mock p-value

            enrichment_results.append({
                'pathway': pathway_name,
                'genes_in_pathway': pathway_total,
                'significant_genes': pathway_sig,
                'enrichment': enrichment,
                'p_value': min(p_value, 1.0)
            })

    enrichment_df = pd.DataFrame(enrichment_results)
    enrichment_df['log_p_value'] = -np.log10(enrichment_df['p_value'] + 1e-300)

    return enrichment_df

def main():
    """
    Main analysis pipeline
    """
    print("=== Multiomics Differential Expression Analysis ===\\n")

    # Load data
    print("1. Loading sample data...")
    expression_df, metadata = load_sample_data()
    print(f"   Loaded {expression_df.shape[0]} genes x {expression_df.shape[1]} samples")

    # Perform differential expression analysis
    print("\\n2. Performing differential expression analysis...")
    de_results = perform_differential_expression(expression_df, metadata)

    # Summary statistics
    n_significant = sum(de_results['significant'])
    print("   Results summary:")
    print(".2f")
    print(f"   - Significant genes (FDR < 0.05): {n_significant}")

    # Show top differentially expressed genes
    print("\\n3. Top differentially expressed genes:")
    top_genes = de_results.sort_values('adjusted_p_value').head(10)
    for _, gene in top_genes.iterrows():
        reg = "UP" if gene['fold_change'] > 1 else "DOWN"
        print(".3f")

    # Create visualizations
    print("\\n4. Creating visualizations...")

    # Volcano plot
    print("   Creating volcano plot...")
    vol_fig, vol_ax = create_volcano_plot(de_results,
                                         "Volcano Plot - Control vs Disease")
    plt.savefig('volcano_plot.png', dpi=300, bbox_inches='tight')
    print("   Saved volcano plot as 'volcano_plot.png'")

    # Heatmap
    print("   Creating expression heatmap...")
    heat_fig, heat_ax = create_heatmap(expression_df, metadata, top_n=20)
    plt.savefig('expression_heatmap.png', dpi=300, bbox_inches='tight')
    print("   Saved heatmap as 'expression_heatmap.png'")

    # Pathway enrichment
    print("\\n5. Analyzing pathway enrichment...")
    pathway_results = analyze_pathway_enrichment(de_results)
    print("   Top enriched pathways:")
    for _, pathway in pathway_results.sort_values('p_value').head(5).iterrows():
        print("+.3f")

    # Save results
    print("\\n6. Saving results...")
    de_results.to_csv('differential_expression_results.csv', index=False)
    pathway_results.to_csv('pathway_enrichment_results.csv', index=False)
    print("   Results saved to CSV files")

    print("\\n=== Analysis Complete ===")
    print("\\nKey findings:")
    print(".1f")
    print(f"- {len(de_results[de_results['fold_change'] > 2])} genes upregulated >2-fold")
    print(f"- {len(de_results[de_results['fold_change'] < 0.5])} genes downregulated >2-fold")
    print(f"- {len(pathway_results[pathway_results['log_p_value'] > 2])} pathways significantly enriched")

if __name__ == "__main__":
    main()
