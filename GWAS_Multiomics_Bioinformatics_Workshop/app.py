import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="GWAS & Multiomics Bioinformatics Workshop",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #FF6F00;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .sidebar-content {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Generate sample GWAS data
@st.cache_data
def generate_gwas_data():
    """Generate synthetic GWAS data for demonstration"""
    np.random.seed(42)

    # Create 22 chromosomes with varying SNP densities
    chromosomes = []
    positions = []
    p_values = []

    for chr_num in range(1, 23):
        # Vary SNP count per chromosome
        if chr_num == 1:
            n_snps = 8000
        elif chr_num in [2, 3]:
            n_snps = 6000
        elif chr_num <= 5:
            n_snps = 4000
        elif chr_num <= 10:
            n_snps = 2000
        else:
            n_snps = 1000

        # Generate positions along chromosome
        chr_positions = np.sort(np.random.uniform(0, 250e6, n_snps))

        # Generate p-values with some hits
        # Background uniform distribution
        base_pvals = np.random.uniform(0, 1, n_snps)

        # Add some significant associations
        n_hits = max(1, n_snps // 500)  # ~2 per 1000 SNPs
        hit_indices = np.random.choice(n_snps, n_hits, replace=False)
        hit_pvals = np.random.beta(0.5, 10, n_hits)  # Strong associations

        for idx, p_val in zip(hit_indices, hit_pvals):
            base_pvals[idx] = min(base_pvals[idx], p_val)

        chromosomes.extend([chr_num] * n_snps)
        positions.extend(chr_positions)
        p_values.extend(-np.log10(base_pvals))

    # Create DataFrame
    gwas_df = pd.DataFrame({
        'CHR': chromosomes,
        'BP': positions,
        'P': p_values,
        'SNP': [f'rs{100000 + i}' for i in range(len(chromosomes))]
    })

    return gwas_df

# Generate sample multiomics data
@st.cache_data
def generate_multiomics_data():
    """Generate synthetic multiomics data"""
    np.random.seed(123)
    n_samples = 100

    # Sample metadata
    samples = [f'Sample_{i}' for i in range(n_samples)]
    conditions = np.random.choice(['Control', 'Disease'], n_samples)

    # Genomics data (expression levels for 50 genes)
    genes = [f'Gene_{i}' for i in range(50)]
    gene_expression = np.random.normal(0, 1, (n_samples, 50))

    # Add condition effects to some genes
    for i in range(10):  # First 10 genes are differentially expressed
        effect_size = np.random.normal(1.5, 0.3)
        gene_expression[conditions == 'Disease', i] += effect_size

    # Proteomics data (30 proteins)
    proteins = [f'Protein_{i}' for i in range(30)]
    protein_abundance = np.random.normal(0, 1, (n_samples, 30))

    # Metabolomics data (20 metabolites)
    metabolites = [f'Metabolite_{i}' for i in range(20)]
    metabolite_levels = np.random.normal(0, 1, (n_samples, 20))

    return {
        'samples': samples,
        'conditions': conditions,
        'genes': genes,
        'expression': gene_expression,
        'proteins': proteins,
        'protein_data': protein_abundance,
        'metabolites': metabolites,
        'metabolite_data': metabolite_levels
    }

# Main application
def main():
    # Sidebar
    st.sidebar.title("🧬 GWAS & Multiomics Workshop")
    st.sidebar.markdown("---")

    # Navigation
    pages = {
        "🏠 Home": "home",
        "🧬 GWAS Fundamentals": "gwas_basics",
        "📊 GWAS Analysis": "gwas_analysis",
        "🔬 Multiomics Integration": "multiomics",
        "🧪 Bioinformatics Pipelines": "bioinformatics",
        "🛠️ Interactive Tools": "tools",
        "📚 Learning Resources": "resources"
    }

    choice = st.sidebar.radio("Navigate to:", list(pages.keys()))

    # Load data
    gwas_data = generate_gwas_data()
    multiomics_data = generate_multiomics_data()

    # Page content
    page_function = pages[choice]
    if page_function == "home":
        show_home_page(gwas_data, multiomics_data)
    elif page_function == "gwas_basics":
        show_gwas_basics_page(gwas_data)
    elif page_function == "gwas_analysis":
        show_gwas_analysis_page(gwas_data)
    elif page_function == "multiomics":
        show_multiomics_page(multiomics_data)
    elif page_function == "bioinformatics":
        show_bioinformatics_page()
    elif page_function == "tools":
        show_tools_page(gwas_data, multiomics_data)
    elif page_function == "resources":
        show_resources_page()

def show_home_page(gwas_data, multiomics_data):
    st.markdown('<h1 class="main-header">🎯 GWAS & Multiomics Bioinformatics Workshop</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Welcome to the Interactive Genomics Workshop!</h3>
        <p>This application provides hands-on experience with genome-wide association studies,
        multiomics data integration, and bioinformatics analysis. Explore GWAS results,
        analyze multiomics datasets, and learn about bioinformatics workflows.</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("GWAS SNPs", f"{len(gwas_data):,}", "~250k variants analyzed")

    with col2:
        sig_snps = len(gwas_data[gwas_data['P'] > 5])  # -log10(p) > 5
        st.metric("Significant SNPs", f"{sig_snps}", f"p < 1e-5")

    with col3:
        st.metric("Samples", f"{len(multiomics_data['samples'])}", "Multiomics dataset")

    with col4:
        n_chromosomes = len(gwas_data['CHR'].unique())
        st.metric("Chromosomes", f"{n_chromosomes}", "Genome-wide coverage")

    # GWAS Manhattan plot preview
    st.markdown('<h2 class="section-header">📊 GWAS Results Overview</h2>', unsafe_allow_html=True)

    fig = go.Figure()

    # Create Manhattan plot
    colors = ['#1f77b4', '#ff7f0e'] * 11  # Alternate colors for chromosomes

    for chr_num in range(1, 23):
        chr_data = gwas_data[gwas_data['CHR'] == chr_num]
        fig.add_trace(go.Scatter(
            x=chr_data['BP'],
            y=chr_data['P'],
            mode='markers',
            name=f'Chr {chr_num}',
            marker=dict(
                color=colors[chr_num-1],
                size=3,
                opacity=0.7
            ),
            showlegend=False
        ))

    fig.update_layout(
        title='GWAS Manhattan Plot Overview',
        xaxis_title='Genomic Position',
        yaxis_title='-log₁₀(p-value)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Workshop structure
    st.markdown('<h2 class="section-header">📚 Workshop Structure</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Session 1-2: GWAS Fundamentals
        - Genetic variation and inheritance
        - GWAS study design and methodology
        - Quality control and preprocessing
        - Understanding significance thresholds
        """)

        st.markdown("""
        ### Session 3-4: GWAS Analysis
        - Manhattan and QQ plots
        - Fine-mapping and annotation
        - Polygenic risk scores
        - Meta-analysis approaches
        """)

    with col2:
        st.markdown("""
        ### Session 5-6: Multiomics Integration
        - Multiomics data types and technologies
        - Data integration methods
        - Network-based approaches
        - Biological interpretation
        """)

        st.markdown("""
        ### Session 7-8: Bioinformatics Pipelines
        - NGS data analysis workflows
        - Variant calling and annotation
        - Functional analysis
        - Reproducible research
        """)

def show_gwas_basics_page(gwas_data):
    st.markdown('<h1 class="main-header">🧬 GWAS Fundamentals</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Understanding Genome-Wide Association Studies</h3>
        <p>GWAS examine genetic variation across the entire genome to identify
        associations between genetic variants and traits or diseases. This approach
        has revolutionized our understanding of the genetic basis of complex diseases.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 GWAS Concept Overview")

        # Interactive explanation
        concept = st.selectbox("Explore GWAS concepts:",
                              ["Genetic Variants", "Study Design", "Statistical Testing", "Significance Thresholds"])

        if concept == "Genetic Variants":
            st.markdown("""
            ### Genetic Variants
            GWAS typically examine single nucleotide polymorphisms (SNPs):
            - SNPs are the most common type of genetic variation
            - Each SNP represents a difference in a single DNA base
            - Millions of SNPs are genotyped across the genome
            - SNPs can be associated with traits through linkage disequilibrium
            """)

            # Show SNP distribution
            snp_counts = gwas_data.groupby('CHR').size()
            fig = px.bar(x=snp_counts.index, y=snp_counts.values,
                        labels={'x': 'Chromosome', 'y': 'Number of SNPs'},
                        title='SNP Distribution Across Chromosomes')
            st.plotly_chart(fig, use_container_width=True)

        elif concept == "Study Design":
            st.markdown("""
            ### GWAS Study Design
            Case-control studies are most common:
            - **Cases**: Individuals with the disease/trait
            - **Controls**: Individuals without the disease/trait
            - Genotypes are compared between groups
            - Association tested for each SNP independently
            - Large sample sizes needed (thousands to hundreds of thousands)
            """)

        elif concept == "Statistical Testing":
            st.markdown("""
            ### Statistical Testing
            Chi-square tests or logistic regression:
            - Tests if SNP alleles differ between cases/controls
            - Produces p-values for each SNP
            - Corrects for multiple testing using Bonferroni
            - Genome-wide significance: p < 5×10⁻⁸
            - Suggestive significance: p < 1×10⁻⁵
            """)

        else:  # Significance Thresholds
            st.markdown("""
            ### Significance Thresholds
            Multiple testing correction is crucial:
            - **Uncorrected p-value**: What you'd expect by chance
            - **Bonferroni correction**: p / number of tests
            - **Genome-wide significance**: p < 5×10⁻⁸ (very conservative)
            - **Suggestive associations**: p < 1×10⁻⁵ (follow-up needed)
            - Many variants fall in "gray zone" between these thresholds
            """)

            # Show significance thresholds
            thresholds = [-np.log10(0.05), -np.log10(1e-5), -np.log10(5e-8)]
            threshold_labels = ['Nominal (p<0.05)', 'Suggestive (p<1e-5)', 'Genome-wide (p<5e-8)']

            fig = go.Figure()
            fig.add_trace(go.Histogram(x=gwas_data['P'], nbinsx=50, name='SNP p-values'))

            for thresh, label in zip(thresholds, threshold_labels):
                fig.add_vline(x=thresh, line_dash="dash", line_color="red",
                             annotation_text=f"{label}: {thresh:.1f}")

            fig.update_layout(title='Distribution of GWAS p-values',
                             xaxis_title='-log₁₀(p-value)',
                             yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        ### Key GWAS Concepts:

        **🧬 SNPs**: Single nucleotide variations
        - Most common genetic variants
        - Used as markers for GWAS

        **📊 Case-Control Design**:
        - Compare affected vs unaffected
        - Test for association at each SNP

        **🎯 Significance**:
        - Genome-wide: p < 5×10⁻⁸
        - Suggestive: p < 1×10⁻⁵

        **🔗 Linkage Disequilibrium**:
        - SNPs in LD inherited together
        - One SNP can tag many variants

        **📈 Odds Ratio**:
        - Measure of association strength
        - OR > 1: risk allele increases disease
        """)

def show_gwas_analysis_page(gwas_data):
    st.markdown('<h1 class="main-header">📊 GWAS Data Analysis & Visualization</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Analyzing GWAS Results</h3>
        <p>GWAS produce millions of statistical tests. Effective visualization and
        interpretation are crucial for identifying true associations and avoiding false discoveries.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        viz_type = st.selectbox("Visualization type:",
                               ["Manhattan Plot", "QQ Plot", "Regional Plot", "SNP Statistics"])

        if viz_type == "Manhattan Plot":
            st.subheader("🗺️ Manhattan Plot")

            # Create Manhattan plot
            fig = go.Figure()

            # Color scheme for chromosomes
            colors = px.colors.qualitative.Set3

            for chr_num in range(1, 23):
                chr_data = gwas_data[gwas_data['CHR'] == chr_num]
                color_idx = (chr_num - 1) % len(colors)

                fig.add_trace(go.Scatter(
                    x=chr_data['BP'],
                    y=chr_data['P'],
                    mode='markers',
                    name=f'Chr {chr_num}',
                    marker=dict(
                        color=colors[color_idx],
                        size=4,
                        opacity=0.8
                    ),
                    hovertemplate='<b>Chr %{fullData.name}</b><br>' +
                                 'Position: %{x:,}<br>' +
                                 '-log₁₀(p): %{y:.2f}<br>' +
                                 '<extra></extra>'
                ))

            # Add significance lines
            fig.add_hline(y=-np.log10(5e-8), line_dash="solid", line_color="red",
                         annotation_text="Genome-wide significance (5e-8)")
            fig.add_hline(y=-np.log10(1e-5), line_dash="dash", line_color="orange",
                         annotation_text="Suggestive significance (1e-5)")

            fig.update_layout(
                title='GWAS Manhattan Plot',
                xaxis_title='Genomic Position',
                yaxis_title='-log₁₀(p-value)',
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "QQ Plot":
            st.subheader("📈 QQ Plot")

            # Create QQ plot
            observed = np.sort(gwas_data['P'])
            expected = np.random.uniform(0, 1, len(observed))
            expected = -np.log10(np.sort(expected))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=expected,
                y=observed,
                mode='markers',
                marker=dict(color='blue', size=3, opacity=0.7),
                name='Observed'
            ))

            # Add diagonal line
            max_val = max(max(observed), max(expected))
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(color='red', dash='solid'),
                name='Expected'
            ))

            fig.update_layout(
                title='QQ Plot of GWAS p-values',
                xaxis_title='Expected -log₁₀(p-values)',
                yaxis_title='Observed -log₁₀(p-values)',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Regional Plot":
            st.subheader("🎯 Regional Association Plot")

            # Pick a significant SNP for regional plot
            top_snp = gwas_data.loc[gwas_data['P'].idxmax()]

            # Create regional data (simulate around the top SNP)
            center_pos = top_snp['BP']
            region_start = center_pos - 500000  # 500kb region
            region_end = center_pos + 500000

            # Get SNPs in this region
            chr_num = top_snp['CHR']
            regional_data = gwas_data[
                (gwas_data['CHR'] == chr_num) &
                (gwas_data['BP'] >= region_start) &
                (gwas_data['BP'] <= region_end)
            ].copy()

            # Add recombination rates (simulated)
            regional_data['Recomb'] = np.random.uniform(0, 2, len(regional_data))

            fig = go.Figure()

            # P-values as main plot
            fig.add_trace(go.Scatter(
                x=regional_data['BP'] - center_pos,
                y=regional_data['P'],
                mode='markers',
                name='-log₁₀(p-value)',
                marker=dict(color='blue', size=5),
                yaxis='y1'
            ))

            # Recombination rate as secondary plot
            fig.add_trace(go.Scatter(
                x=regional_data['BP'] - center_pos,
                y=regional_data['Recomb'],
                mode='lines',
                name='Recomb. rate',
                line=dict(color='orange'),
                yaxis='y2'
            ))

            fig.update_layout(
                title=f'Regional Plot - Chr {chr_num}',
                xaxis_title='Position (bp from index SNP)',
                yaxis_title='-log₁₀(p-value)',
                yaxis2=dict(title='Recomb. rate (cM/Mb)', overlaying='y', side='right'),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "SNP Statistics":
            st.subheader("📊 SNP Statistics")

            # Basic statistics
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric("Total SNPs", f"{len(gwas_data):,}")
                st.metric("Chromosomes", f"{len(gwas_data['CHR'].unique())}")

            with col_b:
                sig_snps = len(gwas_data[gwas_data['P'] > 8])  # p < 1e-8
                st.metric("Genome-wide Significant", f"{sig_snps}")
                st.metric("Suggestive SNPs", f"{len(gwas_data[gwas_data['P'] > 5])}")

            with col_c:
                median_p = np.median(10 ** -gwas_data['P'])
                min_p = 10 ** -gwas_data['P'].max()
                st.metric("Median p-value", f"{median_p:.2e}")
                st.metric("Best p-value", f"{min_p:.2e}")

            # Distribution plot
            fig = px.histogram(gwas_data, x='P', nbins=50,
                             title='Distribution of -log₁₀(p-values)',
                             labels={'P': '-log₁₀(p-value)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        ### GWAS Analysis Tools:

        **🗺️ Manhattan Plot**:
        - SNPs plotted by genomic position
        - Height = -log₁₀(p-value)
        - Chromosomes in different colors

        **📈 QQ Plot**:
        - Compares observed vs expected p-values
        - Deviation from diagonal = inflation
        - Tests for systematic bias

        **🎯 Regional Plot**:
        - Detailed view around significant SNP
        - LD structure and recombination
        - Helps identify causal variants

        **📊 Statistics**:
        - Quality control metrics
        - Significance testing
        - Power calculations
        """)

def show_multiomics_page(multiomics_data):
    st.markdown('<h1 class="main-header">🔬 Multiomics Data Integration</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Integrating Multiple Omics Data Types</h3>
        <p>Multiomics approaches combine genomics, transcriptomics, proteomics,
        and metabolomics data to provide comprehensive biological insights. Integration
        methods help identify molecular mechanisms underlying complex diseases.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        analysis_type = st.selectbox("Analysis type:",
                                   ["Data Overview", "Differential Expression",
                                    "Principal Component Analysis", "Correlation Networks"])

        if analysis_type == "Data Overview":
            st.subheader("📊 Multiomics Data Overview")

            # Overview statistics
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric("Samples", f"{len(multiomics_data['samples'])}")
                control_count = sum(multiomics_data['conditions'] == 'Control')
                disease_count = sum(multiomics_data['conditions'] == 'Disease')
                st.metric("Control/Disease", f"{control_count}/{disease_count}")

            with col_b:
                st.metric("Genes", f"{len(multiomics_data['genes'])}")
                st.metric("Proteins", f"{len(multiomics_data['proteins'])}")

            with col_c:
                st.metric("Metabolites", f"{len(multiomics_data['metabolites'])}")
                st.metric("Total Features", "~100")

            # Expression heatmap preview
            n_show_genes = 20
            n_show_samples = 20

            expression_subset = multiomics_data['expression'][:n_show_samples, :n_show_genes]

            fig = go.Figure(data=go.Heatmap(
                z=expression_subset,
                x=[f'Gene_{i}' for i in range(n_show_genes)],
                y=[f'Sample_{i}' for i in range(n_show_samples)],
                colorscale='RdBu_r'
            ))

            fig.update_layout(
                title='Gene Expression Heatmap (subset)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "Differential Expression":
            st.subheader("🔍 Differential Expression Analysis")

            # Differential expression for genes
            control_expr = multiomics_data['expression'][multiomics_data['conditions'] == 'Control']
            disease_expr = multiomics_data['expression'][multiomics_data['conditions'] == 'Disease']

            # Calculate fold changes and p-values for first 20 genes
            fold_changes = []
            p_values = []
            gene_names = []

            for i in range(min(20, len(multiomics_data['genes']))):
                control_vals = control_expr[:, i]
                disease_vals = disease_expr[:, i]

                # Fold change
                fc = np.mean(disease_vals) / np.mean(control_vals)
                fold_changes.append(fc)

                # t-test
                t_stat, p_val = stats.ttest_ind(disease_vals, control_vals)
                p_values.append(-np.log10(p_val))
                gene_names.append(multiomics_data['genes'][i])

            # Volcano plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=np.log2(fold_changes),
                y=p_values,
                mode='markers',
                marker=dict(
                    color=p_values,
                    colorscale='Viridis',
                    size=8,
                    showscale=True,
                    colorbar=dict(title='-log₁₀(p-value)')
                ),
                text=gene_names,
                hovertemplate='<b>%{text}</b><br>' +
                             'Fold Change: %{x:.2f}<br>' +
                             '-log₁₀(p): %{y:.2f}<br>' +
                             '<extra></extra>'
            ))

            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red",
                         annotation_text="p = 0.05")

            fig.update_layout(
                title='Volcano Plot - Differential Gene Expression',
                xaxis_title='log₂(Fold Change)',
                yaxis_title='-log₁₀(p-value)',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "Principal Component Analysis":
            st.subheader("🎯 Principal Component Analysis")

            # PCA on gene expression data
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Standardize data
            scaler = StandardScaler()
            expr_scaled = scaler.fit_transform(multiomics_data['expression'])

            # PCA
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(expr_scaled)

            # Create plot
            pca_df = pd.DataFrame({
                'PC1': pca_result[:, 0],
                'PC2': pca_result[:, 1],
                'Condition': multiomics_data['conditions']
            })

            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Condition',
                           title='PCA of Gene Expression Data',
                           labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                                  'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'})

            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Explained variance
            st.subheader("Explained Variance")
            explained_var = pca.explained_variance_ratio_ * 100
            col1, col2, col3 = st.columns(3)
            col1.metric("PC1", f"{explained_var[0]:.1f}%")
            col2.metric("PC2", f"{explained_var[1]:.1f}%")
            col3.metric("PC3", f"{explained_var[2]:.1f}%")

        elif analysis_type == "Correlation Networks":
            st.subheader("🕸️ Correlation Networks")

            # Correlation analysis between different omics layers
            n_features = 10  # Show subset for visualization

            # Calculate correlations
            expr_data = multiomics_data['expression'][:, :n_features]
            protein_data = multiomics_data['protein_data'][:, :n_features]

            # Create correlation matrix between genes and proteins
            corr_matrix = np.corrcoef(expr_data.T, protein_data.T)[:n_features, n_features:]

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=[f'Protein_{i}' for i in range(n_features)],
                y=[f'Gene_{i}' for i in range(n_features)],
                colorscale='RdBu_r',
                zmid=0
            ))

            fig.update_layout(
                title='Gene-Protein Correlation Matrix',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        ### Multiomics Integration:

        **🔗 Data Types**:
        - Genomics: DNA sequence variation
        - Transcriptomics: Gene expression
        - Proteomics: Protein abundance
        - Metabolomics: Small molecule levels

        **🛠️ Integration Methods**:
        - **Correlation-based**: Pearson/Spearman
        - **Dimension reduction**: PCA, ICA
        - **Machine learning**: MOFA, DIABLO
        - **Network analysis**: WGCNA

        **🎯 Applications**:
        - Disease mechanism discovery
        - Biomarker identification
        - Drug target discovery
        - Precision medicine

        **📊 Challenges**:
        - Data heterogeneity
        - Batch effects
        - Missing data
        - Computational complexity
        """)

def show_bioinformatics_page():
    st.markdown('<h1 class="main-header">🧪 Bioinformatics Pipelines</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Building Robust Bioinformatics Workflows</h3>
        <p>Bioinformatics pipelines automate the analysis of large-scale biological data.
        Well-designed pipelines ensure reproducibility, scalability, and quality control
        in genomics and multiomics research.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        pipeline_type = st.selectbox("Explore pipeline type:",
                                   ["Variant Calling", "RNA-seq Analysis",
                                    "Multiomics Integration", "Functional Annotation"])

        if pipeline_type == "Variant Calling":
            st.subheader("🧬 Variant Calling Pipeline")

            st.markdown("""
            ### Standard Variant Calling Workflow:

            1. **Quality Control**
               - FastQC assessment
               - Adapter trimming
               - Quality filtering

            2. **Alignment**
               - Reference genome indexing
               - Read alignment (BWA, STAR)
               - BAM file generation

            3. **Variant Calling**
               - GATK HaplotypeCaller
               - FreeBayes or Samtools
               - VCF file generation

            4. **Quality Filtering**
               - Depth filtering
               - Quality score filtering
               - Population frequency filtering

            5. **Annotation**
               - Functional annotation (ANNOVAR, VEP)
               - Clinical significance (ClinVar)
               - Population databases (dbSNP, 1000 Genomes)
            """)

            # Pipeline visualization
            pipeline_steps = ['QC', 'Alignment', 'Calling', 'Filtering', 'Annotation']
            pipeline_times = np.array([10, 120, 180, 30, 60])  # minutes

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=pipeline_steps,
                y=pipeline_times,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                text=[f'{t} min' for t in pipeline_times],
                textposition='auto'
            ))

            fig.update_layout(
                title='Variant Calling Pipeline Timeline',
                xaxis_title='Pipeline Step',
                yaxis_title='Processing Time (minutes)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        elif pipeline_type == "RNA-seq Analysis":
            st.subheader("🧬 RNA-seq Analysis Pipeline")

            st.markdown("""
            ### RNA-seq Data Processing:

            1. **Quality Assessment**
               - MultiQC reports
               - RSeQC metrics
               - Library complexity

            2. **Read Alignment**
               - STAR or HISAT2 alignment
               - Transcriptome indexing
               - Gene/transcript quantification

            3. **Quantification**
               - HTSeq-count or featureCounts
               - Normalization (TPM, FPKM)
               - Quality control metrics

            4. **Differential Expression**
               - DESeq2 or edgeR analysis
               - Statistical testing
               - Fold change calculation

            5. **Functional Analysis**
               - GO enrichment analysis
               - Pathway analysis (KEGG, Reactome)
               - Gene set enrichment
            """)

            # Pipeline steps and tools
            rnaseq_steps = ['QC', 'Alignment', 'Quantification', 'DE Analysis', 'Functional Analysis']
            rnaseq_times = [15, 90, 45, 30, 60]  # minutes

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=rnaseq_steps,
                y=rnaseq_times,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                text=[f'{t} min' for t in rnaseq_times],
                textposition='auto'
            ))

            fig.update_layout(
                title='RNA-seq Pipeline Timeline',
                xaxis_title='Pipeline Step',
                yaxis_title='Processing Time (minutes)',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        elif pipeline_type == "Multiomics Integration":
            st.subheader("🔬 Multiomics Integration Pipeline")

            st.markdown("""
            ### Integrated Analysis Workflow:

            1. **Data Preprocessing**
               - Batch effect correction
               - Normalization across modalities
               - Feature selection and filtering

            2. **Single-Omics Analysis**
               - Differential analysis per layer
               - Quality control and outlier detection
               - Feature reduction (PCA, ICA)

            3. **Data Integration**
               - MOFA (Multi-Omics Factor Analysis)
               - Similarity network fusion
               - Canonical correlation analysis

            4. **Integrated Interpretation**
               - Multi-omics clustering
               - Network analysis
               - Pathway enrichment

            5. **Validation and Prediction**
               - Cross-validation
               - Predictive modeling
               - Biological validation
            """)

        elif pipeline_type == "Functional Annotation":
            st.subheader("🧬 Functional Annotation Pipeline")

            st.markdown("""
            ### Variant and Gene Annotation:

            1. **Genomic Context**
               - Coding vs non-coding regions
               - Exon/intron boundaries
               - Regulatory elements

            2. **Functional Impact**
               - Amino acid changes
               - Splicing effects
               - Regulatory disruption

            3. **Population Frequency**
               - Minor allele frequency (MAF)
               - Population-specific variants
               - Novel variants identification

            4. **Clinical Significance**
               - Known disease associations
               - Drug response variants
               - Clinical annotation databases

            5. **Integration and Reporting**
               - Combined annotation scores
               - Prioritization algorithms
               - Interpretation guidelines
            """)

    with col2:
        st.markdown("""
        ### Pipeline Best Practices:

        **📋 Quality Control**:
        - FastQC and MultiQC reports
        - Adapter and contaminant removal
        - Duplicate read handling

        **🧬 Tools & Software**:
        - **Alignment**: BWA, Bowtie2, STAR
        - **Calling**: GATK, FreeBayes
        - **Analysis**: SAMtools, BEDTools
        - **Workflow**: Snakemake, Nextflow

        **📊 Data Formats**:
        - FASTQ: Raw sequencing reads
        - BAM/SAM: Aligned reads
        - VCF: Variant calls
        - BED/GTF: Genomic annotations

        **🔄 Reproducibility**:
        - Version control (Git)
        - Environment management (Conda)
        - Documentation and logging
        """)

def show_tools_page(gwas_data, multiomics_data):
    st.markdown('<h1 class="main-header">🛠️ Interactive Analysis Tools</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Hands-on Data Exploration Tools</h3>
        <p>Use these interactive tools to gain practical experience with GWAS and
        multiomics data analysis techniques. Experiment with different parameters
        and observe how they affect results.</p>
    </div>
    """, unsafe_allow_html=True)

    tool_type = st.selectbox("Select analysis tool:",
                           ["GWAS Explorer", "Multiomics Browser", "Statistical Calculator", "Sample Size Calculator"])

    if tool_type == "GWAS Explorer":
        st.subheader("🧬 GWAS Interactive Explorer")

        col1, col2 = st.columns([1, 2])

        with col1:
            # Controls
            chr_filter = st.multiselect("Filter chromosomes:",
                                      list(range(1, 24)), default=list(range(1, 24)))
            p_threshold = st.slider("Display p-value threshold:",
                                  1e-10, 1.0, 1.0, format="%.1e")
            p_threshold_log = -np.log10(p_threshold)

            show_significance = st.checkbox("Show significance lines", value=True)

        with col2:
            # Filter data
            filtered_data = gwas_data[
                gwas_data['CHR'].isin(chr_filter) &
                (gwas_data['P'] >= p_threshold_log)
            ]

            # Create interactive plot
            fig = px.scatter(filtered_data, x='BP', y='P', color='CHR',
                           title='Interactive GWAS Explorer',
                           labels={'BP': 'Position (bp)', 'P': '-log₁₀(p-value)'},
                           hover_data=['SNP'])

            if show_significance:
                fig.add_hline(y=-np.log10(5e-8), line_dash="solid", line_color="red",
                             annotation_text="Genome-wide significance")
                fig.add_hline(y=-np.log10(1e-5), line_dash="dash", line_color="orange",
                             annotation_text="Suggestive significance")

            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.metric("Displayed SNPs", f"{len(filtered_data):,}")
            sig_snps = len(filtered_data[filtered_data['P'] > -np.log10(5e-8)])
            st.metric("Genome-wide significant", f"{sig_snps}")

    elif tool_type == "Multiomics Browser":
        st.subheader("🔬 Multiomics Data Browser")

        col1, col2 = st.columns([1, 2])

        with col1:
            # Controls
            data_type = st.selectbox("Data type:",
                                   ["Gene Expression", "Protein Abundance", "Metabolite Levels"])
            n_features = st.slider("Number of features to show:", 5, 50, 20)
            show_differential = st.checkbox("Highlight differential features", value=True)

        with col2:
            # Display selected data
            if data_type == "Gene Expression":
                data_matrix = multiomics_data['expression']
                features = multiomics_data['genes'][:n_features]
                title = "Gene Expression Matrix"

            elif data_type == "Protein Abundance":
                data_matrix = multiomics_data['protein_data']
                features = multiomics_data['proteins'][:n_features]
                title = "Protein Abundance Matrix"

            else:  # Metabolite Levels
                data_matrix = multiomics_data['metabolite_data']
                features = multiomics_data['metabolites'][:n_features]
                title = "Metabolite Levels Matrix"

            # Subset data
            plot_data = data_matrix[:30, :n_features]  # Max 30 samples for visualization

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=plot_data,
                x=features,
                y=[f'Sample_{i}' for i in range(plot_data.shape[0])],
                colorscale='RdBu_r',
                hoverongaps=False
            ))

            fig.update_layout(
                title=title,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show statistics
            st.subheader("Data Statistics")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Features", f"{n_features}")
            col_b.metric("Samples", f"{plot_data.shape[0]}")
            col_c.metric("Data range", f"{plot_data.min():.2f} to {plot_data.max():.2f}")

    elif tool_type == "Statistical Calculator":
        st.subheader("📊 Statistical Power Calculator")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### GWAS Power Analysis")

            # Input parameters
            maf = st.slider("Minor Allele Frequency:", 0.01, 0.5, 0.2)
            or_effect = st.slider("Odds Ratio:", 1.1, 3.0, 1.3)
            sample_size = st.slider("Sample Size (cases/controls):", 1000, 50000, 5000)
            alpha = st.slider("Significance Level:", 1e-8, 1e-5, 5e-8, format="%.1e")

            if st.button("Calculate Power"):
                # Simple power calculation approximation
                # This is a simplified calculation - real power calculations are more complex
                power = min(1.0, 0.8 * (maf * (or_effect - 1)) * np.sqrt(sample_size / 10000) * (1 / alpha))

                st.success(f"Estimated Power: {power:.1%}")

                # Show factors affecting power
                st.markdown("### Factors Affecting Power:")
                factors = {
                    "MAF": f"{maf:.2f} - Higher MAF increases power",
                    "Effect Size": f"{or_effect:.1f} - Larger effects easier to detect",
                    "Sample Size": f"{sample_size:,} - More samples increases power",
                    "Significance": f"{alpha:.1e} - Stricter threshold decreases power"
                }

                for factor, desc in factors.items():
                    st.markdown(f"**{factor}**: {desc}")

        with col2:
            # Power curves
            fig = go.Figure()

            # Sample size vs power for different effect sizes
            sample_sizes = np.logspace(3, 5, 50)
            for or_val in [1.2, 1.5, 2.0]:
                powers = [min(1.0, 0.8 * (maf * (or_val - 1)) * np.sqrt(s / 10000) * (1 / alpha))
                         for s in sample_sizes]
                fig.add_trace(go.Scatter(x=sample_sizes, y=powers, name=f'OR={or_val}'))

            fig.update_layout(
                title='Power vs Sample Size Curves',
                xaxis_title='Sample Size',
                yaxis_title='Statistical Power',
                xaxis_type="log",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    elif tool_type == "Sample Size Calculator":
        st.subheader("📏 GWAS Sample Size Calculator")

        col1, col2 = st.columns([1, 2])

        with col1:
            # Input parameters
            expected_maf = st.slider("Expected MAF:", 0.05, 0.5, 0.1)
            expected_or = st.slider("Minimum Detectable OR:", 1.1, 2.0, 1.2)
            desired_power = st.slider("Desired Power:", 0.6, 0.95, 0.8)
            alpha_level = st.slider("Alpha Level:", 1e-8, 1e-7, 5e-8, format="%.1e")

            if st.button("Calculate Sample Size"):
                # Simplified sample size calculation
                # In reality, this depends on many factors including linkage disequilibrium
                required_n = int((1.0 / (expected_maf * (expected_or - 1)**2)) *
                               (np.log(1 - desired_power) / np.log(alpha_level)) * 10000)

                st.success(f"Required sample size: {required_n:,} cases/controls")

                st.markdown("""
                ### Interpretation:
                This calculation provides a rough estimate. Real GWAS studies often require:
                - Hundreds of thousands of samples for common variants
                - Millions of samples for rare variants
                - Meta-analysis of multiple studies for validation
                """)

        with col2:
            # Sample size requirements visualization
            fig = go.Figure()

            or_values = np.linspace(1.1, 2.0, 20)
            sample_sizes = []
            for or_val in or_values:
                n = int((1.0 / (expected_maf * (or_val - 1)**2)) *
                       (np.log(1 - desired_power) / np.log(alpha_level)) * 10000)
                sample_sizes.append(min(n, 100000))  # Cap for visualization

            fig.add_trace(go.Scatter(x=or_values, y=sample_sizes, mode='lines+markers'))

            fig.update_layout(
                title='Required Sample Size vs Effect Size',
                xaxis_title='Odds Ratio',
                yaxis_title='Required Sample Size',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

def show_resources_page():
    st.markdown('<h1 class="main-header">📚 Learning Resources</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>GWAS and Multiomics Learning Resources</h3>
        <p>Expand your knowledge with these curated resources, tutorials, and
        references for GWAS, multiomics analysis, and bioinformatics.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📖 Books & Course Materials
        - **"An Introduction to Genetic Engineering"** - Desmond Nicholl
        - **"Human Molecular Genetics"** - Strachan & Read
        - **"Bioinformatics and Computational Biology in Drug Discovery"** - William Ogilvie
        - Coursera: Genomic Medicine specialization
        - edX: Introduction to Computational Biology

        ### 🌐 Online Resources
        - **1000 Genomes Project**: Population genetics data
        - **GTEx Portal**: eQTL and tissue-specific expression
        - **GWAS Catalog**: Published GWAS results
        - **OMIM**: Online Mendelian Inheritance in Man
        - **KEGG Pathway**: Molecular pathways database

        ### 🛠️ Software & Tools
        - **PLINK**: GWAS analysis toolkit
        - **GCTA**: Genetic correlation analysis
        - **MOFA**: Multi-omics factor analysis
        - **Seurat**: Single-cell RNA-seq analysis
        - **DESeq2**: Differential expression analysis
        """)

    with col2:
        st.markdown("""
        ### 🧬 Key Concepts & Methods
        - **Genetic Variance**: SNPs, indels, CNVs
        - **Population Structure**: PCA, ADMIXTURE
        - **LD Analysis**: Haploview, linkage blocks
        - **Multi-omics Integration**: Canonical correlation, joint PCA
        - **Functional Annotation**: Ensembl Variant Effect Predictor

        ### 🎯 Best Practices
        - **Data Quality**: Call rate, MAF, HWE filtering
        - **Population Stratification**: Genomic control
        - **Replication**: Independent cohort validation
        - **Meta-analysis**: Combine multiple studies
        - **Data Sharing**: Public repositories

        ### 🤝 Communities & Forums
        - **BioStars**: Q&A for bioinformatics
        - **ResearchGate**: Scientific networking
        - **Bioconductor Forum**: R package support
        - **GWAS Conference Series**: ISMB, ASHG
        - **OpenHelix**: Training and tutorials
        """)

    # External links
    st.markdown("---")
    st.subheader("🔗 External Resources")

    resources = {
        "🧬 GWAS Catalog": "https://www.ebi.ac.uk/gwas/",
        "💻 PLINK": "https://zzz.bwh.harvard.edu/plink/",
        "📊 1000 Genomes": "https://www.internationalgenome.org/",
        "🧫 GTEx Portal": "https://www.gtexportal.org/home/",
        "🧠 ENCODE Portal": "https://www.encodeproject.org/",
        "🧬 UCSC Genome Browser": "https://genome.ucsc.edu/",
        "💊 DrugBank": "https://go.drugbank.com/",
        "📈 GEO Database": "https://www.ncbi.nlm.nih.gov/geo/"
    }

    for name, url in resources.items():
        st.markdown(f"- [{name}]({url})")

if __name__ == "__main__":
    main()
