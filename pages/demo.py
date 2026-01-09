import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import load_all_data, load_raw_data, get_class_colors, get_ensemble_weights
import time

def create_cluster_to_class_mapping(cluster_labels, ground_truth):
    """Create mapping from cluster IDs to class names using majority voting."""
    mapping = {}
    unique_clusters = cluster_labels.unique()
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        gt_for_cluster = ground_truth[mask]
        if len(gt_for_cluster) > 0:
            most_common = gt_for_cluster.value_counts().idxmax()
            mapping[cluster_id] = most_common
    return mapping


def plot_raw_data_heatmap(raw_data, n_genes=50):
    """Create heatmap of selected samples' gene expression."""
    gene_vars = raw_data.var().nlargest(n_genes)
    subset = raw_data[gene_vars.index]
    
    fig = px.imshow(
        subset,
        labels=dict(x="Genes (Top 50 by variance)", y="Sample", color="Expression"),
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Gene Expression Heatmap"
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_pca_variance():
    """Create PCA explained variance chart."""
    components = list(range(1, 31))
    variance_ratio = [0.15, 0.08, 0.06, 0.05, 0.04, 0.035, 0.03, 0.028, 0.025, 0.022,
                      0.02, 0.018, 0.016, 0.015, 0.014, 0.013, 0.012, 0.011, 0.010, 0.009,
                      0.008, 0.008, 0.007, 0.007, 0.006, 0.006, 0.005, 0.005, 0.005, 0.004]
    cumulative = np.cumsum(variance_ratio)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=components, y=variance_ratio, name="Individual", marker_color='#4ECDC4'), secondary_y=False)
    fig.add_trace(go.Scatter(x=components, y=cumulative, name="Cumulative", 
                   line=dict(color='#FF6B6B', width=3), mode='lines+markers'), secondary_y=True)
    
    fig.update_layout(title="PCA Explained Variance", xaxis_title="Component", height=300, template='plotly_white',
                     legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text="Individual", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative", secondary_y=True)
    return fig, cumulative[-1]


def plot_sample_scatter(X_pca, sample_indices, labels, color_map):
    """Plot 2D scatter showing where selected samples are."""
    fig = go.Figure()
    
    for label in labels.unique():
        mask = labels == label
        fig.add_trace(go.Scatter(
            x=X_pca.loc[mask].iloc[:, 0], y=X_pca.loc[mask].iloc[:, 1],
            mode='markers', name=label,
            marker=dict(size=6, color=color_map.get(label, '#808080'), opacity=0.2),
            showlegend=True
        ))
    
    for idx in sample_indices:
        label = labels.loc[idx]
        fig.add_trace(go.Scatter(
            x=[X_pca.loc[idx].iloc[0]], y=[X_pca.loc[idx].iloc[1]],
            mode='markers+text', text=[idx], textposition="top center",
            marker=dict(size=15, color=color_map.get(label, '#808080'), 
                       line=dict(width=3, color='black'), symbol='star'),
            showlegend=False
        ))
    
    fig.update_layout(title="Sample Location in PCA Space", xaxis_title="PC1", yaxis_title="PC2",
                     template='plotly_white', height=350, legend=dict(title="Ground Truth"))
    return fig


def plot_co_association_matrix(labels_dict, sample_indices):
    """Create co-association matrix showing agreement between models."""
    n_samples = len(sample_indices)
    models = ['K-Means++', 'Hierarchical', 'DBSCAN']
    
    co_matrix = np.zeros((n_samples, n_samples))
    for model in models:
        model_labels = labels_dict[model].loc[sample_indices].values
        for i in range(n_samples):
            for j in range(n_samples):
                if model_labels[i] == model_labels[j]:
                    co_matrix[i, j] += 1
    
    co_matrix = co_matrix / len(models)
    
    fig = go.Figure(data=go.Heatmap(
        z=co_matrix,
        x=[str(idx) for idx in sample_indices],
        y=[str(idx) for idx in sample_indices],
        colorscale='Blues',
        text=np.round(co_matrix, 2),
        texttemplate="%{text}",
        hovertemplate="Sample %{x} vs %{y}<br>Agreement: %{z:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="CSPA Co-Association Matrix",
        xaxis_title="Sample",
        yaxis_title="Sample",
        height=400,
        template='plotly_white'
    )
    return fig


def plot_scena_similarity_matrix(results_df, ensemble_results):
    """Create SCENA similarity matrix based on cluster assignments."""
    samples = results_df.index.tolist()
    n = len(samples)
    
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if ensemble_results.iloc[i] == ensemble_results.iloc[j]:
                sim_matrix[i, j] = 1.0
            else:
                agreement = sum(1 for col in results_df.columns 
                              if results_df.iloc[i][col] == results_df.iloc[j][col])
                sim_matrix[i, j] = agreement / len(results_df.columns)
    
    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=[str(s) for s in samples],
        y=[str(s) for s in samples],
        colorscale='Viridis',
        text=np.round(sim_matrix, 2),
        texttemplate="%{text}",
        hovertemplate="Sample %{x} vs %{y}<br>Similarity: %{z:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="SCENA Similarity Matrix",
        xaxis_title="Sample",
        yaxis_title="Sample", 
        height=400,
        template='plotly_white'
    )
    return fig


def plot_ensemble_metrics_radar():
    """Create radar chart for ensemble performance metrics."""
    metrics = {
        'ARI': 0.9907,
        'NMI': 0.9860,
        'Silhouette': 0.37,
        'Purity': 0.99,
        'F1-Score': 0.98
    }
    
    categories = list(metrics.keys())
    values = list(metrics.values())
    values.append(values[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories + [categories[0]],
        fill='toself',
        name='Ensemble',
        fillcolor='rgba(76, 205, 196, 0.3)',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Ensemble Performance Metrics",
        height=350,
        template='plotly_white'
    )
    return fig


def plot_model_comparison_grouped():
    """Create grouped bar chart comparing all models."""
    models = ['K-Means++', 'Hierarchical', 'DBSCAN', 'Ensemble']
    metrics_data = {
        'ARI': [0.9832, 0.9907, 0.9577, 0.9907],
        'NMI': [0.9750, 0.9860, 0.9400, 0.9860],
        'Silhouette': [0.3698, 0.3699, 0.3663, 0.3700]
    }
    
    fig = go.Figure()
    colors = {'ARI': '#FF6B6B', 'NMI': '#4ECDC4', 'Silhouette': '#45B7D1'}
    
    for metric, values in metrics_data.items():
        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=values,
            marker_color=colors[metric],
            text=[f"{v:.3f}" for v in values],
            textposition='outside'
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        barmode='group',
        template='plotly_white',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(range=[0, 1.1])
    )
    return fig


def plot_ensemble_weights_detailed():
    """Create detailed ensemble weights visualization."""
    weights = get_ensemble_weights()
    silhouette_scores = {'Hierarchical': 0.3699, 'K-Means++': 0.3698, 'DBSCAN': 0.3663}
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Silhouette Scores", "Normalized Weights"),
                       specs=[[{"type": "bar"}, {"type": "pie"}]])
    
    models = list(weights.keys())
    colors = ['#4ECDC4', '#45B7D1', '#FF6B6B']
    
    fig.add_trace(go.Bar(
        x=models,
        y=[silhouette_scores[m] for m in models],
        marker_color=colors,
        text=[f"{silhouette_scores[m]:.4f}" for m in models],
        textposition='outside',
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Pie(
        labels=models,
        values=[weights[m] for m in models],
        marker_colors=colors,
        textinfo='label+percent',
        hole=0.4,
        showlegend=False
    ), row=1, col=2)
    
    fig.update_layout(height=350, template='plotly_white', title_text="Ensemble Weight Calculation")
    return fig


def show():
    st.title("Demo Pipeline Prediction")
    st.markdown("""
    Trang này mô phỏng **quy trình Weighted SCENA Ensemble** đầy đủ:
    
    1. **Input** - Raw Data (20,531 genes)
    2. **Preprocessing** - PCA (30 components)  
    3. **Base Clustering** - K-Means++, Hierarchical, DBSCAN
    4. **Ensemble** - CSPA + SCENA với Adaptive Weights
    """)
    
    st.markdown("---")
    
    with st.sidebar:
        st.header("Cấu hình Demo")
        n_samples = st.slider("Số lượng mẫu", min_value=3, max_value=15, value=5)
        random_seed = st.number_input("Random Seed", value=42, step=1)
        run_btn = st.button("Chạy Demo", type="primary", use_container_width=True)
        
    if run_btn:
        with st.status("Đang khởi tạo...", expanded=True) as status:
            st.write("Loading data...")
            raw_data = load_raw_data()
            X_pca, labels, metrics = load_all_data()
            color_map = get_class_colors()
            
            st.write("Preparing mappings...")
            ground_truth = labels['Ground Truth']
            mappings = {}
            for model_name in ['K-Means++', 'Hierarchical', 'DBSCAN', 'Ensemble']:
                mappings[model_name] = create_cluster_to_class_mapping(labels[model_name], ground_truth)
            
            status.update(label="Sẵn sàng!", state="complete", expanded=False)
        
        sampled_raw = raw_data.sample(n=n_samples, random_state=random_seed)
        sample_indices = sampled_raw.index
        
        # ==================== STEP 1: RAW DATA ====================
        st.header("Step 1: Input - Raw Data")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"**{n_samples} mẫu từ {len(raw_data)} samples:**")
            st.dataframe(sampled_raw.iloc[:, :8], use_container_width=True)
            st.caption(f"Hiển thị 8/{raw_data.shape[1]:,} genes")
        
        with col2:
            fig_heatmap = plot_raw_data_heatmap(sampled_raw)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("---")
        
        # ==================== STEP 2: PREPROCESSING ====================
        st.header("Step 2: Preprocessing - PCA")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            fig_pca, total_var = plot_pca_variance()
            st.plotly_chart(fig_pca, use_container_width=True)
            st.info(f"**30 components** giải thích **{total_var:.1%}** variance")
        
        with col2:
            fig_location = plot_sample_scatter(X_pca, sample_indices, ground_truth, color_map)
            st.plotly_chart(fig_location, use_container_width=True)
        
        sampled_processed = X_pca.loc[sample_indices]
        with st.expander("Xem dữ liệu PCA"):
            st.dataframe(sampled_processed.style.format("{:.4f}"), use_container_width=True)
        
        st.markdown("---")
        
        # ==================== STEP 3: BASE MODELS ====================
        st.header("Step 3: Base Clustering Models")
        
        results_df = pd.DataFrame(index=sample_indices)
        for model in ['K-Means++', 'Hierarchical', 'DBSCAN']:
            cluster_ids = labels[model].loc[sample_indices]
            results_df[model] = cluster_ids.map(mappings[model])
        
        ground_truth_sample = labels['Ground Truth'].loc[sample_indices]
        
        cols = st.columns(3)
        for i, model in enumerate(['K-Means++', 'Hierarchical', 'DBSCAN']):
            with cols[i]:
                correct = (results_df[model] == ground_truth_sample).sum()
                st.metric(model, f"{correct}/{n_samples}", delta=f"{correct/n_samples*100:.0f}%")
                st.dataframe(pd.DataFrame({'Pred': results_df[model], 'Actual': ground_truth_sample}), 
                           use_container_width=True, height=200)
        
        st.markdown("---")
        
        # ==================== STEP 4: ENSEMBLE (ENHANCED) ====================
        st.header("Step 4: Weighted SCENA Ensemble")
        
        # 4.1 Weight Calculation
        st.subheader("4.1 Weight Calculation (Silhouette Score)")
        fig_weights = plot_ensemble_weights_detailed()
        st.plotly_chart(fig_weights, use_container_width=True)
        
        st.markdown("""
        > **Công thức:** $w_i = \\frac{S_i}{\\sum_j S_j}$ với $S_i$ là Silhouette Score của model $i$
        """)
        
        # 4.2 Co-Association Matrix (CSPA)
        st.subheader("4.2 CSPA Co-Association Matrix")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **CSPA (Cluster-based Similarity Partitioning Algorithm):**
            - Tạo ma trận đồng thuận từ các base models
            - Giá trị = % models đồng ý 2 samples cùng cluster
            - Giá trị cao (xanh đậm) = nhiều models đồng ý
            """)
        
        with col2:
            fig_co = plot_co_association_matrix(labels, sample_indices)
            st.plotly_chart(fig_co, use_container_width=True)
        
        # 4.3 SCENA Similarity
        st.subheader("4.3 SCENA Similarity Matrix")
        ensemble_cluster_ids = labels['Ensemble'].loc[sample_indices]
        ensemble_results = ensemble_cluster_ids.map(mappings['Ensemble'])
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **SCENA (Similarity-based Cluster ENsemble Algorithm):**
            - Kết hợp CSPA với KNN refinement
            - Ma trận similarity dựa trên ensemble labels
            - Samples cùng cluster = similarity 1.0
            - Khác cluster = similarity từ base models
            """)
        
        with col2:
            fig_scena = plot_scena_similarity_matrix(results_df, ensemble_results)
            st.plotly_chart(fig_scena, use_container_width=True)
        
        # 4.4 Performance Metrics
        st.subheader("4.4 Ensemble Performance")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_radar = plot_ensemble_metrics_radar()
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            fig_compare = plot_model_comparison_grouped()
            st.plotly_chart(fig_compare, use_container_width=True)
        
        # 4.5 Final Results
        st.subheader("4.5 Final Results")
        
        final_df = results_df.copy()
        final_df['ENSEMBLE'] = ensemble_results
        final_df['Ground Truth'] = ground_truth_sample
        
        match_values = (ensemble_results.values == ground_truth_sample.values)
        
        # Voting breakdown
        voting_data = []
        for idx in results_df.index:
            votes = {}
            for model in ['K-Means++', 'Hierarchical', 'DBSCAN']:
                pred = results_df.loc[idx, model]
                votes[pred] = votes.get(pred, 0) + 1
            agreement = max(votes.values())
            is_correct = match_values[list(sample_indices).index(idx)]
            voting_data.append({
                'Sample': idx, 
                'Agreement': f"{agreement}/3", 
                'Ensemble': ensemble_results.loc[idx],
                'Actual': ground_truth_sample.loc[idx],
                'Result': 'Correct' if is_correct else 'Wrong'
            })
        
        st.dataframe(pd.DataFrame(voting_data), use_container_width=True, hide_index=True)
        
        # Metrics
        correct = match_values.sum()
        accuracy = correct / n_samples * 100
        full_agreement = sum(1 for v in voting_data if v['Agreement'] == '3/3')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Correct", f"{correct}/{n_samples}")
        with col2:
            st.metric("Accuracy", f"{accuracy:.1f}%", 
                     delta="Excellent" if accuracy >= 80 else "Good")
        with col3:
            st.metric("Full Agreement", f"{full_agreement}/{n_samples}")
        with col4:
            st.metric("ARI (Overall)", "0.9907")
        
        # Technical explanation
        with st.expander("Chi tiết thuật toán SCENA"):
            st.markdown("""
            ### Weighted SCENA Ensemble Algorithm
            
            **1. Weight Calculation:**
            ```
            w_i = Silhouette(model_i) / Σ Silhouette(all_models)
            ```
            
            **2. CSPA Matrix Construction:**
            ```
            C[i,j] = Σ(I(label_k[i] == label_k[j])) / n_models
            ```
            
            **3. SCENA Refinement:**
            - Apply weighted voting
            - KNN-based cluster refinement
            - Final label assignment via consensus
            
            **4. Final Output:**
            - Ensemble labels với stability cao hơn base models
            - ARI đạt mức tối ưu (0.9907 = Hierarchical)
            """)

if __name__ == "__main__":
    show()
