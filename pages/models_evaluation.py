"""
Models Evaluation Page - Individual Model Performance
"""
import streamlit as st
from utils import (
    load_all_data,
    get_class_colors,
    get_model_notes,
    plot_scatter_2d_with_colors
)


def show():
    """Display the models evaluation page"""
    
    st.header("Đánh giá các mô hình đơn lẻ")
    
    # Load data
    with st.spinner("Loading data..."):
        X_pca, labels, metrics = load_all_data()
        color_map = get_class_colors()
        model_notes = get_model_notes()
    
    st.markdown("""
    Trang này cho phép so sánh kết quả của **3 thuật toán phân cụm cơ sở** được sử dụng 
    trong Ensemble model. Chọn một mô hình để xem chi tiết.
    """)
    
    
    # Model selection
    st.markdown("## Chọn mô hình")
    
    model_options = ['K-Means++', 'Hierarchical', 'DBSCAN']
    selected_model = st.selectbox(
        "Chọn mô hình để phân tích:",
        model_options,
        index=1  # Default to Hierarchical (best single model)
    )
    
    
    # Display results for selected model
    st.markdown(f"## Kết quả: **{selected_model}**")
    
    # Create two columns: visualization (left) and metrics (right)
    col_viz, col_metrics = st.columns([2, 1])
    
    with col_viz:
        st.markdown("### Visualization")
        
        # Get predicted labels for selected model
        pred_labels = labels[selected_model]
        
        # Create scatter plot
        fig = plot_scatter_2d_with_colors(
            X_pca,
            pred_labels,
            color_map,
            title=f"{selected_model} Clustering Results"
        )
        st.plotly_chart(fig, width='stretch')
    
    with col_metrics:
        st.markdown("### Performance Metrics")
        
        # Get metrics for selected model
        model_metrics = metrics[selected_model]
        
        # Display metrics
        st.metric(
            label="Adjusted Rand Index (ARI)",
            value=f"{model_metrics['ARI']:.4f}"
        )
        
        st.metric(
            label="Normalized Mutual Information (NMI)",
            value=f"{model_metrics['NMI']:.4f}"
        )
        
        st.markdown("---")
        
        # Display model note
        st.info(model_notes[selected_model])
        
        st.markdown("---")
        
        # Model-specific information
        st.markdown("### Đặc điểm")
        
        if selected_model == 'K-Means++':
            st.markdown("""
            **K-Means++ Clustering:**
            - Khởi tạo centroid thông minh
            - Phù hợp với clusters hình cầu
            - **Ưu điểm:**
              - Nhanh, hiệu quả
              - Dễ implement
            - **Nhược điểm:**
              - Phụ thuộc random seed
              - Kết quả không ổn định
              - Cần biết số clusters trước
            """)
        
        elif selected_model == 'Hierarchical':
            st.markdown("""
            **Hierarchical Clustering:**
            - Linkage: Ward
            - Agglomerative (bottom-up)
            - **Ưu điểm:**
              - Kết quả ổn định
              - Không phụ thuộc random
              - Dendrogram trực quan
            - **Nhược điểm:**
              - Chậm với dataset lớn
              - Không scale tốt
            """)
        
        elif selected_model == 'DBSCAN':
            st.markdown("""
            **DBSCAN Clustering:**
            - Density-based clustering
            - Tự động phát hiện outliers
            - **Ưu điểm:**
              - Không cần biết số clusters
              - Lọc nhiễu tốt
              - Tìm clusters bất kỳ hình dạng
            - **Nhược điểm:**
              - Nhạy cảm với tham số
              - Khó với density không đều
            """)
    
    
    # Comparison with Ground Truth
    st.markdown("## So sánh với Ground Truth")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Ground Truth")
        fig_gt = plot_scatter_2d_with_colors(
            X_pca,
            labels['Ground Truth'],
            color_map,
            title="Ground Truth Labels"
        )
        st.plotly_chart(fig_gt, width='stretch')
    
    with col2:
        st.markdown(f"### {selected_model} Prediction")
        fig_pred = plot_scatter_2d_with_colors(
            X_pca,
            labels[selected_model],
            color_map,
            title=f"{selected_model} Predicted Labels"
        )
        st.plotly_chart(fig_pred, width='stretch')
    
    
    # All models comparison table
    st.markdown("## Bảng so sánh tất cả mô hình")
    
    import pandas as pd
    
    comparison_data = []
    for model in model_options:
        comparison_data.append({
            'Model': model,
            'ARI': f"{metrics[model]['ARI']:.4f}",
            'NMI': f"{metrics[model]['NMI']:.4f}",
            'Ranking': 'Best' if metrics[model]['ARI'] == max(metrics[m]['ARI'] for m in model_options) else ''
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, width='stretch', hide_index=True)
    
    st.success("**Hierarchical** cho kết quả tốt nhất (ARI 0.9907). Ensemble sẽ kết hợp điểm mạnh của cả 3 mô hình.")
    
    
    # Metrics explanation
    with st.expander("About ARI & NMI"):
        st.markdown("""
        **ARI (Adjusted Rand Index):** Đo độ tương đồng giữa clusters và ground truth. Giá trị [-1, 1], càng cao càng tốt.
        
        **NMI (Normalized Mutual Information):** Đo lượng thông tin chung. Giá trị [0, 1], càng cao càng tốt.
        """)


if __name__ == "__main__":
    show()
