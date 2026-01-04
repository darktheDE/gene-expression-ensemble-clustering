"""
Dataset Page - Data Exploration and Ground Truth Visualization
"""
import streamlit as st
import pandas as pd
from utils import (
    load_all_data,
    get_class_colors,
    get_dataset_info,
    plot_scatter_2d_with_colors,
    plot_class_distribution
)


def show():
    """Display the dataset exploration page"""
    
    st.header("Khám phá dữ liệu")
    
    # Load data
    with st.spinner("Loading data..."):
        X_pca, labels, metrics = load_all_data()
        dataset_info = get_dataset_info()
        color_map = get_class_colors()
    
    # Dataset Statistics
    st.markdown("## Thống kê Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tổng số mẫu", f"{dataset_info['total_samples']:,}")
    
    with col2:
        st.metric("Số chiều gốc", f"{dataset_info['original_features']:,} genes")
    
    with col3:
        st.metric("PCA Components", dataset_info['pca_components'])
    
    with col4:
        st.metric("Số lớp", dataset_info['n_classes'])
    
    st.markdown("---")
    
    # Class Information
    st.markdown("## Thông tin các lớp ung thư")
    
    class_info_data = []
    for class_code, class_name in dataset_info['class_names'].items():
        count = (labels['Ground Truth'] == class_code).sum()
        class_info_data.append({
            'Mã': class_code,
            'Tên bệnh': class_name,
            'Số mẫu': count,
            'Tỷ lệ (%)': f"{count/dataset_info['total_samples']*100:.2f}%"
        })
    
    df_class_info = pd.DataFrame(class_info_data)
    st.dataframe(df_class_info, width='stretch', hide_index=True)
    
    st.markdown("---")
    
    # Data Preview
    st.markdown("## Xem trước dữ liệu")
    
    # Combine PCA data with labels
    df_preview = X_pca.copy()
    df_preview['Class'] = labels['Ground Truth']
    
    st.markdown("**Dữ liệu PCA với nhãn Ground Truth** (5 dòng đầu):")
    st.dataframe(df_preview.head(), width='stretch')
    
    with st.expander("Xem thêm dữ liệu"):
        n_rows = st.slider("Số dòng hiển thị:", 5, 50, 10)
        st.dataframe(df_preview.head(n_rows), width='stretch')
    
    # Visualizations
    st.markdown("## Visualization")
    
    # Class Distribution
    st.markdown("### Phân phối các lớp")
    fig_dist = plot_class_distribution(labels['Ground Truth'])
    st.plotly_chart(fig_dist, width='stretch')
    
    st.markdown("---")
    
    # Ground Truth Scatter Plot
    st.markdown("### Ground Truth Visualization (2D PCA)")
    
    st.info("""
    Biểu đồ 2D sử dụng PC1 và PC2 từ PCA. Mỗi màu = 1 loại ung thư.  
    Có vùng chồng lấn giữa một số nhóm → bài toán phân cụm không dễ.
    """)
    
    fig_gt = plot_scatter_2d_with_colors(
        X_pca, 
        labels['Ground Truth'], 
        color_map,
        title="Ground Truth - Gene Expression Data (PCA 2D Projection)"
    )
    st.plotly_chart(fig_gt, width='stretch')
    
    st.markdown("---")
    
    with st.expander("About PCA (Principal Component Analysis)"):
        st.markdown("""
        **Tại sao cần PCA?** Dữ liệu gốc có **20,531 genes** → Curse of dimensionality.
        
        **Cách hoạt động:** Tìm các trục mới (principal components) bảo toàn variance tối đa. Giữ lại 30 components.
        
        **Lợi ích:** Giảm noise, tăng tốc training, dễ visualization, tránh overfitting.
        """)
    
    st.success("**Dataset đã sẵn sàng!** Chuyển sang trang Models Evaluation để xem kết quả.")


if __name__ == "__main__":
    show()
