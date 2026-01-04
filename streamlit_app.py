"""
Gene Expression Clustering with Weighted SCENA-based Ensemble
Main Streamlit App (Dashboard / Home Page)
"""
import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Gene Expression Clustering",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        padding-top: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 600;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    h3 {
        color: #34495e;
        font-weight: 500;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Dataframe */
    .dataframe {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - Team Info
st.sidebar.markdown("### Nhóm thực hiện")
st.sidebar.info("""
**Nhóm 1** - Học Máy (MALE431984_09)

Huỳnh Ngọc Thạch - 23133072  
Huỳnh Hữu Huy - 23133027  
Đỗ Kiến Hưng - 23133030

**GVHD:** TS. Phan Thị Huyền Trang  
**HCMUTE - 01/2026**
""")

# Title and introduction
st.title("Phân cụm Gene Expression")
st.markdown("**Ensemble Learning - Nhóm 1 - HCMUTE**")

st.markdown("---")

# ========== DASHBOARD CONTENT ==========
st.header("Tổng quan dự án")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Giới thiệu
    
    Dự án tập trung vào bài toán **phân cụm dữ liệu biểu hiện gen** từ 5 loại ung thư khác nhau 
    sử dụng phương pháp **Ensemble Learning** với cơ chế **Weighted SCENA**.
    
    ### Mục tiêu
    - Phân loại chính xác 5 nhóm bệnh ung thư: **BRCA, KIRC, COAD, LUAD, PRAD**
    - So sánh hiệu suất giữa các thuật toán phân cụm đơn lẻ
    - Xây dựng mô hình Ensemble tối ưu với adaptive weights
    
    ### Thách thức
    - **Dữ liệu nhiều chiều**: ~20,000 genes → Giảm chiều với PCA
    - **Nhiễu cao**: Gene expression data có variance lớn
    - **Phân bố không đồng đều**: Một số classes có ít samples
    """)

with col2:
    st.info("""
    ### Dataset
    
    **Số mẫu:** 801  
    **Số chiều gốc:** 20,531 genes  
    **Sau PCA:** 30 components  
    **Số classes:** 5
    
    ---
    
    ### Cancer Types
    
    - **BRCA**: Breast Cancer
    - **KIRC**: Kidney Cancer
    - **COAD**: Colon Cancer
    - **LUAD**: Lung Cancer
    - **PRAD**: Prostate Cancer
    """)


# Methodology
st.markdown("""
## Quy trình thực hiện
""")

# Workflow diagram
pipeline_path = Path(__file__).parent / 'assets' / 'pipeline.png'

if pipeline_path.exists():
    st.image(str(pipeline_path), caption="Sơ đồ khối thuật toán Weighted SCENA Ensemble", use_container_width=True)
else:
    # Show text-based workflow if image not available
    st.info("""
    ### Workflow (Sơ đồ khối thuật toán):
    
    1. **Raw Data** (801 samples × 20,531 genes)
    2. **↓ Preprocessing & PCA**
    3. **Reduced Data** (30 dimensions)
    4. **↓ Clustering with 3 Base Models**
    5. **K-Means++** | **Hierarchical** | **DBSCAN**
    6. **↓ Performance Evaluation**
    7. **Adaptive Weight Assignment**
    8. **↓ Weighted SCENA Ensemble**
    9. **Final Cluster Labels**
    
    *Note: Vui lòng thêm file `assets/pipeline.png` để hiển thị sơ đồ chi tiết*
    """)

st.markdown("---")

# Key Results Preview
st.markdown("## Kết quả chính")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="K-Means++ ARI",
        value="0.9832",
        delta="Variable"
    )

with col2:
    st.metric(
        label="Hierarchical ARI",
        value="0.9907",
        delta="Stable"
    )

with col3:
    st.metric(
        label="DBSCAN ARI",
        value="0.9577",
        delta="Good"
    )

with col4:
    st.metric(
        label="Ensemble ARI",
        value="0.9907",
        delta="Best",
        delta_color="normal"
    )

st.success("**Kết quả:** Ensemble = Hierarchical (ARI 0.9907), nhưng ổn định hơn nhờ kết hợp 3 mô hình.")


# Technical Details
with st.expander("Chi tiết kỹ thuật"):
    st.markdown("""
    ### Các thuật toán sử dụng
    
    1. **K-Means++**
       - Khởi tạo centroid thông minh
       - Tối ưu hóa random seed
       - ARI trung bình ~0.82, tối đa 0.9832
    
    2. **Hierarchical Clustering (Agglomerative)**
       - Linkage: Ward
       - Kết quả ổn định, ARI ~0.9907
       - Baseline tốt nhất trong các mô hình đơn lẻ
    
    3. **DBSCAN**
       - Phát hiện và lọc nhiễu hiệu quả
       - Tìm được các clusters không đều đặn
       - ARI ~0.9577
    
    4. **Weighted SCENA Ensemble**
       - Kết hợp CSPA (Cluster-based Similarity Partitioning Algorithm)
       - KNN Enhancement cho refined clustering
       - Adaptive weights dựa trên **Silhouette Score** của mỗi mô hình
       - Weights gần bằng nhau (~0.37) do Silhouette tương đương
    
    ### Metrics đánh giá
    
    - **ARI (Adjusted Rand Index)**: Đo độ tương đồng giữa clusters và ground truth
    - **NMI (Normalized Mutual Information)**: Đo lượng thông tin chung
    - **Silhouette Score**: Đo độ phân tách giữa các clusters (dùng để tính weights)
    """)


st.markdown("""
---
**Tài liệu & Liên kết**

[Dataset UCI](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq) | 
[GitHub](https://github.com/darktheDE/gene-expression-ensemble-clustering) | 
[Google Drive](https://drive.google.com/drive/folders/1WhS-cjF85jZPDEgFYu_WWi4oI9NKyYhl?usp=drive_link) | 
[Slide Canva](https://www.canva.com/design/DAG9cEMkrt8/RMMGM3-p3dCOLMkjmRZ9dQ/edit)

---
**Project Info**  
**Course**: Học Máy (MALE431984_09) | **GVHD**: TS. Phan Thị Huyền Trang | **HCMUTE - 01/2026**
""")
