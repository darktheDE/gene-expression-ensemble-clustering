"""
Gene Expression Clustering with Weighted SCENA-based Ensemble
Main Streamlit App
"""
import streamlit as st

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

# Title and introduction
st.title("Phân cụm Gene Expression")
st.markdown("**Ensemble Learning - Nhóm 1 - HCMUTE**")

st.markdown("---")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Chọn trang:",
    ["Dashboard", "Dataset", "Models Evaluation", "Ensemble Results"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Nhóm thực hiện")
st.sidebar.info("""
**Nhóm 1** - Học Máy (MALE431984_09)

Huỳnh Ngọc Thạch - 23133072  
Huỳnh Hữu Huy - 23133027  
Đỗ Kiến Hưng - 23133030

**GVHD:** TS. Phan Thị Huyền Trang  
**HCMUTE - 01/2026**
""")

# Page routing
if page == "Dashboard":
    # Import and run dashboard page
    import pages.dashboard as dashboard
    dashboard.show()
    
elif page == "Dataset":
    # Import and run dataset page
    import pages.dataset as dataset
    dataset.show()
    
elif page == "Models Evaluation":
    # Import and run models page
    import pages.models_evaluation as models
    models.show()
    
elif page == "Ensemble Results":
    # Import and run ensemble page
    import pages.ensemble_results as ensemble
    ensemble.show()
