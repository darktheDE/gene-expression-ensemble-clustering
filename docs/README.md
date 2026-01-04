# Gene Expression Clustering - Streamlit Demo App

**Phân cụm dữ liệu biểu hiện gen với Weighted SCENA-based Ensemble Learning**

---

## Giới thiệu

Ứng dụng Streamlit demo cho đồ án môn Machine Learning - HCMUTE. App trình bày kết quả nghiên cứu về phân cụm 801 samples từ 5 loại ung thư (BRCA, KIRC, COAD, LUAD, PRAD) sử dụng phương pháp Ensemble Learning.

### Nhóm thực hiện
**Nhóm 1:**
- Kiến Hưng
- Ngọc Thạch
- Hữu Huy

---

## Hướng dẫn chạy App

### Yêu cầu hệ thống
- Python 3.8+
- pip hoặc conda

### Cài đặt

1. **Clone repository**
```bash
git clone https://github.com/darktheDE/gene-expression-ensemble-clustering.git
cd gene-expression-ensemble-clustering
```

2. **Tạo virtual environment (khuyến nghị)**
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Cài đặt dependencies**
```bash
pip install -r requirements.txt
```

### Chạy App

```bash
streamlit run streamlit_app.py
```

App sẽ tự động mở trong browser tại: **http://localhost:8501**

---

## Chức năng App

### Dashboard
- Tổng quan về dự án và phương pháp
- Thông tin dataset và thách thức
- Sơ đồ workflow thuật toán
- Preview metrics của các models

### Dataset
- Thống kê dataset chi tiết
- Thông tin 5 loại ung thư
- Xem trước dữ liệu PCA
- **Ground Truth Visualization** (2D scatter plot)
- Biểu đồ phân phối classes

### Models Evaluation
- So sánh 3 mô hình cơ sở:
  - K-Means++
  - Hierarchical Clustering
  - DBSCAN
- Metrics cho từng model (ARI, NMI)
- Side-by-side comparison với Ground Truth
- Giải thích đặc điểm mỗi model

### Ensemble Results
- Cơ chế **Adaptive Weights** (pie chart)
- Kết quả cuối cùng của Ensemble
- So sánh tổng hợp tất cả models (bar chart)
- Visual comparison (4 models, 2x2 grid)
- Chi tiết kỹ thuật về SCENA

---

## Dữ liệu

### Dataset
- **Nguồn**: Gene Expression Cancer RNA-Seq
- **Samples**: 801 mẫu
- **Features gốc**: 20,531 genes
- **Sau PCA**: 30 components
- **Classes**: 5 loại ung thư

### Files trong `data/Processed/`
- `data_pca30.csv`: Dữ liệu sau PCA (30 chiều)
- `labels.csv`: Ground Truth labels
- `kmeans_labels.csv`: Kết quả K-Means++
- `hierarchical_manual_labels.csv`: Kết quả Hierarchical
- `dbscan_labels.csv`: Kết quả DBSCAN
- `ensemble_scena_labels.csv`: Kết quả Ensemble cuối cùng

---

## Cấu trúc dự án

```
gene-expression-ensemble-clustering/
├── streamlit_app.py              # Main app
├── requirements.txt              # Dependencies
├── .streamlit/
│   └── config.toml              # Streamlit config
├── pages/                        # 4 trang chính
│   ├── dashboard.py
│   ├── dataset.py
│   ├── models_evaluation.py
│   └── ensemble_results.py
├── utils/                        # Utilities
│   ├── data_loader.py           # Data loading
│   └── visualizations.py        # Plotly charts
├── data/Processed/               # CSV data files
├── assets/                       # Images (optional)
├── docs/
│   ├── GUIDE.MD                 # Development guide
│   └── DEPLOYMENT.md            # Deployment guide
└── code/                         # Jupyter notebooks
```

---

## Kết quả chính

### Performance Metrics (ARI - Adjusted Rand Index)

| Model | ARI | NMI | Stability |
|-------|-----|-----|-----------|
| K-Means++ | 0.9832 | 0.9750 | ⭐⭐ (Variable) |
| Hierarchical | 0.9907 | 0.9860 | ⭐⭐⭐ (Stable) |
| DBSCAN | 0.9577 | 0.9400 | ⭐⭐⭐ (Good) |
| **Ensemble** | **0.9907** | **0.9860** | **⭐⭐⭐ (Best)** |

### Ensemble Weights
- Hierarchical: **70%**
- DBSCAN: **20%**
- K-Means++: **10%**

---

## Features

### Visualization
- ✅ Interactive scatter plots (Plotly)
- ✅ Bar charts với annotations
- ✅ Pie charts cho weights
- ✅ Professional color palette
- ✅ Hover tooltips

### Performance
- ✅ Data caching với `@st.cache_data`
- ✅ Fast loading (< 3 seconds)
- ✅ Responsive layout
- ✅ Custom CSS styling

### UX
- ✅ Sidebar navigation
- ✅ Metrics cards
- ✅ Expandable technical details
- ✅ Professional theme

---

## Demo Online

App đã được deploy lên Streamlit Cloud:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://geneExEnCluG1.streamlit.app/)

---


## Dependencies

Các thư viện chính:
- `streamlit` - Web framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning
- `plotly` - Interactive visualization
- `matplotlib`, `seaborn` - Additional plotting

Xem đầy đủ trong [requirements.txt](requirements.txt)

---

## Contributing

Đây là đồ án học tập. Nếu có đóng góp hoặc phát hiện lỗi, vui lòng:
1. Fork repository
2. Tạo branch mới
3. Commit changes
4. Tạo Pull Request

---

## Contact

**HCMUTE - Machine Learning Course**

Nếu có câu hỏi, vui lòng liên hệ qua GitHub Issues.

---

## License

Dự án này được tạo cho mục đích học tập tại HCMUTE.

---

**Chúc bạn khám phá app thành công!**
