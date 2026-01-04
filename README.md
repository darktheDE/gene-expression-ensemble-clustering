# Phân cụm dữ liệu biểu hiện gen với Ensemble Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://geneExEnCluG1.streamlit.app/)

Đồ án cuối kỳ môn **Học Máy (MALE431984_09)** - Trường Đại học Sư phạm Kỹ thuật TP.HCM (HCMUTE)

## Thông tin dự án

| Thông tin | Chi tiết |
|-----------|----------|
| **Đề tài** | Phân cụm dữ liệu biểu hiện gen với Ensemble Learning |
| **Phương pháp** | Weighted SCENA-based Approach |
| **GVHD** | TS. Phan Thị Huyền Trang |
| **Nhóm** | Nhóm 1 |

## Thành viên nhóm

| STT | Họ và tên | MSSV |
|-----|-----------|------|
| 1 | Huỳnh Ngọc Thạch | 23133072 |
| 2 | Huỳnh Hữu Huy | 23133027 |
| 3 | Đỗ Kiến Hưng | 23133030 |

## Tài liệu & Liên kết

| Tài liệu | Liên kết |
|----------|----------|
| **Dataset (UCI)** | [Gene Expression Cancer RNA-Seq](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq) |
| **Demo App** | [geneExEnCluG1.streamlit.app](https://geneExEnCluG1.streamlit.app/) |
| **GitHub Repo** | [github.com/darktheDE/gene-expression-ensemble-clustering](https://github.com/darktheDE/gene-expression-ensemble-clustering) |
| **Google Drive** | [Process, Colab, Báo cáo](https://drive.google.com/drive/folders/1WhS-cjF85jZPDEgFYu_WWi4oI9NKyYhl?usp=drive_link) |
| **Slide trình bày** | [Canva Presentation](https://www.canva.com/design/DAG9cEMkrt8/RMMGM3-p3dCOLMkjmRZ9dQ/edit) |

Dự án sử dụng phương pháp **Weighted SCENA-based Ensemble Learning** để phân cụm dữ liệu biểu hiện gen từ 5 loại ung thư:
- BRCA (Breast Cancer)
- KIRC (Kidney Cancer)
- COAD (Colon Cancer)
- LUAD (Lung Cancer)
- PRAD (Prostate Cancer)

### Kết quả chính

- **Dataset**: 801 mẫu, 20,531 genes → PCA 30 components
- **ARI đạt được**: 0.9907 (Ensemble = Hierarchical)
- **NMI đạt được**: 0.9860

## Cài đặt

```bash
# Clone repo
git clone https://github.com/your-username/gene-expression-ensemble-clustering.git
cd gene-expression-ensemble-clustering

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run streamlit_app.py
```

## Hướng dẫn lấy dữ liệu Raw

1. Truy cập [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq).
2. Tải bộ dữ liệu (file `.zip`).
3. Giải nén và lưu 2 file `data.csv` và `labels.csv` vào thư mục `data/Raw/`.

## Cấu trúc thư mục

```
├── streamlit_app.py      # Main app (Dashboard)
├── pages/                # Các trang Streamlit
│   ├── dataset.py
│   ├── models_evaluation.py
│   └── ensemble_results.py
├── utils/                # Utility functions
│   ├── data_loader.py
│   └── visualizations.py
├── data/                 # Dữ liệu
│   ├── Raw/              # Chứa data.csv và labels.csv gốc
│   └── Processed/        # Dữ liệu đã xử lý và kết quả PCA
├── code/                 # Notebooks phân tích
└── assets/               # Hình ảnh và tài nguyên
```

## Công nghệ sử dụng

- Python 3.10+
- Streamlit
- Scikit-learn
- Plotly
- Pandas, NumPy

## License

MIT License - HCMUTE 2026
