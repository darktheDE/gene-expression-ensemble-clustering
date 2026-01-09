# Phân Cụm Dữ Liệu Biểu Hiện Gen Sử Dụng Weighted SCENA Ensemble Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://geneExEnCluG1.streamlit.app/)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Đồ án cuối kỳ môn **Học Máy (Machine Learning)** - MALE431984_09  
**Trường Đại học Sư phạm Kỹ thuật TP.HCM (HCMUTE)**

---

## 📑 Mục Lục

1. [Giới thiệu Dự án](#1-giới-thiệu-dự-án)
2. [Thông tin Nhóm thực hiện](#2-thông-tin-nhóm-thực-hiện)
3. [Dữ liệu & Tiền xử lý](#3-dữ-liệu--tiền-xử-lý)
4. [Phương pháp Đề xuất: Weighted SCENA](#4-phương-pháp-đề-xuất-weighted-scena)
5. [Kết quả Thực nghiệm](#5-kết-quả-thực-nghiệm)
6. [Cấu trúc Dự án & Công nghệ](#6-cấu-trúc-dự-án--công-nghệ)
7. [Hướng dẫn Cài đặt & Sử dụng](#7-hướng-dẫn-cài-đặt--sử-dụng)
8. [Tài liệu Tham khảo](#8-tài-liệu-tham-khảo)

---

## 1. Giới thiệu Dự án

Phân tích dữ liệu biểu hiện gen (Gene Expression Data) là một bài toán quan trọng trong tin sinh học nhằm phát hiện các phân nhóm ung thư (cancer subtypes). Tuy nhiên, dữ liệu này thường có đặc điểm **số chiều rất lớn (high-dimensional)** nhưng **số lượng mẫu nhỏ (small sample size)**, gây khó khăn cho các thuật toán phân cụm truyền thống.

Dự án này đề xuất một kiến trúc **Ensemble Clustering** dựa trên thuật toán **SCENA (Spectral Clustering ENsemble Approach)** có cải tiến trọng số (Weighted), nhằm kết hợp ưu điểm của nhiều mô hình cơ sở để tạo ra kết quả phân cụm chính xác và ổn định hơn.

**Mục tiêu:** Phân loại 5 loại ung thư từ dữ liệu RNA-Seq.

---

## 2. Thông tin Nhóm thực hiện

| Vai trò | Họ và tên | MSSV | Nhiệm vụ chính |
|---------|-----------|------|----------------|
| **Thành viên** | Huỳnh Ngọc Thạch | 23133072 | Data Analysis, Modeling (K-Means++) |
| **Thành viên** | Huỳnh Hữu Huy | 23133027 | Modeling (Hierarchical), Visualization |
| **Thành viên** | Đỗ Kiến Hưng | 23133030 | Modeling (DBSCAN), Ensemble Logic |

**Giảng viên hướng dẫn:** TS. Phan Thị Huyền Trang

---

## 3. Dữ liệu & Tiền xử lý

### 3.1. Bộ dữ liệu (Dataset)
Sử dụng bộ dữ liệu **Gene Expression Cancer RNA-Seq** từ UCI Machine Learning Repository.
- **Số lượng mẫu:** 801 mẫu
- **Số chiều:** 20,531 gen
- **Số lớp (Classes):** 5 loại ung thư
  - **BRCA**: Breast Invasive Carcinoma
  - **KIRC**: Kidney Renal Clear Cell Carcinoma
  - **COAD**: Colon Adenocarcinoma
  - **LUAD**: Lung Adenocarcinoma
  - **PRAD**: Prostate Adenocarcinoma

### 3.2. Tiền xử lý (Preprocessing)
Do dữ liệu có số chiều quá lớn, quy trình xử lý bao gồm:
1. **Chuẩn hóa (Normalization):** Standard Scaling (Z-score) để đưa dữ liệu về cùng thang đo.
2. **Giảm chiều (Dimensionality Reduction):** Sử dụng **PCA (Principal Component Analysis)** để nén dữ liệu từ 20,531 chiều xuống **30 thành phần chính (components)**, giữ lại phần lớn thông tin quan trọng (variance).

---

## 4. Phương pháp Đề xuất: Weighted SCENA

Hệ thống sử dụng mô hình Ensemble với 3 thuật toán clustering cơ sở:

1.  **K-Means++**: Thuật toán phân cụm dựa trên centroid, nhanh và hiệu quả.
2.  **Hierarchical Clustering**: Phân cụm phân cấp (Ward linkage), nắm bắt tốt cấu trúc dữ liệu.
3.  **DBSCAN**: Phân cụm dựa trên mật độ, giúp xử lý nhiễu (noise/outliers).

### Kiến trúc Ensemble (Weighted SCENA)
Thay vì dùng SCENA thuần túy, nhóm đề xuất cơ chế **Adaptive Weighting**:

1.  **Tính trọng số ($w_i$):** Dựa trên **Silhouette Score** của từng mô hình cơ sở.
    $$w_i = \frac{S_i}{\sum S_j}$$
2.  **Xây dựng Ma trận Đồng thuận (Consensus Matrix):** Sử dụng CSPA (Cluster-based Similarity Partitioning Algorithm) có trọng số.
3.  **Tinh chỉnh (Refinement):** Sử dụng KNN để tăng cường độ chính xác cục bộ.
4.  **Phân cụm cuối cùng:** Áp dụng Spectral Clustering trên ma trận tương đồng đã tinh chỉnh.

---

## 5. Kết quả Thực nghiệm

Hệ thống đạt được hiệu suất rất cao, chứng minh tính hiệu quả của phương pháp Ensemble.

| Mô hình | ARI (Adjusted Rand Index) | NMI (Normalized Mutual Information) | Đánh giá |
|:--------|:-------------------------:|:-----------------------------------:|:---------|
| **K-Means++** | 0.9832 | 0.9750 | Tốt |
| **DBSCAN** | 0.9577 | 0.9400 | Khá (lọc được nhiễu) |
| **Hierarchical** | **0.9907** | **0.9860** | Xuất sắc |
| **Ensemble** | **0.9907** | **0.9860** | **Tối ưu & Ổn định** |

> **Nhận xét:** Kết quả Ensemble hội tụ về mức tốt nhất của mô hình thành phần (Hierarchical), đồng thời loại bỏ được sự bất ổn định của K-Means và tận dụng khả năng xử lý nhiễu của DBSCAN.

---

## 6. Cấu trúc Dự án & Công nghệ

### Công nghệ sử dụng
- **Ngôn ngữ:** Python 3.10+
- **Giao diện:** Streamlit
- **Machine Learning:** Scikit-learn
- **Trực quan hóa:** Plotly, Matplotlib
- **Xử lý dữ liệu:** Pandas, NumPy

### Cây thư mục
```bash
gene-expression-clustering/
├── streamlit_app.py        # 🏠 Main App Entry
├── pages/                  # 📄 Các trang chức năng
│   ├── demo.py             # Demo pipeline từng bước
│   ├── dataset.py          # Khám phá dữ liệu
│   ├── models_evaluation.py# Đánh giá Model đơn lẻ
│   └── ensemble_results.py # Kết quả Ensemble
├── utils/                  # 🛠️ Hàm tiện ích
│   ├── data_loader.py      # Load & Cache dữ liệu
│   └── visualizations.py   # Vẽ biểu đồ
├── data/                   # 💾 Dữ liệu
│   ├── Raw/                # Dữ liệu thô (sampledata.csv)
│   └── Processed/          # Dữ liệu PCA & Labels
├── code/                   # 📓 Jupyter Notebooks huấn luyện
└── assets/                 # 🖼️ Hình ảnh báo cáo
```

---

## 7. Hướng dẫn Cài đặt & Sử dụng

### 7.1. Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/darktheDE/gene-expression-ensemble-clustering.git
cd gene-expression-ensemble-clustering

# Tạo môi trường ảo (Khuyến nghị)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 7.2. Chạy ứng dụng

```bash
streamlit run streamlit_app.py
```
Sau đó truy cập trình duyệt tại địa chỉ: `http://localhost:8501`

### 7.3. Demo Mode
Trong ứng dụng, truy cập trang **Demo Pipeline Prediction** để trải nghiệm quá trình xử lý:
1. Chọn số lượng mẫu test.
2. Nhấn **"Chạy Demo"**.
3. Quan sát data đi qua các bước: Raw -> PCA -> Base Models -> Ensemble Voting.

---

## 8. Tài liệu Tham khảo

1.  **UCI Machine Learning Repository**: [Gene Expression Cancer RNA-Seq Dataset](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq)
2.  **Scikit-learn Documentation**: Clustering Algorithms.
3.  *Strehl, A., & Ghosh, J. (2002). Cluster ensembles---a knowledge reuse framework for combining multiple partitions.*

---
© 2026 HCMUTE - Nhóm 1. All Rights Reserved.
