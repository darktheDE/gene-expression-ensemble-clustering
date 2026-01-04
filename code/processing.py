
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '1'


def main():
    # Đường dẫn dữ liệu
    DATA_PATH = './'  # Thư mục hiện tại cho local, thay vì Google Drive
    
    # Load dữ liệu
    print("📁 Loading data...")
    X = pd.read_csv(DATA_PATH + 'data.csv', index_col=0)
    y = pd.read_csv(DATA_PATH + 'labels.csv', index_col=0, header=None, names=['class'])['class']
    y = y.reindex(X.index)
    
    print(f"✅ Data loaded successfully!")
    print(f"X Shape: {X.shape}")
    print(f"y Shape: {y.shape}")
    print(f"\nClass distribution:\n{y.value_counts()}")
    
    # Label Encoding
    print("\n🔢 Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes: {le.classes_}")
    
    # Standardization 
    print("\n📊 Standardizing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # # PCA
    print("\n🔄 Applying PCA with 30 components...")
    pca30 = PCA(n_components=30, random_state=2025)
    X_pca30 = pca30.fit_transform(X_scaled)  # Dùng X gốc như trong notebook
    
    print(f"PCA components: {X_pca30.shape[1]}")
    print(f"Explained variance ratio: {pca30.explained_variance_ratio_.sum():.4f}")
    
    # Tạo DataFrame với index từ X gốc
    print("\n💾 Creating data_pca30.csv...")
    pca_columns = [f'PC{i+1}' for i in range(30)]
    df_pca30 = pd.DataFrame(
        X_pca30,
        index=X.index,
        columns=pca_columns
    )
    
    # Lưu file
    output_path = DATA_PATH + 'data_pca30.csv'
    df_pca30.to_csv(output_path)
    
    print(f"✅ File saved: {output_path}")
    print(f"Shape: {df_pca30.shape}")
    print(f"\nFirst few rows:")
    print(df_pca30.head())
    
    # Lưu thêm labels để tiện sử dụng sau này
    labels_output_path = DATA_PATH + 'labels_encoded.csv'
    pd.DataFrame(
        {'class': y, 'class_encoded': y_encoded},
        index=X.index
    ).to_csv(labels_output_path)
    print(f"\n✅ Labels saved: {labels_output_path}")


if __name__ == "__main__":
    main()
