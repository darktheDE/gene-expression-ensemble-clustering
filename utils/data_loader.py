"""
Data Loading Module for Gene Expression Clustering App
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


@st.cache_data
def load_all_data():
    """
    Load all data files with caching for performance.
    
    Returns:
        tuple: (X_pca, labels_dict, metrics_dict)
    """
    # Base paths
    base_dir = Path(__file__).parent.parent / 'data'
    processed_path = base_dir / 'Processed'
    raw_path = base_dir / 'Raw'
    result_path = base_dir / 'result'
    
    # Load PCA features (Processed)
    X_pca = pd.read_csv(processed_path / 'data_pca30.csv', index_col=0)
    
    # Load all labels
    labels = {}
    # Ground Truth is now in Raw/labels.csv
    labels['Ground Truth'] = pd.read_csv(raw_path / 'labels.csv', index_col=0)['Class']
    
    # Clustering labels are now in result/
    labels['K-Means++'] = pd.read_csv(result_path / 'kmeans_plusplus_labels.csv', index_col=0).iloc[:, 0]
    labels['Hierarchical'] = pd.read_csv(result_path / 'hierarchical_manual_labels.csv', index_col=0).iloc[:, 0]
    labels['DBSCAN'] = pd.read_csv(result_path / 'dbscan_labels.csv', index_col=0).iloc[:, 0]
    labels['Ensemble'] = pd.read_csv(result_path / 'ensemble_scena_labels.csv', index_col=0).iloc[:, 0]
    
    # Hard-coded metrics (pre-computed)
    metrics = {
        'K-Means++': {'ARI': 0.9832, 'NMI': 0.9750},
        'Hierarchical': {'ARI': 0.9907, 'NMI': 0.9860},
        'DBSCAN': {'ARI': 0.9577, 'NMI': 0.9400},
        'Ensemble': {'ARI': 0.9907, 'NMI': 0.9860}
    }
    
    return X_pca, labels, metrics


@st.cache_data
def load_raw_data():
    """
    Load raw gene expression data.
    
    Returns:
        pd.DataFrame: Raw data
    """
    # Base path
    data_path = Path(__file__).parent.parent / 'data' / 'Raw'
    
    # Load raw data
    raw_data = pd.read_csv(data_path / 'sampledata.csv', index_col=0)
    
    return raw_data


def get_class_colors():
    """
    Get consistent color mapping for cancer classes.
    
    Returns:
        dict: Color mapping for each cancer type
    """
    return {
        'BRCA': '#FF6B6B',    # Breast Cancer - Red
        'KIRC': '#4ECDC4',    # Kidney Cancer - Teal
        'COAD': '#45B7D1',    # Colon Cancer - Blue
        'LUAD': '#FFA07A',    # Lung Cancer - Light Salmon
        'PRAD': '#98D8C8'     # Prostate Cancer - Mint
    }


def get_dataset_info():
    """
    Get dataset statistics and information.
    
    Returns:
        dict: Dataset information
    """
    return {
        'total_samples': 801,
        'original_features': 20531,
        'pca_components': 30,
        'n_classes': 5,
        'classes': ['BRCA', 'KIRC', 'COAD', 'LUAD', 'PRAD'],
        'class_names': {
            'BRCA': 'Breast invasive carcinoma',
            'KIRC': 'Kidney renal clear cell carcinoma',
            'COAD': 'Colon adenocarcinoma',
            'LUAD': 'Lung adenocarcinoma',
            'PRAD': 'Prostate adenocarcinoma'
        }
    }


def get_model_notes():
    """
    Get notes and comments for each model.
    
    Returns:
        dict: Model notes
    """
    return {
        'K-Means++': 'Đạt được khi tinh chỉnh random seed tối ưu. Trung bình ~0.82',
        'Hierarchical': 'Baseline tốt nhất, kết quả ổn định',
        'DBSCAN': 'Lọc nhiễu tốt, phát hiện outliers hiệu quả'
    }


def get_ensemble_weights():
    """
    Get ensemble model weights based on Silhouette Score.
    
    Weights are calculated from each model's Silhouette Score, which
    measures how well-defined the clusters are.
    
    Returns:
        dict: Weights for each base model (Silhouette scores)
    """
    return {
        'Hierarchical': 0.3699,
        'K-Means++': 0.3698,
        'DBSCAN': 0.3663
    }
