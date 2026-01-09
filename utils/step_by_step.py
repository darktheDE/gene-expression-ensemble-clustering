
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
import streamlit as st
from utils.data_loader import load_all_data

class StepByStepDemo:
    def __init__(self):
        """
        Initialize the Step-by-Step Demo class.
        Loads the PCA data and pre-computed labels.
        Computes global centroids for K-Means explanation.
        """
        self.data, labels_dict, _ = load_all_data()
        
        self.kmeans_labels = labels_dict['K-Means++']
        self.dbscan_labels = labels_dict['DBSCAN']
        self.hierarchical_labels = labels_dict['Hierarchical']
        self.ensemble_labels = labels_dict['Ensemble']

        # Pre-compute K-Means Centroids (Global)
        if not self.kmeans_labels.empty:
            self.kmeans_centroids = self._compute_kmeans_centroids()
        else:
            self.kmeans_centroids = None

    def _compute_kmeans_centroids(self):
        """Compute mean position for each K-Means cluster."""
        df = self.data.copy()
        df['label'] = self.kmeans_labels.values
        centroids = df.groupby('label').mean()
        return centroids

    def select_samples(self, n=5):
        """Randomly select n sample indices."""
        if len(self.data) < n:
            return self.data.index.tolist()
        
        # Use simple random sampling
        indices = np.random.choice(self.data.index, n, replace=False)
        return indices

    def explain_kmeans(self, selected_indices):
        """
        Explain K-Means assignment for selected samples.
        Calculates distance to each global centroid.
        """
        if self.kmeans_centroids is None:
            return "K-Means labels not found.", None

        samples = self.data.loc[selected_indices]
        
        # Calculate distances to all centroids
        # centroids is a DataFrame (k_clusters, n_features)
        # samples is a DataFrame (n_samples, n_features)
        
        dists = pairwise_distances(samples, self.kmeans_centroids)
        
        # Create a display DataFrame
        # Columns: Dist to Cluster 0, Dist to Cluster 1, ...
        # Index: Sample Name
        
        cols = [f"Khoảng cách đến C{c}" for c in self.kmeans_centroids.index]
        dist_df = pd.DataFrame(dists, columns=cols, index=selected_indices)
        
        # Identify closest cluster
        closest_cluster = dist_df.idxmin(axis=1)
        # Clean up column name to just Cluster ID for display
        closest_cluster = closest_cluster.apply(lambda x: x.split(" ")[-1])
        
        dist_df['Cụm được gán'] = closest_cluster.values
        
        return dist_df

    def explain_dbscan(self, selected_indices, eps=111.017, min_samples=5):
        """
        Explain DBSCAN classification for selected samples.
        Counts global neighbors within eps.
        """
        samples = self.data.loc[selected_indices]
        all_data = self.data
        
        # Calculate distances from selected samples to ALL points
        dists = pairwise_distances(samples, all_data)
        
        results = []
        for i, idx in enumerate(selected_indices):
            # Count neighbors within eps
            # Euclidean distance
            neighbors_count = np.sum(dists[i] <= eps)
            
            # DBSCAN logic:
            # -1 is the point itself (usually included in sklearn, so neighbors >= min_samples)
            # strictly speaking neighbors includes the point itself
            
            status = "Noise"
            # Logic: If neighbors (inc self) >= min_samples -> Core
            # If < min_samples but neighbor of a Core -> Border (Simplified here: just checking core condition)
            # For simplicity in this demo, we mainly check Core vs Not Core based on density
            
            if neighbors_count >= min_samples:
                status = "Điểm lõi (Core)"
            else:
                # Simplification: identifying border points requires checking their neighbors' status
                # We will label as "Noise/Border" for clarity
                status = "Nhiễu/Biên (Noise/Border)"
            
            results.append({
                "Sample": idx,
                f"Số hàng xóm (khoảng cách<{eps:.1f})": neighbors_count,
                "Ngưỡng (min_samples)": min_samples,
                "Trạng thái": status
            })
            
        return pd.DataFrame(results).set_index("Sample")

    def explain_hierarchical(self, selected_indices):
        """
        Explain Hierarchical (Ward) Logic on the subset.
        Shows distance matrix and first merge for the subset.
        """
        samples = self.data.loc[selected_indices]
        
        # Compute distance matrix between these 5 points
        dist_matrix = squareform(pdist(samples, metric='euclidean'))
        dist_df = pd.DataFrame(dist_matrix, index=selected_indices, columns=selected_indices)
        
        # Find minimum non-zero distance (simple linkage step)
        # Mask diagonal and lower triangle to avoid duplicates/zeros
        mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)
        valid_dists = dist_matrix[mask]
        
        if len(valid_dists) > 0:
            min_dist = np.min(valid_dists)
            # Find indices of min dist
            min_loc = np.where((dist_matrix == min_dist) & mask)
            # Access first pair found
            p1_idx = min_loc[0][0]
            p2_idx = min_loc[1][0]
            
            merge_pair = (selected_indices[p1_idx], selected_indices[p2_idx])
            explanation = f"Cặp gần nhất: **{merge_pair[0]}** và **{merge_pair[1]}** (Khoảng cách: {min_dist:.4f}). \n\nTrong liên kết Ward, cặp này sẽ là ứng viên để gộp dựa trên mức tăng phương sai nhỏ nhất."
        else:
            explanation = "Không đủ mẫu để tính toán liên kết."
            
        return dist_df, explanation

    def explain_ensemble(self, selected_indices):
        """
        Explain Ensemble Voting.
        Shows base labels and final assignment.
        """
        if self.ensemble_labels.empty:
            return "Ensemble labels not found."

        # Fetch labels for selected indices
        # We need to ensure indices align. The labels df uses same index as data.
        
        try:
            k_labels = self.kmeans_labels.loc[selected_indices]
            d_labels = self.dbscan_labels.loc[selected_indices]
            h_labels = self.hierarchical_labels.loc[selected_indices]
            final_labels = self.ensemble_labels.loc[selected_indices]
        except KeyError:
            return "Error: Some indices not found in label files."

        df_vote = pd.DataFrame({
            "K-Means": k_labels,
            "DBSCAN": d_labels,
            "Hierarchical": h_labels,
            "Kết quả Ensemble": final_labels
        })
        
        # Similarity explanation (Co-Association) for this subset
        # We simply compute the agreement among the 3 base models for these 5 points
        # to show the user "why" they might be grouped.
        
        # Base matrix (3 models x 5 samples)
        base_matrix = np.array([k_labels.values, d_labels.values, h_labels.values]) # shape (3, 5)
        
        n_samples = len(selected_indices)
        co_assoc = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                # Count how many models agree for sample i and sample j
                agreements = np.sum(base_matrix[:, i] == base_matrix[:, j])
                co_assoc[i, j] = agreements / 3.0
                
        co_assoc_df = pd.DataFrame(co_assoc, index=selected_indices, columns=selected_indices)
        
        return df_vote, co_assoc_df
