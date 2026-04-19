import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

class HCFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Hierarchical Clustering Feature Engineer
    
    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters
    return_scaled : bool, default=False
        If True: Returns scaled features (for SVR/MLP)
        If False: Returns unscaled features (for RF/XGBoost)
    """
    def __init__(self, 
                 n_clusters: int = 2, 
                 return_scaled: bool = False):
        
        self.n_clusters = n_clusters
        self.return_scaled = return_scaled
    
    def fit(self, X, y=None):
        """Fit HC on scaled data"""
        X_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        
        # Scaling
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_array)
        
        # Fit HC on scaled data
        self.cluster_model_ = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage="ward"
        )
        self.cluster_labels_ = self.cluster_model_.fit_predict(X_scaled)
        
        # Compute centroids in scaled space
        self.centroids_ = np.array([
            X_scaled[self.cluster_labels_ == i].mean(axis=0)
            for i in range(self.n_clusters)
        ])
        
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = None
        
        return self
    
    def transform(self, X):
        """
        Transform data by adding cluster features
        
        Returns scaled or unscaled based on return_scaled parameter
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        
        # Scale data (always needed for cluster assignment)
        X_scaled = self.scaler_.transform(X_array)
        
        # Assign to nearest centroid (in scaled space)
        distances = pairwise_distances(X_scaled, self.centroids_)
        cluster_labels = np.argmin(distances, axis=1)
        
        # Distance to centroid (in scaled space)
        dist_to_centroid = distances[np.arange(len(X_scaled)), cluster_labels]
        
        # Combine features based on return_scaled
        if self.return_scaled:
            # Return SCALED features (for SVR/MLP)
            X_new = np.hstack([
                X_scaled,                          # Scaled original features
                cluster_labels.reshape(-1, 1),     # cluster_id (0 or 1)
                dist_to_centroid.reshape(-1, 1)    # Scaled distance
            ])
        else:
            # Return UNSCALED features (for RF/XGBoost)
            X_new = np.hstack([
                X_array,                           # Original unscaled features
                cluster_labels.reshape(-1, 1),     # cluster_id (0 or 1)
                dist_to_centroid.reshape(-1, 1)    # Distance in scaled space
            ])
        
        # Return as DataFrame if input was DataFrame
        if isinstance(X, pd.DataFrame) and not self.return_scaled:
            # Only return DataFrame for unscaled (to preserve column names)
            new_columns = self.feature_names_in_ + ['cluster_id', 'dist_to_centroid']
            return pd.DataFrame(X_new, columns=new_columns, index=X.index)
        
        return X_new
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        if self.feature_names_in_ is not None:
            return np.array(self.feature_names_in_ + ['cluster_id', 'dist_to_centroid'])
        return None