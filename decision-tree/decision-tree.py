import pandas as  pd 
import numpy as np
from sklearn.datasets import load_iris

class DecisionTreeClassifier:
    
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        
        
    def gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        g = 1 - np.sum((counts / counts.sum())**2)
        return g
    
    def best_split(self, X, y):
        best_gain = -1
        feature_idx = None
        threshold = None
        
        parent_gini = self.gini(y)
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for th in thresholds:
                left_mask = X[:, feature] <= th
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue
            
            left_gini = self.gini(y[left_mask])
            right_gini = self.gini(y[right_mask])
            
            n = y.shape[0]
            child_gini = (left_mask.sum() / n) * left_gini + (right_mask.sum() / n) * right_gini
            gain = parent_gini - child_gini
            
            if gain > best_gain:
                best_gain = gain
                feature_idx = feature
                threshold = th
            
        return feature_idx, threshold
    
    
    def _build_tree(self, X, y, depth=0):
        