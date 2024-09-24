#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import typing as ty

# In[11]:


class Visualizer:
    def __init__(self, X, labels , y_encoded):
        self.X = X
        self.labels = labels
        self.y_encoded = y_encoded
        
    def reduce_dimensions(self):
        pca = PCA(n_components=2)
        self.X_pca = pca.fit_transform(self.X)
        #print("[---- reduce dimensions.----]")
        
    def plot_confusion_matrix(self, conf_matrix):
        plt.figure(figsize=(8,6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.labels, yticklabels=self.labels)

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
    def plot_clusters(self):
        plt.figure(figsize=(12, 6))

        # GMM Predicted Clusters
        plt.subplot(1, 2, 1)
        plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], c=self.labels, cmap='viridis', s=50)
        plt.title('GMM Predicted Clusters')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # Actual Varieties
        plt.subplot(1, 2, 2)
        plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], c=self.y_encoded, cmap='tab10', s=50)
        plt.title('Actual Varieties')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        plt.tight_layout()
        plt.show()

