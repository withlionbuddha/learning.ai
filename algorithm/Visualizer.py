#!/usr/bin/env python
# coding: utf-8

# In[9]:



# In[10]:

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[11]:


class Visualizer:
    def __init__(self, X, labels, y_encoded):
        self.X = X
        self.labels = labels
        self.y_encoded = y_encoded

    def reduce_dimensions(self):
        pca = PCA(n_components=2)
        self.X_pca = pca.fit_transform(self.X)
        print("차원 축소 완료.")

    def plot_clusters(self):
        plt.figure(figsize=(12, 6))

        # 예측된 군집
        plt.subplot(1, 2, 1)
        plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], c=self.labels, cmap='viridis', s=50)
        plt.title('GMM Predicted Clusters')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # 실제 품종
        plt.subplot(1, 2, 2)
        plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], c=self.y_encoded, cmap='tab10', s=50)
        plt.title('Actual Varieties')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        plt.tight_layout()
        plt.show()

