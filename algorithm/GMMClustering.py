#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode

class GMMClustering:
    def __init__(self, n_components=3, covariance_type='full', random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type, random_state=self.random_state)
        self.scaler = StandardScaler()

    def fit(self, X):
        self.X_scaled = self.scaler.fit_transform(X)
        self.gmm.fit(self.X_scaled)
        self.labels = self.gmm.predict(self.X_scaled)
        return self.labels

    def adjust_labels(self, y_true):
        labels = np.zeros_like(self.labels)
        for i in np.unique(self.labels):
            mask = (self.labels == i)
            labels[mask] = mode(y_true[mask], keepdims=True)[0][0]
        self.labels_adjusted = labels
        return self.labels_adjusted

    def calculate_accuracy(self, y_true):
        accuracy = accuracy_score(y_true, self.labels_adjusted)
        return accuracy

    def get_confusion_matrix(self, y_true):
        cm = confusion_matrix(y_true, self.labels_adjusted)
        return cm

    def get_scaled_data(self):
        return self.X_scaled


# In[ ]:




