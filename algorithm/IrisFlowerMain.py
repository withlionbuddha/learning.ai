#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IrisFlowerFactory import IrisFlowerFactory
from GMMClustering import GMMClustering
from Visualizer import Visualizer
from DataAnalyzer import DataAnalyzer
from sklearn import datasets
from sklearn.datasets import load_iris


# In[ ]:


class IrisFlowerMain:
    def __init__(self):
        self.X = None
        self.y_true = None
        self.iris_flowers = []
        self.labels = None
        self.labels_adjusted = None
        self.gmm_cluster = None
        self.accuracy = None
        self.cm = None
        self.visualizer = None

    def load_data(self):
        # Iris 데이터셋 로드
        iris = datasets.load_iris()
        self.X = iris.data
        self.y_true = iris.target
        
        data_analyzer = DataAnalyzer(self.X, self.y_true)
        data_analyzer.create_dataframe(iris)
        data_analyzer.analyze_data()

    def create_iris_flowers(self):
        # IrisFlowerFactory를 사용하여 iris_flowers 리스트 생성
        factory = IrisFlowerFactory()
        self.iris_flowers = factory.get_iris_flowers()

        # 각 IrisFlower 객체 출력
        for iris_flower in self.iris_flowers:
            print(iris_flower)
            print('-' * 40)

    def perform_clustering(self):
        # GMM 클러스터링 수행
        self.gmm_cluster = GMMClustering()
        self.labels = self.gmm_cluster.fit(self.X)

        # 레이블 조정
        self.labels_adjusted = self.gmm_cluster.adjust_labels(self.y_true)

        # 정확도 및 혼동 행렬 계산
        self.accuracy = self.gmm_cluster.calculate_accuracy(self.y_true)
        print(f'Accuracy: {self.accuracy:.2f}')

        self.cm = self.gmm_cluster.get_confusion_matrix(self.y_true)
        print('Confusion Matrix:')
        print(self.cm)

    def visualize_results(self):
        # 시각화
        X_scaled = self.gmm_cluster.get_scaled_data()
        self.visualizer = Visualizer(X_scaled, self.labels_adjusted, self.y_true)
        self.visualizer.reduce_dimensions()
        self.visualizer.plot_clusters()

    def run(self):
        # 메서드를 순서대로 실행
        self.load_data()
        self.create_iris_flowers()
        self.perform_clustering()
        self.visualize_results()

if __name__ == '__main__':
    app = IrisFlowerMain()
    app.run()
