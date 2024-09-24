import pandas as pd

class DataAnalyzer:
    def __init__(self, X, y_true):
        self.X = X
        self.y_true = y_true
        self.df = None

    def create_dataframe(self, iris):
        feature_names = iris.feature_names

        self.df = pd.DataFrame(data=self.X, columns=feature_names)

        # 타겟(종류)을 추가
        self.df['species'] = self.y_true

        # species 컬럼의 숫자를 실제 품종 이름으로 매핑
        self.df['species'] = self.df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    def analyze_data(self):
        if self.df is None:
            raise ValueError("DataFrame이 생성되지 않았습니다. 먼저 create_dataframe() 메서드를 호출하세요.")

        print("DataFrame Info")
        print(self.df.info())
        print("\n DataFrame descibe")
        print(self.df.describe())
        print("\n DataFrame head")
        print(self.df.head())

  