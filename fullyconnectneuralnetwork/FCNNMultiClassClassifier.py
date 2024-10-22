import numpy as np
import MNIST
from FCNN import FCNN
import ActivationFunction as af
import CostFunction as cf
import Optimizer as optimizer
   
class FCNNMultiClassClassifier:
    def __init__(self):
        self.inputs:np.ndarray 
        self.labels:np.ndarray 
        self.fcnn:FCNN 
     
    def dataload(self, inputs, labels) :
        self.inputs = inputs
        self.labels = labels
        
    def createFCNN(self):
        # x_train으로부터 이미지 크기 추출 및 입력 뉴런 수 계산
        image_shape = self.inputs.shape[1:]  # (784) 차원 추출
        number_of_inputs = np.prod(image_shape) 

        self.fcnn = FCNN().addHiddenLayer(
                    16, number_of_inputs, activation_fn=lambda z: af.ReLU.activation_fn(z)
                ).addHiddenLayer(
                    16, 16, activation_fn=lambda z: af.ReLU.activation_fn(z)
                ).addOutputLayer(
                    16, 10, activation_fn=lambda z: af.Softmax.activation_fn(z))

    def train(self, inputs: np.ndarray, labels: np.ndarray, epochs: int):
        
        for epoch in range(epochs):
            self.fcnn.forward_propagation(inputs)
            
            cost = cf.CostFunction(labels, self.fcnn.get_output_layer().get_number_of_neurons())
            cost.calculate(self.fcnn.get_output_layer())
            
            gradient = optimizer.GradientDescent()
            gradient.step(self.fcnn, cost)
            
            if epoch % 1000 == 0:
                print(f'Epoch {epoch + 1}, Loss: {cost.get_loss()}, accuracy: {cost.get_accuracy()}, step: {gradient.get_step_count()}')

    def predict(self, MULTIPLE_INPUTS):
        # 예측 함수 호출
        return self.fcnn.forward_propagation(MULTIPLE_INPUTS)

    @staticmethod
    def main():

        # FCNNMultiClassClassifier 객체 생성
        multiclassifier = FCNNMultiClassClassifier()
        
        MULTIPLE_INPUTS, labels = MNIST.load_mnist_data()
        multiclassifier.dataload(MULTIPLE_INPUTS, labels)
        multiclassifier.createFCNN()
        multiclassifier.train(MULTIPLE_INPUTS, labels, epochs=5)

        # 테스트 입력 데이터 불러오기 (MNIST 데이터)
        #test_input, _ = load_mnist_data()

        #multiclassifier = multiclassifier.predict(test_input)

        # 예측 결과 출력
        #print(f"multiclassifier: {multiclassifier}")

# FCNNPredicter main 함수 호출
if __name__ == "__main__":
    FCNNMultiClassClassifier.main()
    
