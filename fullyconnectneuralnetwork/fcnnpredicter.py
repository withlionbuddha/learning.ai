import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

# MNIST 데이터셋 로드 함수
def load_mnist_data():
  # Define a transform to convert the images to tensors and normalize them
    transform = transforms.Compose([
        transforms.ToTensor()         # Convert images to tensor
    ])

    # Load the MNIST dataset (train set for example)
    mnist_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Create a DataLoader to fetch images
    dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=2, shuffle=True)

    # Get a batch of images
    data_iter = iter(dataloader)
    x_train, labels = next(data_iter)  # Corrected this line

    # x_train을 (batch_size, 28*28) 형식으로 변환
    x_train = x_train.view(x_train.size(0), -1).numpy()  # 28x28 이미지를 784차원으로 변환
    labels = labels.numpy()  # 레이블을 numpy 배열로 변환    

    return x_train, labels

class MultiInputNeuron:
    def __init__(self, number_of_inputs: int):
        self.weights: np.ndarray = np.random.rand(number_of_inputs)
        self.bias: float = np.random.rand(1)
        self.activation: float = 0  # 하나의 활성화 값 저장

    def activate(self, inputs: np.ndarray, activation_fn=None) -> float:
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        self.activate = activation_fn(weighted_sum) if activation_fn else weighted_sum
        return self.activation

class Layer:
    def __init__(self, name: str, number_of_neurons: int, number_of_inputs: int):
        self.name = name
        self.number_of_neurons = number_of_neurons
        self.neurons = [MultiInputNeuron(number_of_inputs) for _ in range(number_of_neurons)]
                
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError("forward 메서드는 하위 클래스에서 구현되어야 합니다.")

# HiddenLayer 클래스 정의 (Layer 상속)
class HiddenLayer(Layer):
    def __init__(self, name: str, number_of_neurons: int, number_of_inputs: int, activation_fn=None):
        super().__init__(name, number_of_neurons, number_of_inputs)
        self.activation_fn = activation_fn

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.array([neuron.activate(inputs, self.activation_fn) for neuron in self.neurons])

# OutputLayer 클래스 정의 (Layer 상속)
class OutputLayer(Layer):
    def __init__(self, number_of_neurons: int, number_of_inputs: int):
        super().__init__(name="output", number_of_neurons=number_of_neurons, number_of_inputs=number_of_inputs)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.array([neuron.activate(inputs, activation_fn=None) for neuron in self.neurons])

# FCNN 클래스 정의
class FCNN:
    def __init__(self, hidden_layers: list, output_layer: OutputLayer):
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.learning_rate = 0.01

    def forward_propagation(self, inputs: np.ndarray) -> np.ndarray:
        v_inputs = inputs
        for hidden_layer in self.hidden_layers:
            v_outputs = hidden_layer.forward(v_inputs)
            v_inputs = v_outputs

        self.predicted_values = self.output_layer.forward(v_outputs)
        return self.predicted_values

    def backpropagation(self, inputs: np.ndarray, labels: np.ndarray, predicted_value: np.ndarray):
        output_error = (labels - predicted_value) * self.sigmoid_derivative(predicted_value)
        for i, neuron in enumerate(self.output_layer.neurons):
            neuron.weights += self.learning_rate * np.dot(self.hidden_layers[-1].forward(inputs).T, output_error[i])
            neuron.bias += self.learning_rate * np.sum(output_error[i])

    def calculate_cost_function(self, labels: np.ndarray, predicted_values: np.ndarray) -> float:
        return np.mean((labels - predicted_values) ** 2)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def train(self, inputs: np.ndarray, labels: np.ndarray, epochs: int):
        for epoch in range(epochs):
            predicted_value = self.forward_propagation(inputs)
            loss = self.calculate_cost_function(labels, predicted_value)
            self.backpropagation(inputs, labels, predicted_value)
            if epoch % 1000 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss}')

    @staticmethod
    def calculate_number_of_inputs(image_shape: tuple) -> int:
        return np.prod(image_shape)  # (28, 28) -> 784

# FCNNPredicter 클래스에서 test_input을 load_mnist_data()로 변경

class FCNNPredicter:
    def __init__(self):
        # MNIST 데이터 불러오기
        self.x_train, self.labels = load_mnist_data()

        # 모델 생성
        self.fcnn = self.createFCNN()

    def createFCNN(self):
        """
        FCNN 모델을 생성하는 메서드
        :return: FCNN 인스턴스
        """
        # x_train으로부터 이미지 크기 추출 및 입력 뉴런 수 계산
        image_shape = self.x_train.shape[1:]  # (784) 차원 추출
        number_of_inputs = FCNN.calculate_number_of_inputs(image_shape) 

        hidden_layer1 = HiddenLayer(
            name="hidden_layer_1"
            , number_of_neurons=16
            , number_of_inputs=number_of_inputs
            , activation_fn=lambda x: np.maximum(0, x))
        
        hidden_layer2 = HiddenLayer(
            name="hidden_layer_2"
            , number_of_neurons=16
            , number_of_inputs=16
            , activation_fn=lambda x: np.maximum(0, x))
        
        hidden_layers = [hidden_layer1, hidden_layer2]

        number_of_output_neurons = 10
        output_layer = OutputLayer(number_of_output_neurons, 16)

        return FCNN(hidden_layers=hidden_layers, output_layer=output_layer)

    def train_model(self):
        # FCNN 학습
        self.fcnn.train(self.x_train, self.labels, epochs=10)

    def predict(self, inputs):
        # 예측 함수 호출
        return self.fcnn.forward_propagation(inputs)

    @staticmethod
    def main():
        # FCNNPredicter 객체 생성
        fcnn_predicter = FCNNPredicter()

        # 모델 학습
        fcnn_predicter.train_model()

        # 테스트 입력 데이터 불러오기 (MNIST 데이터)
        #test_input, _ = load_mnist_data()

        # 예측 수행
        #prediction = fcnn_predicter.predict(test_input)

        # 예측 결과 출력
        #print(f"Prediction: {prediction}")

# FCNNPredicter main 함수 호출
if __name__ == "__main__":
    FCNNPredicter.main()

