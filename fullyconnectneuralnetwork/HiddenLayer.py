import numpy as np
from Layer import Layer

class HiddenLayer(Layer):
    def __init__(self, name: str, number_of_neurons: int, number_of_input: int, activation_fn=None):
        super().__init__(name, number_of_neurons, number_of_input)
        self.activation_fn = activation_fn
        self.activations:np.ndarray = np.zeros(number_of_neurons)
        print(f"activation_fn is {self.activation_fn}")

    '''
        if inputs(batch_size=2, pixcel=784), 
            1st hidden layer의 출력의 형태는 (batch_size, hidden layer의 neurons의 수)가 되어야함.
            첫 번째 차원은 배치 크기 2를 나타내며, 각각 2개의 입력 데이터를 처리한 결과를 나타냅니다.
            두 번째 차원은 각각의 입력 데이터에 대한 16개의 뉴런 활성화 값을 나타냅니다.
     '''
    def forward(self, MULTIPLE_INPUTS: np.ndarray) -> np.ndarray:
        print(f"hiddenlayer's name is {self.name}", "forwardpropagation")
        self.activations = np.stack(
                    np.array(
                        [neuron.activate(MULTIPLE_INPUTS, self.activation_fn) for neuron in self.neurons]
                    ), axis=1)
        return self.activations
    
    def back(self, weights_changes:np.ndarray, bias_changes:np.ndarray) -> np.ndarray:
        print(f"hiddenlayer's name is {self.name}", "backpropagation")
        try :
            for neuron in enumerate(self.neurons):
                weights = neuron.weights - weights_changes
                bias = neuron.bias - bias_changes
                neuron.update(weights, bias)
                #self.activations = np.zeros(self.neurons)
        except Exception as e:
            print(e)
            raise e

    def get_activations(self) :
        return self.activations