import numpy as np
from Layer import Layer

class OutputLayer(Layer):
    def __init__(self, number_of_inputs: int, number_of_neurons: int,  activation_fn=None):
        super().__init__(
            name="output_layer" 
            , number_of_inputs=number_of_inputs
            , number_of_neurons=number_of_neurons
           )
        self.activation_fn = activation_fn
        self.activations:np.ndarray =  np.zeros(number_of_neurons)
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        print(f"hiddenlayer's name is {self.name}", "forwardprogation")

        #output layer에서는 각각의 neuron은 가중합(weighted sum)만을 계산한다.
        #output layer에서는 전체 weighted sum에 대해서 activation funcion을 적용한다.
        self.activations = self.activation_fn (
            np.stack(
                np.array(
                    [neuron.activate(inputs, None) for neuron in self.neurons]
                )
            , axis=1)
        )
        return self.activations
        
    def back(self, weights_changes:np.ndarray, bias_changes:np.ndarray) -> None : 
        print(f"hiddenlayer's name is {self.name}", "backpropagation")

        try :
            for i, neuron in enumerate(self.neurons):
                weights = neuron.weights - weights_changes[i]
                bias = neuron.bias - bias_changes[i]
                neuron.update(weights, bias)
        except Exception as e:
            print(e)
            raise e
            
    def get_activations(self) -> np.ndarray:
        return self.activations
    