import numpy as np

class MultiInputNeuron:
    def __init__(self, number_of_inputs: int):
        self.weights:np.ndarray = np.random.rand(number_of_inputs)
        self.bias:float = np.random.rand(1)

    def activate(self, inputs: np.ndarray, activation_fn=None) -> float:
        weighted_sum = (inputs @ self.weights) + self.bias
        self.activation = activation_fn(weighted_sum) if activation_fn else weighted_sum
        
        print(f"[neuron.activate]inputs {inputs}, weighted_sum {weighted_sum}")
        print(f"[neuron.activate]activation {self.activation}")
        return self.activation
    
    def update(self, weights:np.ndarray, bias:float) : 
        self.weights = weights
        self.bias = bias
        
        print(f"[neuron.update] weights {weights}, bias {bias}")
    
  
