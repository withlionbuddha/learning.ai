import numpy as np

class MultiInputNeuron:
    def __init__(self, number_of_input: int):
        '''
            하나의 입력 크기에 대한 weights 생성, neuron의 weights(가중치)는 서로 다른 여러 입력 데이터에 동일하게 적용
        '''
        self.weights:np.ndarray = np.random.rand(number_of_input) 
        self.bias:float = np.random.rand(1)

    def activate(self, MULTIPLE_INPUTS: np.ndarray, activation_fn=None) -> float:
        '''
            MULTIPLE_INPUTS 다수의 inputs
        '''
        weighted_sum = (MULTIPLE_INPUTS @ self.weights) + self.bias
        self.activation = activation_fn(weighted_sum) if activation_fn else weighted_sum
        
        print(f"[neuron.activate]inputs {MULTIPLE_INPUTS} , weighted_sum {weighted_sum}")
        print(f"[neuron.activate]activation {self.activation}")
        return self.activation
    
    def update(self, weights:np.ndarray, bias:float) : 
        self.weights:np.ndarray = weights
        self.bias:float = bias
        
        print(f"[neuron.update] weights {weights}, bias {bias}")
    
  
