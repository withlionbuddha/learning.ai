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
    
    ''' softmax(self, z: np.ndarray) -> np.ndarray
        출력값이 범위 0 <= softmax(z) <= 1
        Softmax(z_i) = e^(z_i - z_max) / ∑(j=1 to n) e^(z_j - z_max)
    '''
    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        print("activation function is softmax")
        if z is None or np.any(z == None):
            raise ValueError("Invalid input: None values detected in softmax input.")
        
        z_max = np.max(z, axis=-1, keepdims=True)
        exp_for_z = np.exp(z - z_max) 
        sum_exp_for_z = np.sum(exp_for_z, axis=-1, keepdims=True)  
        return exp_for_z / sum_exp_for_z

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
        return z * (1 - z)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        print("activation function is relu")
        return np.maximum(0, z)
    
    @staticmethod
    def relu_partial_derivative(z: np.ndarray) -> np.ndarray :
        print("relu partial derivative")
        return 0