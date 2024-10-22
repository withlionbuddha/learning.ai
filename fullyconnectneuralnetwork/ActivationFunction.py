import numpy as np

class ReLU:
    @staticmethod
    def activation_fn(z: np.ndarray) -> np.ndarray:
        print("activation function is relu")

        return np.maximum(0, z)
    
    @staticmethod
    def derivative(z: np.ndarray) -> np.ndarray:
        print("relu partial derivative")
        return np.where(z > 0, 1, 0)

class Sigmoid:
    @staticmethod
    def activation_fn(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def derivative(z: np.ndarray) -> np.ndarray:
        return z * (1 - z)

class Softmax:
    ''' softmax(self, z: np.ndarray) -> np.ndarray
        출력값이 범위 0 <= softmax(z) <= 1
        Softmax(z_i) = e^(z_i - z_max) / ∑(j=1 to n) e^(z_j - z_max)
    '''
    
    @staticmethod
    def activation_fn(z: np.ndarray) -> np.ndarray:
        print("activation function is softmax")
        if z is None or np.any(z == None):
            raise ValueError("Invalid input: None values detected in softmax input.")
        
        z_max = np.max(z, axis=-1, keepdims=True)
        exp_for_z = np.exp(z - z_max)
        sum_exp_for_z = np.sum(exp_for_z, axis=-1, keepdims=True)
        return exp_for_z / sum_exp_for_z
