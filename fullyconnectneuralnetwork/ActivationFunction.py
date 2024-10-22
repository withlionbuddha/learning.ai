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
    
    @staticmethod
    def activation_fn(z: np.ndarray) -> np.ndarray:
        print("activation function is softmax")
        
        z_max = np.max(z, axis=-1, keepdims=True)
        exp_for_z = np.exp(z - z_max)
        sum_exp_for_z = np.sum(exp_for_z, axis=-1, keepdims=True)
        return exp_for_z / sum_exp_for_z
