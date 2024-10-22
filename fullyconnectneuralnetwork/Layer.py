import numpy as np
from MultiInputNeuron import MultiInputNeuron

class Layer:
    def __init__(self, name: str, number_of_neurons: int, number_of_input: int):
        self.name:str = name  
        self.number_of_neurons:int = number_of_neurons
        self.number_of_input:int = number_of_input
        self.neurons:np.ndarray = [MultiInputNeuron(number_of_input) for _ in range(number_of_neurons)]

        print(f"hiddenlayer's name is", self.name)
        print(f"number_of_neurons {number_of_neurons}", "number_of_input {number_of_input}")
        
    def forward(self, MULTIPLE_INPUTS: np.ndarray) -> np.ndarray:
        raise NotImplementedError("forward 메서드는 하위 클래스에서 구현되어야 합니다.")
    
    def back(self, errors: np.ndarray) -> np.ndarray:
        raise NotImplementedError("back 메서드는 하위 클래스에서 구현되어야 합니다.")
    
    def get_name(self) :
        return self.name
    def get_number_of_neurons(self) :
        return self.number_of_neurons
    def get_neurons(self) :
        return self.neurons
    def get_number_of_input(self) :
        return self.number_of_input
