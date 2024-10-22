import numpy as np
from CostFunction import CostFunction
from FCNN import FCNN
from abc import ABC, abstractmethod

class Optimizer(ABC) :      
    @abstractmethod
    def step(self, model:FCNN, cost:CostFunction) -> float:
        pass
    
class GradientDescent(Optimizer) :
    def __init__(self):
        self.learning_rate:float = 0.01
        self.step_count:int = 0
        
    def step(self, model:FCNN, cost:CostFunction) -> float:
        try :
            weights_changes = self._get_weights_changes(
                    model.get_hidden_layers()[-1].get_activations().T
                    , cost.get_errors())
            
            bias_changes = self._get_bias_changes(cost.get_errors())
            
            model.get_output_layer().back(weights_changes, bias_changes)
            
           # for hidden_layer in enumerate(model.get_hidden_layers()):
           #     hidden_layer.back()
            self.step_count += 1
        except Exception as e:
            print(e)
            
    '''
    # weight의 변화량
        ΔW_output = learning_rate * output_error * hidden_layer_output.T
            learning_rate는 학습률 
            output_error는 출력층에서 계산된 오차 
            hidden_layer_output.T는 은닉층의 출력값의 전치행렬 
    '''
    def _get_weights_changes(self, activations:np.ndarray, errors:np.ndarray) -> np.ndarray:
        if activations is None or errors is None:
            raise ValueError("Cannot get weights_changes: activations or errors is None.")

        return self.learning_rate * np.dot(activations, errors)

    def _get_bias_changes(self, errors:np.ndarray) -> np.ndarray :
        if errors is None :
            raise ValueError("Cannot get bias_changes: errors is None.")
        
        return self.learning_rate * np.sum(errors, axis=0)
    
    def get_step_count(self) :
        return self.step_count
