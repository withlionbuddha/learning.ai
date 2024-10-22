import numpy as np
from OutputLayer import OutputLayer

class CostFunction() :
    def __init__(self, labels:np.ndarray, number_of_inputs:int):
        self.labels:np.ndarray = labels
        self.accuracy:float = 0.0
        self.errors:np.ndarray = np.zeros(number_of_inputs)
        self.loss:float = 0.0
        
    def calculate(self, output_layer:OutputLayer) -> np.ndarray:
        probility_values = output_layer.get_activations()
        print(f"outputlayer.activations(=probility_values) is {probility_values}")
        
        self.erros = self._calculate_errors(probility_values, output_layer.get_number_of_neurons())
        self.accuracy = self._calculate_accuracy(probility_values)
        self.loss = self._calculate_cross_entropy_loss(probility_values)
            
    def get_errors(self) -> np.ndarray:
        return self.erros 
    
    def get_accuracy(self) -> float:
        return self.accuracy
    
    def get_loss(self) -> float:
        return self.loss
    
    def _calculate_errors(self, activations:np.ndarray, number_of_neurons:int) ->np.ndarray:
        return activations - self._get_one_hot_coding(number_of_neurons)

    def _get_one_hot_coding(self, number_of_neurons:int) -> np.ndarray:
        return np.array([np.eye(number_of_neurons)[label] 
                    for label in self.labels])
    
    ''' def _calculate_cross_entropy_loss(self) 
        L = - (1/N) * Σ (from i=1 to N) [log(p_true^i + ε)]
            N은 배치 크기,
            Σ는 합(sum)을 의미하며, i=1부터 N까지의 합을 구합니다,
            p_true^i는 i번째 샘플의 정답 클래스에 대한 예측 확률,
            ε는 작은 수로, 로그에서 0으로 나누는 것을 방지하기 위한 값입니다.
    '''
    def _calculate_cross_entropy_loss(self, activations: np.ndarray) -> float :
        p = self._get_probility_values_for_labels(activations)
        return self._average_after_sigma(self._loss(p))
    
    def _get_probility_values_for_labels(self, activations: np.ndarray) -> np.ndarray:
        batch_size = activations.shape[0]
        dimension = activations.ndim
        if (dimension == 1) :
            probability_for_labels = activations[self.labels]
        else :
            probability_for_labels = activations[np.arange(batch_size), self.labels]
            return probability_for_labels
        
    def _loss(self, p:np.ndarray) -> float:
        # 0 < x < 1 => 음수
        # 로그를 사용하여 확률 값을 손실 값으로 변환
        epsilon = 1e-15 # log(0)의 오류 방지를 위해서
        minusone = -1   # 음수에서 양수로 전환하여 부호에 상관없이 값의 크기를 증폭
        return np.log(p + epsilon) * minusone
    
    def _average_after_sigma(self, operand) -> float:
        return np.mean(operand) # == 1/n * sigma(operand)
    
    def _calculate_accuracy(self, activations:np.ndarray) -> np.ndarray:
        if activations is None :
            raise ValueError("Cannot calculate accuracy: activations is None.")
        
        max_values_indexs = np.argmax(activations, axis=-1) 
        self.accuracy = np.sum(max_values_indexs == self.labels) / self.labels.size