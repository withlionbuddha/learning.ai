import numpy as np
from HiddenLayer import HiddenLayer
from OutputLayer import OutputLayer

def chainable_method(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self  # 메서드가 끝난 후 self 반환
    return wrapper
    
class FCNN:
    def __init__(self):
        self.hidden_layers:list = []
        self.output_layer:OutputLayer = None

    @chainable_method
    def addHiddenLayer(self, number_of_neurons:int, number_of_inputs:int ,activation_fn=None) -> None :
        self.hidden_layers.append(
            HiddenLayer(
                "hiddenlayer_"+ str(len(self.hidden_layers))
                , number_of_neurons
                , number_of_inputs
                , activation_fn
            ))
        
    @chainable_method
    def addOutputLayer(self, number_of_neurons:int, number_of_inputs:int , activation_fn=None) :
        self.output_layer = OutputLayer(
                number_of_neurons
                , number_of_inputs
                , activation_fn)
        return self
    
    def forward_propagation(self, inputs: np.ndarray) -> np.ndarray:
        for layer_index, hidden_layer in enumerate(self.hidden_layers):
            if layer_index == 0:
                hidden_layer_outputs = hidden_layer.forward(inputs)
            else :
                hidden_layer_outputs = hidden_layer.forward(hidden_layer_outputs)
        self.output_layer.forward(hidden_layer_outputs)

    def get_output_layer(self) :
        return self.output_layer
    def get_hidden_layers(self) :
        return self.hidden_layers
    
 