import torch
import torch.nn as nn

class Linear(nn.Module):
    """An identity activation function"""
    def forward(self, x):
        return x
    

class SimpleVGG(nn.Module):
    def __init__(self, disable_activations: bool = False):
        """disable_activations: whether to generate network with Relus or not."""
        super(SimpleVGG, self).__init__()
        self.features = self._make_layers(disable_activations)

    def forward(self, x):
        # index of layers with feature maps
        select = [8, 13]
        # self.feature_maps is a list of Tensors, PytorchReceptiveField looks for 
        # this parameter and compute receptive fields for Tensors in self.feature_maps.
        self.feature_maps = []
        for l, layer in enumerate(self.features):
            x = layer(x)
            if l in select:
                self.feature_maps.append(x)
        return x

    def _make_layers(self, disable_activations: bool):
        activation = lambda: Linear() if disable_activations else nn.ReLU()
        layers = [
            nn.Conv2d(3, 64, kernel_size=3),
            activation(),
            nn.Conv2d(64, 64, kernel_size=3),
            activation(),
            
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            activation(),
            nn.Conv2d(128, 128, kernel_size=3),
            activation(), # 8
            
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3),
            activation(),
            nn.Conv2d(256, 256, kernel_size=3),
            activation(), # 13
        ]        
        return nn.Sequential(*layers)
    
def model_fn():
    model = SimpleVGG(disable_activations=True)
    model.eval()
    return model

from receptivefield.pytorch import PytorchReceptiveField

input_shape = [96, 96, 3]
rf = PytorchReceptiveField(model_fn)
rf_params = rf.compute(input_shape = input_shape)
import matplotlib.pyplot as plt
from receptivefield.image import get_default_image
rf.plot_rf_grids(get_default_image(input_shape, name='cat'), figsize=(20, 12), layout=(1, 2))
plt.savefig("temp.jpg")