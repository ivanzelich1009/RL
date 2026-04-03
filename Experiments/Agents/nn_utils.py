import torch
import numpy as np
import torch.nn as nn
from Experiments.Envs.Connect4 import Connect4Env
num_filters=64
action_size=Connect4Env().action_space.n


'''Here we define utility functions that will be used to create the actor and critic networks for each agent.'''

def initialize_layer(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes the weights and biases of a given layer.

    Parameters:
    layer (nn.Module): The neural network layer to initialize.
    std (float): The standard deviation for orthogonal initialization of weights. Default is sqrt(2).
    bias_const (float): The constant value to initialize the biases. Default is 0.0.

    Returns:
    nn.Module: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

#Defines the skip-connections part of the convolutional neural net. 
# TO DO: Could see if adding batchNorm before conv changes things.
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            initialize_layer(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.GroupNorm(1, channels),
            nn.LeakyReLU(),
            initialize_layer(nn.Conv2d(channels, channels, 3, padding=1)),
            nn.GroupNorm(1, channels)
        )
        self.activation = nn.LeakyReLU()
    def forward(self, x):
        return self.activation(x+self.block(x))
    
'''Produce a neural network from spec = list of tuples of the form ("name of layer", out_dim, {std: , bias:,}, {kernel, padding}).
We will keep track of the shape after applying each layer.
'''
def parse_layer_def(layer_def):
    kind = layer_def[0]
    args = layer_def[1:]
    if len(args) == 0:
        return kind, None, {}, {}
    elif len(args) == 1:
        return kind, args[0], {}, {}
    else:
        if kind == 'conv2d':
            return kind, args[0], args[1], args[2]
        else:
            return kind, args[0], args[1], {}

def build_network(input_shape, spec):
    layers=[]
    current_shape = input_shape

    for i, layer_def in enumerate(spec):
        kind, out, initial, convparams = parse_layer_def(layer_def)
        std = initial.get('std', np.sqrt(2))
        bias = initial.get('bias', 0.0)
        kernel = convparams.get('kernel', 3)
        padding = convparams.get('padding', 1)
        stride = convparams.get('stride', 1)
        
        if kind == 'conv2d':
            layer = initialize_layer(nn.Conv2d(current_shape[0], out, kernel_size=kernel, stride=stride, padding=padding), std = std, bias_const = bias)
            new_layers = [layer, nn.GroupNorm(1, out), nn.LeakyReLU()]
            layers += new_layers
            with torch.no_grad():
                dummy = torch.zeros(1, *current_shape)
                for l in new_layers:
                    dummy = l(dummy)
                current_shape = dummy.shape[1:]

        elif kind == 'residual':
            layers += [ResidualBlock(out)]

        elif kind == 'flatten':
            layers += [nn.Flatten()]
            current_shape = (int(np.prod(current_shape)),)

        elif kind == 'linear':
            layers += [initialize_layer(nn.Linear(current_shape[-1], out), std = std, bias_const = bias)]
            if len(current_shape)==1:
                current_shape=(out,)
            else:
                current_shape=(*current_shape[:-1],out)
        
        elif kind == 'tanh':
            layers.append(nn.Tanh())

        elif kind == 'relu':
            layers.append(nn.ReLU())

        elif kind == 'leaky_relu':
            layers.append(nn.LeakyReLU())

        else:
            raise ValueError(f'Unknown layer type: {kind}')
    
    return nn.Sequential(*layers)
