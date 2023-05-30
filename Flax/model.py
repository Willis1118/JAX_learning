'''
    Creating custom model based on Flax
'''

import jax
from jax import lax, random, numpy as jnp

# NN lib built on top of JAX developed
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax.training import train_state

import optax

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import functools
from typing import Any, Callable, Sequence, Optional

import numpy as np

class MLP(nn.Module):
    num_neuron_per_layer: Sequence[int] # data field

    def setup(self): # since data class implicitly called __init__
        self.layers = [nn.Dense(n) for n in self.num_neuron_per_layer]
    
    def __call__(self, x): #overwrite the __call__ operator to make the class callable
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x
    
seed = 42
x_key, init_key = random.split(random.PRNGKey(seed))

model = MLP([16,8,1])
x = random.normal(x_key, (4,4))
params = model.init(init_key, x)

y = model.apply(params, x)

print(jax.tree_map(
    jnp.shape, params
))

print(f'output: {y}')
