'''
    Important terminology in Flax representing the trainables
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

class MyDenseImp(nn.Module):
    num_neurons: int
    weight_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        weight = self.param(
            'weight', #--> key in frozen dict
            self.weight_init, # --> random state implicitly called when calling init
            (x.shape[-1], self.num_neurons) # --> shape info
        )
        bias = self.param(
            'bias', 
            self.bias_init,
            (self.num_neurons,)
        )

        return jnp.dot(x, weight) + bias

seed = 42
x_key, init_key = random.split(random.PRNGKey(seed))

model = MyDenseImp(num_neurons=3)
x = random.normal(x_key, (4,4))
params = model.init(init_key, x)
y = model.apply(params, x)

print(jax.tree_map(
    jnp.shape,
    params
))