'''
    Implementation of a simple ResNet Block
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

class DDBlock(nn.Module):
    num_neurons: int
    training: bool # --> training flag for enabling dropout & bn

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.num_neurons)(x)
        x = nn.Dropout(rate=0.5, deterministic = not self.training)(x)
        x = nn.BatchNorm(use_running_average = not self.training)(x) # using batch average instead of running avg when training
        return x

seed = 42  
keys = random.split(random.PRNGKey(seed), num=4)

model = DDBlock(num_neurons=3, training=True)
x = random.uniform(keys[0], (3,4,4))

# the unique identifier is due to Dropout
variables = model.init({'params': keys[1], 'dropout': keys[2]}, x)
print(variables)

y, non_trainable_params = model.apply(variables, x, rngs={'dropout': keys[3]}, mutable=['batch_stats'])

## eval mode
eval_model = DDBlock(num_neurons=3, training=False)

## No state update is needed
y = eval_model.apply(variables, x)