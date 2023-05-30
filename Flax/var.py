'''
    Important terminology in Flax representing both the trainables and non-trainables
    (i.e. in BatchNorm, the parameters and the running mean)
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

class BiasAdderWithRunningMean(nn.Module):
    decay: float = 0.99

    @nn.compact
    def __call__(self, x):
        is_init = self.has_variable('batch_stats', 'ema')

        ## Notice that batch_stats is not an arbitrary name
        ## Flax uses this hardcoded name in implementation of Batch Norm
        ## Here it is a collection name
        ema = self.variable('batch_stats', 'ema', lambda shape: jnp.zeros(shape), x.shape[1:]) # --> contains non-trainable params

        ## self.params will by default add the variable to 'param' collection (compare with ema above)
        ## the function sig required non-positional argument key
        bias = self.param('bias', lambda key, shape: jnp.zeros(shape), x.shape[1:])

        if is_init:
            # self.variable returns a reference hence .value
            ema.value = self.decay * ema.value + (1.0 - self.decay) * jnp.mean(x, axis=0, keepdims=True)
        
        return x - ema.value + bias # ema stands for exponentially moving average

seed = 42
x_key, init_key = random.split(random.PRNGKey(seed))

model = BiasAdderWithRunningMean([16,8,1]) # --> calling data class
x = random.normal(x_key, (10,4))
params = model.init(init_key, x)

print(f'Multiple collections = {params}')

## for variables to be updated during the run, we need to explicitly specify them as mutable
y, updated_non_params = model.apply(params, x, mutable=['batch_stats'])
print(updated_non_params)

