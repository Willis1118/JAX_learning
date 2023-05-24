import jax.numpy as jnp
import numpy as np

# vital transform functions
import jax
from jax import grad, jit, vmap, pmap

from jax import lax

from jax import make_jaxpr
from jax import random
from jax import device_put

from flax import linen as nn
### build first nn using JAX & Flax here

from copy import deepcopy
from typing import Tuple, NamedTuple
import functools

# Params State class
class Params(NamedTuple):
    weight: jnp.ndarray
    bias: jnp.ndarray

lr = 5e-3

def init_model(rng):
    '''
        initialize model weight
        input: jax PRNG key
        output: initialized model weights & biases
    '''
    weight_key, bias_key = jax.random.split(rng)
    weight = jax.random.normal(weight_key, ())
    bias = jax.random.normal(bias_key, ())

    return Params(weight, bias)

def forward(params, xs):
    return params.weight * xs + params.bias

# def loss_fn(params, )

'''
    Modeling data: y = w*x + b + eps
'''
true_w, true_b = 2, -1
xs = np.random.normal(size=(128,1))
noise = 0.5*np.random.normal(size=(128,1))
ys = true_w * xs + true_b + noise

params = init_model(jax.random.PRNGKey(0))
n_devices = jax.local_device_count()
replicated_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), params) # --> sending params to all local devices
print(replicated_params)
