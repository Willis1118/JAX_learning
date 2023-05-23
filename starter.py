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

##### Parallelism in JAX #####

x = np.arange(5) #signal
w = np.array([2.,3.,4.]) #kernel

def convolve(w, x):
    output = []

    for i in range(1, len(x)-1):
        output.append(jnp.dot(x[i-1:i+2], w)) # convolving on x with w
    
    return jnp.array(output)

result = convolve(w, x)
print(repr(result)) # return the canonical representation of the object

n_devices = jax.local_device_count()
print(f"Number of availabe devices: {n_devices}")

xs = np.arange(5 * n_devices).reshape(-1, 5) # --> -1 is inferred to be n_devices
ws = np.stack([w] * n_devices)

print(xs)
print(xs.shape, ws.shape)
