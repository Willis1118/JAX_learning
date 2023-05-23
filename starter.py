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

print(jax.devices())

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

