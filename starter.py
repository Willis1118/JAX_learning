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

print(jax.devices())