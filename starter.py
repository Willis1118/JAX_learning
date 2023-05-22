import jax.numpy as jnp
import numpy as np

# vital transform functions
from jax import grad, jit, vmap, pmap

from jax import lax

from jax import make_jaxpr
from jax import random
from jax import device_put
import matplotlib.pyplot as plt

### build first nn using JAX & Flax here