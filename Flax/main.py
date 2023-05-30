import jax
from jax import lax, random, numpy as jnp

# NN lib built on top of JAX developed
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax.training import train_state

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import functools
from typing import Any, Callable, Sequence, Optional

import numpy as np
# import matplotlib.pyplot as plt

seed = 0

def init_params(model):
    key1, key2 = jax.random.split(jax.random.PRNGKey(seed))

    ## init dummy input
    x = jax.random.normal(key1, (10,))

    ## init call; remember jax handles state externally
    y, params = model.init_with_output(key2, x)

    print(y)
    print((jax.tree_map(lambda x: x.shape, params)))

    ## Note 1: automatic shape inference
    ##         notice that from above we did not specify shape of input; the input shape is inferred
    ## Note 2: immutable structure (hence frozen dict)
    ## Note 3: init_with_output will produce the inference as another output
    return params

if __name__ == '__main__':
    ## single forward-feed layer
    model = nn.Dense(features=5)

    ## All Flax NN layer inherit from the Module class
    print(nn.Dense.__base__)

    params = init_params(model)
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, (10,))

    ## Will model correctly infer the shape here? Nope, params init with shape (10)
    y = model.apply(params, x)
    print(y)

    ## Also this does not work anymore
    try:
        y = model(x)
    except Exception as e:
        print(e)


