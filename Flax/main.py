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

if __name__ == '__main__':
    ## single forward-feed layer
    model = nn.Dense(features=5)

    ## All Flax NN layer inherit from the Module class
    print(nn.Dense.__base__)

