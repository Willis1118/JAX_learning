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

def make_loss(model, xs, ys):

    def mse_loss(params):
        '''
            Given xs, ys, params, compute the loss
        '''
        ## inner here simply meaning inner product of two 1-D arrays
        return jnp.mean(
            jax.vmap(
                lambda x, y: jnp.inner(y-model.apply(params, x), y-model.apply(params, x)), 
                in_axes=(0,0)
                )(xs, ys), 
            axis=0)
    
    return jax.jit(mse_loss)

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
        print(e) # --> unbound module

    ##### Simple Linear Reg

    n_samples = 1000
    x_dim = 2
    y_dim = 1
    noise_amp = 0.1

    ## Generate gt W & b
    key, w_k, b_k = jax.random.split(jax.random.PRNGKey(seed), num=3)
    W = jax.random.normal(w_k, (x_dim, y_dim))
    b = jax.random.normal(b_k, (y_dim,))

    true_params = freeze({'params': {'bias': b, 'kernel': W}})

    ## Generate Data
    key, x_k, eps_k = jax.random.split(key, num=3)
    xs = jax.random.normal(x_k, (n_samples, x_dim))
    ys = jnp.dot(xs, W) + b
    ys += noise_amp * jax.random.normal(eps_k, (n_samples, y_dim))

    print(f"input shape: {xs.shape}, target shape: {ys.shape}")
 
    model = nn.Dense(features=y_dim)
    params = model.init(key, xs)
    print(f'init params: {params}')

    criterion = make_loss(model, xs, ys)

    ## notice that the dataset is contained in the closure
    value_and_grad_fn = jax.value_and_grad(criterion)

    ## Basic config
    lr = 0.3
    epochs = 20
    log_every = 5

    print('-' * 50)
    for epoch in range(epochs):
        loss, grads = value_and_grad_fn(params)

        ## SGD
        params = jax.tree_map(
            lambda p, g: p - lr*g,
            params,
            grads
        )

        if epoch % log_every == 0:
            print(f'Epoch: {epoch:3f}, Loss: {loss:3f}')
    
    print('-' * 50)
    print(f'Learned Params: {params}')
    print(f'GT_params: {true_params}')



