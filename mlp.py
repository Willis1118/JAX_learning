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

def loss_fn(params, xs, ys):
    return jnp.mean((forward(params, xs) - ys)**2)

@functools.partial(jax.pmap, axis_name='batch') # axis 0 by default
def update(params, xs, ys):
    '''
        Backward propogration
        input: distributed params, data, lable
        output: updated params, loss
    '''
    loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)

    # synchronize the gradient on all devices
    grads = jax.lax.pmean(grads, axis_name='batch')

    # Also the loss
    loss = jax.lax.pmean(loss, axis_name='batch')

    # now performing SGD on EACH device
    new_params = jax.tree_map(
        lambda param, g: param - g*lr, params, grads
    )

    # if using other optimizer, we could do
    # updates, new_opt_state = optimizer(grads, opt_state)
    # then use updates instead of grads to update the params

    return new_params, loss

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

def reshape_for_pmap(data, n_devices):
    '''
        sharding the data to distributed devices
        input: data:[batch, size]
        output: shardedData: [devices, minibatch, size]
    '''
    return data.reshape(n_devices, data.shape[0] // n_devices, *data.shape[1:])

xp = reshape_for_pmap(xs, n_devices)
yp = reshape_for_pmap(ys, n_devices)

print(xp.shape, yp.shape)

def type_after_update(name, obj):
    print(f"after first `update()`, `{name}` is a {type(obj)}")

num_epochs = 5000
for epoch in range(num_epochs):

    # where params and data gets communicated to devices
    replicated_params, loss = update(replicated_params, xp, yp)

    if epoch == 0:
        type_after_update('replicated_params.weight', replicated_params.weight)
        type_after_update('loss', loss)
        type_after_update('xp', xp)
    
    if epoch % 100 == 0:
        print(loss.shape)
        print(f"Step {epoch:3d}, loss {loss[0]:3f}")

# extract the params from distributed to a single device
# device_get will extract data from TPU and fetch into host memory
params = jax.device_get(jax.tree_map(lambda x: x[0], replicated_params))