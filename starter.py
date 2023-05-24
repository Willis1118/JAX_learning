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

##### Firstly optimize with vmap #####
vmap_result = jax.vmap(convolve)(ws, xs) # remember that this broadcast the action on the first dimension by defualt
print(repr(vmap_result))

##### Optimize with pmap #####
pmap_result = jax.pmap(convolve)(ws, xs)
print(repr(pmap_result)) # this now running on multiple devices by sharding the batched data
# notice that this has no cross-device communication costs; Computations are done independently

# Or we can also do this
# None is telling pmap to broadcast w to first dimension of xs
res = jax.pmap(convolve, in_axes=(None, 0))(w, xs)
print(repr(res))

# To communicate cross devices
def normalize_conv(w, x):
    output = []

    for i in range(1, len(x) - 1):
        output.append(jnp.dot(x[i-1:i+2], w))
    
    output = jnp.array(output)

    return output / jax.lax.psum(output, axis_name='batch_dim') 
    # here lax.psum allows as to sum up along the batch dimension over all devices 
    # and broadcast the result to each device to perform the division
    # it can be arbitrary as long as it matches the broadcasted name & axies specified in pmap

res_pmap = jax.pmap(normalize_conv, axis_name='batch_dim', in_axes=(None, 0))(w, xs) # the name 'batch_dim' referring to first axes of x
res_vmap = jax.vmap(normalize_conv, axis_name='batch_dim', in_axes=(None, 0))(w, xs)

print('pmap norm', res_pmap)
print('vmap norm', res_vmap)

print('verify that conv is indeed normalized', repr(sum(res_pmap[:,0])))
print('verify the distributed result', vmap_result[0][0] / sum(vmap_result[:,0]), res_vmap[0][0])

##### A couple more useful functions #####
def mle(x, y):
    return sum((x-y)**2)
x = jnp.arange(4, dtype=jnp.float32)
y = x+1

# An efficient way to return both grad and loss value
print(jax.value_and_grad(mle)(x,y))