'''
    Stateless nature of JAX enables us to keep track of the MLP as pure weights
'''
import numpy as np
import jax

def init_MLP(layer_widths: list, par_key, scale=0.001): 
    '''
        Initialize MLP weight on normal distribution
        Input: 
            layer_widths width of each layer
            key: random key
            scale: to scale down the std
        Output: initialized weights
    '''
    params = []

    keys = jax.random.split(key, num=len(layer_widths)-1)

    for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys): # --> crossing the in & out
        weight_key, bias_key = jax.random.split(key)

        params.append(
           [scale* jax.random.normal(weight_key, (out_width, in_width)),
            scale* jax.random.normal(bias_key, (out_width,))]
        )

    print(jax.tree_map(
        lambda x: x.shape,
        params
    ))

    return params

seed = 0

key = jax.random.PRNGKey(seed)
MLP_params = init_MLP([784, 512, 256, 10], key)