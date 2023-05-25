'''
    Stateless nature of JAX enables us to keep track of the MLP as pure weights
'''
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

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

    keys = jax.random.split(par_key, num=len(layer_widths)-1)

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

def MLP_predict(params, x):
    '''
        forward run of MLP
        input:
            params: parameter of the MLP
            x: data
        output:
            forward result
    '''
    hidden_layer, last_layer = params[:-1], params[-1]

    for w, b in hidden_layer:
        x = jax.nn.relu(jnp.dot(w, x) + b)
    
    logits = jnp.dot(last_layer[0], x) + last_layer[1]

    # log(exp(o1) / sum(exp(o1), exp(o2),...))
    # a softmax probability
    return logits - logsumexp(logits)

dummy = np.random.randn(784)
pred = MLP_predict(MLP_params, dummy)
print('single data pred', pred)

batched_MLP_pred = jax.vmap(MLP_predict, in_axes=(None, 0))
batch_dummy = np.random.randn(16, 784)
pred = batched_MLP_pred(MLP_params, batch_dummy)
print('batched data pred', pred)
