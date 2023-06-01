'''
    Training script on v4-8
'''

import functools
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state
import jax
from jax import lax
import jax.numpy as jnp
from jax import random
import ml_collections
import optax

from einops import rearrange

TIME_STEPS=1000

def create_model(*, model_cls, half_precision: bool, **kwargs):
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == 'tpu':
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    
    return model_cls(dtype=model_dtype)

def initialize(key, image_size, model):
    '''
        Init model params
        notice here we don't have dropout, so no need to init dropout with another key
    '''
    input_shape = (1, image_size, image_size, 3)
    @jax.jit
    def init(*args):
        return model.init(*args)
    variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
    return variables['params'], variables['batch_stats']

def create_learning_rate(
    config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int
):
    '''
        create learning rate scheduler
    '''
    warmup_fn = optax.linear_schedule(
        init_value=0.,
        end_value=base_learning_rate,
        transition_steps=config.warmup_epochs * steps_per_epoch
    )

    cos_epoch = max(config.num_epochs - config.warmup_epochs, 1)
    cos_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cos_epoch * steps_per_epoch,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cos_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch]
    )

def train_step(key, diff, state, batch, learning_rate_fn):
    '''
        perform a single training step
    '''
    def loss_fn(params):
        '''
            loss function for training
        '''

        key, noise_key, t_key, q_key = random.split(key, num=4)
        noise = random.normal(noise_key, batch.shape)
        t = random.randint(t_key, (batch.shape[0],), 0, TIME_STEPS)

        noisy_x = diff.q_sample(q_key, batch, t, noise=noise)

        ## custom apply function; usually just model apply
        output, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            noisy_x,
            mutable=['batch_stats']
        )

        loss = jnp.mean((output - noise) ** 2)
        return loss, (new_model_state, output, loss)
    
    step = state.step
    lr = learning_rate_fn(step)

    grad_fn = jax.grad_and_value(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name='batch')
    new_model_state, output, loss = aux[1]
    
    new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats']) # --> auto grad & update state

    return new_state, loss

class TrainState(train_state.TrainState):
    batch_stats: Any

def restore_ckpt(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)

def save_ckpt(state, workdir):
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    step = int(state.step)
    logging.info('Saving checkpoint step %d', step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=3)

# pmeam only works inside pmap; so wrapped it up here
cross_replica_mean = jax.pmap(lambda x: jax.pmean(x, 'x'), 'x')

def create_train_state(
    rng,
    config: ml_collections.ConfigDict,
    model, 
    image_size, 
    learnng_rate_fn
):
    '''
        instantiate initial training state
    '''
    params, batch_stats = initialize(rng, image_size, model)
    tx = optax.sgd(
        learning_rate=learnng_rate_fn,
        momentum=config.momentum,
        nesterov=True,
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )

    return state

