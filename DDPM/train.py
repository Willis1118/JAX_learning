'''
    Training script on v4-8
'''

import functools
import time
from typing import Any
import random as _random

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
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from einops import rearrange

from torchloader_util import build_transform
from model import UNet
from torch2jax import parse_batch
from diffusion import Diffuser

TIME_STEPS=1000

def seed_worker(worker_id, global_seed, offset_seed=0):
    # worker_seed = torch.initial_seed() % 2**32 + jax.process_index() + offset_seed
    worker_seed = (global_seed + worker_id +
                   jax.process_index() + offset_seed) % 2**32
    np.random.seed(worker_seed)
    _random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def rebuild_data_loader_train(dataset_train, sampler_train, local_batch_size, config, offset_seed):
    rng_torch = torch.Generator()
    rng_torch.manual_seed(offset_seed)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=local_batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=rng_torch,
        worker_init_fn=functools.partial(
            seed_worker, offset_seed=offset_seed, global_seed=config.seed_pt),
        persistent_workers=True,
        timeout=1800.,
    )
    return data_loader_train

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
    def init(*args, **kwargs):
        return model.init(*args, **kwargs)
    variables = init({'params': key}, jnp.ones(input_shape), time=jnp.ones((input_shape[0],)))
    return variables['params']

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

    return schedule_fn

def train_step(key, state, batch, learning_rate_fn):
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

        noisy_x = Diffuser().q_sample(q_key, batch, t, noise=noise)

        ## custom apply function; usually just model apply
        output, new_model_state = state.apply_fn(
            {'params': params},
            noisy_x,
        )

        loss = jnp.mean((output - noise) ** 2)
        return loss, (new_model_state, output, loss)
    
    step = state.step
    lr = learning_rate_fn(step)

    grad_fn = jax.grad_and_value(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name='batch')
    new_model_state, output, loss = aux[1]
    loss = lax.pmean(loss, axis_name='batch')
    
    ## apply updates to params
    new_state = state.apply_gradients(grads=grads) # --> auto grad & update state

    return new_state, loss

def restore_ckpt(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)

def save_ckpt(state, workdir):
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    step = int(state.step)
    logging.info('Saving checkpoint step %d', step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=3)

# pmeam only works inside pmap; so wrapped it up here
# this function will take average of inputs across all devices
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
    params = initialize(rng, image_size, model)
    tx = optax.sgd(
        learning_rate=learnng_rate_fn,
        momentum=config.momentum,
        nesterov=True,
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

    return state

def main():
    n_devices = jax.local_device_count()
    config = ml_collections.ConfigDict()
    config.momentum = 0
    config.num_workers = n_devices
    config.seed_pt = 42
    config.image_size = 32
    config.batch_size = 128
    config.momentum = 0
    config.dim = 128
    config.warmup_epochs = 5
    config.num_epochs = 20

    base_learning_rate = 0.001 * config.batch_size / 256

    transform = build_transform(config.image_size)
    train_dataset = CIFAR10(root='train_cifar', train=True, transform=transform, download=True)
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=jax.process_count(),
        rank=jax.process_index(),
        shuffle=True,
        seed=config.seed_pt,
    )

    print("Initializing...This might take a while...")

    model = UNet(config.dim)

    learning_rate_fn = create_learning_rate(config, base_learning_rate, 60)

    state_key, training_key = random.split(random.PRNGKey(config.seed_pt))

    state = create_train_state(state_key, config, model, config.image_size, learning_rate_fn)
    step_offset = int(state.step)

    train_loader = rebuild_data_loader_train(
        train_dataset, train_sampler, config.batch_size // jax.process_count(), config, offset_seed=step_offset)
    
    batch = next(iter(train_loader))
    batch = parse_batch(batch)

    state = jax_utils.replicate(state)
    training_key = jax_utils.replicate(training_key)

    print(batch['images'].shape)

    p_train_step = jax.pmap(
        functools.partial(train_step, learning_rate_fn=learning_rate_fn),
        axis_name='batch'
    )

    for epoch in range(1):
        print(f'Begin Trainning on epoch{epoch}')
        for batch in train_loader:
            batch = parse_batch(batch)
            state, metrics = p_train_step(training_key, state, batch['images'])

            print(metrics)
    
    ## wait until all computation on XLA side is done
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

if __name__ == '__main__':
    main()

