'''
    In this file we will implementing a fully configured CNN for MNIST
'''

import jax
from jax import lax, random, numpy as jnp

# NN lib built on top of JAX developed
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax.training import train_state

import optax

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import functools
from typing import Any, Callable, Sequence, Optional

import numpy as np

'''
    A simple CNN network
'''

class CNN(nn.Module):
    out: int = 10

    @nn.compact
    def __call__(self, x):
        hidden = [32, 64, 256]
        for i, layer in enumerate(hidden):
            print(x.shape)
            if i != len(hidden) - 1:
                x = nn.Conv(features=layer, kernel_size=(3,3))(x)
                x = nn.relu(x)
                x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2))
            else:
                x = x.reshape((x.shape[0], -1)) # flatten
                x = nn.Dense(layer)(x) # linear layer
                x = nn.relu(x)
        
        x = nn.Dense(self.out)(x)

        return nn.log_softmax(x)
    
def to_flatten_nparr(x):
    '''
        transform PILImage to nparray and normalize to [-1, 1]
    '''
    return np.expand_dims(np.array(x, dtype=np.float32), axis=2) / 255

def collate_fn(data):
    '''
        input: data -> [(image, label)] * batch_size
    '''

    # will turn [(image, lable)] --> ([images], [labels])
    # zip ((,),(,),(,)...) will take the first and second element in every tuple and zip into a tuple of two lists
    transposed_data = list(zip(*data))

    images, labels = np.stack(transposed_data[0]), np.array(transposed_data[1])

    return images, labels

batch_size = 32
train_dataset = MNIST(root='train_mnist', train=True, download=True, transform=to_flatten_nparr)
test_dataset = MNIST(root='test_mnist', train=False, download=True, transform=to_flatten_nparr)
mnist_image_size = [28, 28, 1]

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True # --> drop the last 
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=True
)

test_images = np.expand_dims(jnp.array(test_dataset.data), axis=3)
test_labels = jnp.array(test_dataset.targets)

@jax.jit
def train_step(state, imgs, gt_labels):
    def loss_fn(params):
        logits = CNN().apply({'params': state.params}, imgs)
        one_hot_gt_labels = jax.nn.one_hot(gt_labels, num_classes=10)
        loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
        return loss, logits
    
    (_, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads) # entire update
    metrics = compute_metrics(logits=logits, gt_labels=gt_labels)
    return state, metrics

@jax.jit
def eval_step(state, imgs, gt_labels):
    logits = CNN().apply({'params': state.params}, imgs)
    return compute_metrics(logits=logits, gt_labels=gt_labels)

def train_one_epoch(state, dataloader, epoch):
    batch_metrics = []
    for cnt, (imgs, labels) in enumerate(dataloader):
        state, metrics = train_step(state, imgs, labels)
        batch_metrics.append(metrics)

    batch_metrics_np = jax.device_get(batch_metrics) # pull from TPU to CPU

    ## accumulate metrics from all batches
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics])
        for k in batch_metrics_np[0]
    }

    return state, epoch_metrics_np

def eval_model(state, test_imgs, test_labels):
    '''
        evaluate on test dataset
    '''
    metrics = eval_step(state, test_imgs, test_labels)
    metrics = jax.device_get(metrics)
    metrics = jax.tree_map(lambda x: x.item(), metrics)
    return metrics

def create_train_state(key, lr, momentum):
    '''
        maintaining training state
    '''
    cnn = CNN()
    params = cnn.init(key, jnp.ones([1, *mnist_image_size]))['params']
    sgd_opt = optax.sgd(lr, momentum)
    ## TrainState is a simple built-in wrapper class that makes things a bit cleaner
    ## enabling apply gradient on params & sgd_opt
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=sgd_opt)

def compute_metrics(*, logits, gt_labels):
    '''
        compute cross entropy loss
    '''
    one_hot_labels = jax.nn.one_hot(gt_labels, num_classes=10)

    loss = -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))
    acc = jnp.mean(jnp.argmax(logits, -1) == gt_labels)

    metrics = {
        'loss': loss,
        'acc': acc
    }

    return metrics

seed = 0
lr = 0.1
momentum = 0.9
num_epochs = 2
batch_size = 32

train_state = create_train_state(jax.random.PRNGKey(seed), lr, momentum)

for epoch in range(1, num_epochs + 1):
    train_state, train_metrics = train_one_epoch(train_state, train_loader, epoch)
    print(f"Train epoch: {epoch}, loss: {train_metrics['loss']}, accuracy: {train_metrics['acc']}")

    test_metrics = eval_model(train_state, test_images, test_labels)
    print(f"Train epoch: {epoch}, loss: {test_metrics['loss']}, accuracy: {test_metrics['acc']}")

            
