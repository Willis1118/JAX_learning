'''
    Defining model block
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

from einops import rearrange

## python importing
import math
from typing import Callable, Tuple
from functools import partial

## custom module importing
from helpers import exists, default, Residual, PreNorm, Downsample, Upsample

### Time Positional Embedding ###
class PositionalEmbedding(nn.Module):
    dim: int

    def __call__(self, time):
        '''
            sin pos embedding
            input: (B, 1) time tensor
            output: (B, dim) embeddings
        '''

        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        # embeddings = jnp.inner(time, embeddings)
        embeddings = jnp.concatenate((jnp.sin(embeddings), jnp.cos(embeddings)), axis=-1)
        return embeddings

class Block(nn.Module):
    dim: int
    groups: int

    '''
        A Basic GroupNorm Block
    '''

    @nn.compact
    def __call__(self, x, *, scale_shift=None):

        print(x.shape)

        x = nn.Conv(
            features=self.dim,
            kernel_size=(3,3),
            padding=1,
        )(x)
        x = nn.GroupNorm(
            num_groups=self.groups,
        )(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        
        return nn.activation.silu(x)

class ResNetBlock(nn.Module):
    dim_out: int
    emb_dim: int
    groups: int

    '''
        ResNet Block
    '''
    @nn.compact
    def __call__(self, x, time_emb=None):
        h = Block(self.dim_out, self.groups)(x)

        print(h.shape)

        if exists(time_emb):
            time_emb = nn.Dense(features=self.dim_out)(nn.activation.silu(time_emb))
            h = rearrange(time_emb, 'b c -> b 1 1 c') + h
        
        h = Block(self.dim_out, self.groups)(x)
        
        if x.shape[-1] != h.shape[-1]:
            h += nn.Conv(self.dim_out, (1,1))(x)
        else:
            h += x
        
        return h

class Attention(nn.Module):
    dim: int
    heads: int = 4
    dim_head: int = 32

    '''
        Attention Module
    '''
    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        hidden = self.heads * self.dim_heads
        scale = self.dim_head ** -0.5
        qkv = nn.Conv(
            features=hidden * 3,
            kernel_size=(1,1),
            use_bias=False
        )(x).split(3, axis=-1)

        '''
            Attention process
        '''

        q, k, v = map(
            lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), qkv
        )

        q = q * scale
        esum = jnp.einsum('b h d i, b h d j -> b h i j', q, k)
        esum -= jnp.amax(esum, axis=-1, keepdims=True)
        attn = jax.nn.softmax(esum, axis=-1)

        out = jnp.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b x y (h d)', x=h, y=w)
        return nn.Conv(self.dim, (1,1))(out)
    
class LinearAttention(nn.Module):
    dim: int
    heads: int = 4
    dim_head: int = 32

    @nn.compact
    def __call__(self, x):
        b, h, w, c= x.shape
        hidden = self.heads * self.dim_head
        scale = self.dim_head ** -0.5
        qkv = nn.Conv(
            features=hidden * 3,
            kernel_size=(1,1),
            use_bias=False
        )(x).split(3, axis=-1)

        '''
            Attention process
        '''

        q, k, v = map(
            lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), qkv
        )

        q = jax.nn.softmax(q, axis=-2)
        k = jax.nn.softmax(k, axis=-1)

        q = q * scale

        context = jnp.einsum('b h d n, b h e n -> b h d e', k, v)

        out = jnp.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b x y (h c)', h=self.heads, x=h, y=w)

        return nn.Conv(self.dim, (1,1))(nn.GroupNorm(1)(out))

class UNet(nn.Module):
    dim: int
    init_dim: int = None
    out_dim: int = None
    dim_mults: Tuple[int] = (1,2,4,8)
    channels: int = 3
    with_time_emb: bool = True
    resnet_block_groups: int = 8
    
    @nn.compact
    def __call__(self, x, time):
        b, c, h, w = x.shape

        init_dim = default(self.init_dim, self.dim // 3 * 2)
        init_conv = nn.Conv(init_dim, (7,7), padding=3)

        dims = [init_dim, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_class = partial(ResNetBlock, groups=self.resnet_block_groups)

        if self.with_time_emb:
            time_dim = self.dim * 4
            time_mlp = nn.Sequential(
                [PositionalEmbedding(self.dim),
                nn.Dense(time_dim),
                nn.activation.gelu,
                nn.Dense(time_dim)]
            )
        else:
            time_mlp = None

        x = init_conv(x)
        t = time_mlp(time)
        h = []

        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= len(in_out) - 1

            x = block_class(dim_out=dim_out, emb_dim=time_dim)(x, t)
            x = block_class(dim_out=dim_out, emb_dim=time_dim)(x, t)

            x = Residual(PreNorm(LinearAttention(dim_out)))(x)

            h.append(x)

            if not is_last:
                x = Downsample(dim_out)(x)
            
        mid_dim = dims[-1]
        x = block_class(mid_dim, time_dim)(x, t)
        x = Residual(PreNorm(LinearAttention(mid_dim)))(x)
        x = block_class(mid_dim, time_dim)(x, t)

        for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            x = jnp.concatenate((x, h.pop()), axis=-1) # concate on channels

            is_last = i >= len(in_out) - 1

            x = block_class(dim_out=dim_in, emb_dim=time_dim)(x, t)
            x = block_class(dim_out=dim_in, emb_dim=time_dim)(x, t)

            x = Residual(PreNorm(LinearAttention(dim_in)))(x)

            if not is_last:
                x = Upsample(dim_in)(x)
    
        out_dim = default(self.out_dim, self.channels)
        x = block_class(self.dim, time_dim)(x, t)
        x = nn.Conv(out_dim, (1,1))(x)

        return x

if __name__ == '__main__':
    model = UNet(dim=128)

    x_key, t_key, init_key = random.split(random.PRNGKey(0), num=3)

    x = random.normal(x_key, (10,256,256,3))
    t = random.uniform(t_key, (10,))

    y, params = model.init_with_output(init_key, x, time=t)

    print(jax.tree_map(jnp.shape, params))
    print(y.shape)




        


