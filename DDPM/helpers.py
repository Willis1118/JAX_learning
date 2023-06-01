'''
    Helper functions for implementing the U-Net
'''

import flax
from flax import linen as nn

## import python built-in functions
from inspect import isfunction
from functools import partial
from typing import Any, Callable, Sequence, Optional

def exists(x):
    '''
        check if x is defined
    '''
    return x is not None

def default(val, d):
    '''
        return val if exists else d
    '''
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    fn: Callable # anything with __call__ method

    def __call__(self, x, *args, **kwargs):
        '''
            define the res connection
        '''
        return self.fn(x, *args, **kwargs) + x

class PreNorm(nn.Module):
    fn: Callable

    '''
        Apply Group Norm before fn
    '''

    @nn.compact
    def __call__(self, x):
        x = nn.GroupNorm(1)(x)
        return self.fn(x)

def Upsample(dim):

    '''
        input: (dim, H, W)
        output: 
            (dim,
             (H-1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1),
             (W-1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1),
            )
    '''

    return nn.ConvTranspose(
        features=dim,
        kernel_size=(4,4),
        strides=(2,2),
        padding=(2,2) # --> out_padding
    )

def Downsample(dim):
    return nn.Conv(
        features=dim,
        kernel_size=(4,4),
        strides=2,
        padding=1
    )





