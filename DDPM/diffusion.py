'''
    Diffusion definitions go here
'''

import jax
from jax import lax, random, numpy as jnp
from flax import jax_utils

from functools import partial
from tqdm.auto import tqdm

import numpy as np
import functools

## custom modules
from model import UNet

class VarScheduler:
    def __init__(self, timesteps=1000):
        self.time = timesteps
    
    def cos_beta(self, s=0.008):
        steps = self.time + 1
        x = jnp.linspace(0, self.time, steps)
        alphas_cumprod = jnp.cos( (( x / self.time ) + s) / (1 + s) * jnp.pi * 0.5 ) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        return jnp.clip(betas, 0.0001, 0.9999)

    def linear_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.02
        return jnp.linspace(beta_start, beta_end, self.time)

    def quadratic_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.02
        return jnp.linspace(beta_start**0.5, beta_end**0.5, self.time) ** 2
    
    def sigmoid_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.02
        betas = jnp.linspace(-6, 6, self.time)
        sig = 1 / (1 + jnp.exp(-betas))
        return sig * (beta_end - beta_start) + beta_start
    
class Diffuser:
    
    def __init__(self, timesteps=1000):
        self.time = timesteps
        scheduler = VarScheduler(timesteps)
        self.betas = scheduler.linear_beta_schedule()
        self.alphas = 1. - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = jnp.pad(self.alphas_cumprod[:-1], (1,0), constant_values=1.0)
        self.sqrt_recip_alphas = jnp.sqrt(1.0 / self.alphas)

        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    @staticmethod
    def extract(a, t, x_shape):
        batch = x_shape[0]
        out = np.take_along_axis(a, t, axis=-1)
        return jnp.array(np.reshape(out, (batch, *((1,) * (len(x_shape) - 1)))))
    
    # @partial(jax.jit, static_argnums=(4,))
    def q_sample(self, key, x_start, t, noise=None):
        '''
            forward diffusion process
        '''

        if noise is None:
            noise = jax.random.normal(key, x_start.shape)
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minue_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_start.shape
        )

        ## add noise to image
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minue_alphas_cumprod_t * noise

    # @partial(jax.jit, static_argnums=(4,5))
    def p_losses(self, key, denoise_model, x_start, t, noise=None, loss_type="l1"):
        '''
            evaluate the inference on diffusion mean
        '''

        if noise is None:
            noise = jax.random.normal(key, x_start.shape)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = jnp.mean(abs(noise - predicted_noise))
        elif loss_type == 'l2':
            loss = jnp.mean((noise - predicted_noise) ** 2)
        else:
            raise NotImplementedError()
        
        return loss
    
    # @partial(jax.jit, static_argnums=(5,))
    def p_sample(self, key, params, x, t, t_index):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * UNet(dim=128).apply(params, x, time=t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = random.normal(key, x.shape)
            # Algorithm 2 line 4:
            return model_mean + jnp.sqrt(posterior_variance_t) * noise 
    
    # @partial(jax.jit, static_argnums=(3,))
    def p_sample_loop(self, key, params, shape):

        params = jax.lax.stop_gradient(params)

        n, b = shape[1], shape[0]
        key, noise_key = random.split(key)
        img = random.normal(noise_key, shape)

        pp_sample=jax.pmap(
            self.p_sample,
            axis_name='batch',
            static_broadcasted_argnums=(4,)
        )

        print('Sample Shape: ', shape)

        imgs = []

        # sample_key = jax_utils.replicate(sample_key)

        for i in tqdm(reversed(range(0, self.time)), desc='sampling loop time step', total=self.time):
            key, sample_key = random.split(key)
            img = self.p_sample(sample_key, params, img, jnp.full((b), i, dtype=jnp.int32), i)
            imgs.append(jax.device_get(img))
        
        return imgs
    
    # @partial(jax.jit, static_argnums=(3,4,5))
    def sample(self, key, model, params, image_size, batch_size=16, channels=3):
        shape = (batch_size, image_size, image_size, channels)
        return self.p_sample_loop(key, model, params, shape)


    
