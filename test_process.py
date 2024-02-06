from neural_diffusion_processes.process import *
from neural_diffusion_processes.model import AttentionModel
import torch
import jax
import jax.numpy as jnp
from jax import random
import haiku as hk
from functools import partial


schedule: str = "cosine"
beta_start: float = 3e-4
beta_end: float = 0.5
timesteps: int = 500


beta_t = cosine_schedule(3e-4, 0.5, 500)
process = GaussianDiffusion(beta_t)
# print(process.alpha_bars)

@hk.without_apply_rng
@hk.transform
def network(t, y, x, mask):
    model = AttentionModel(
        n_layers=4,
        hidden_dim=8,
        num_heads=8,
        output_dim=1,
        sparse=False,
    )
    return model(x, y, t, mask)

def net(params, t, yt, x, mask, *, key):
    del key  # the network is deterministic
    #NOTE: Network awkwardly requires a batch dimension for the inputs
    return network.apply(params, t[None], yt[None], x[None], mask[None])[0]

def loss_fn(params, batch: Batch, key):
    net_with_params = partial(net, params)
    kwargs = dict(num_timesteps=500, loss_type="l1")
    return loss(process, net_with_params, batch, key, **kwargs)

X = jnp.ones((1,3,2))
Y = jnp.ones((1,3,1))
T = jnp.asarray([3])

state_key = random.PRNGKey(42)
key, init_rng = jax.random.split(state_key)
t = 1. * jnp.zeros((X.shape[0]))
initial_params = network.init(
    init_rng, t=T, y=Y, x=X, mask=None
)

print(initial_params)

new_key, loss_key = jax.random.split(state_key)
batch = Batch(x_target=X, y_target=Y)

loss_and_grad_fn = jax.value_and_grad(loss_fn)
loss_value, grads = loss_and_grad_fn(initial_params, batch, loss_key)
# print(loss_value)
# print(grads)
# print(loss_and_grad_fn)

# print(beta_t)
# print(process)