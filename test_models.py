from neural_diffusion_processes.model import AttentionModel
from neural_diffusion_processes.new_model import NewAttentionModel
import torch
import jax
import jax.numpy as jnp
from jax import random
import haiku as hk

def forward_fn(X, Y, T):
  mod = AttentionModel(n_layers=4, hidden_dim=8, num_heads=8, output_dim=1, init_zero=False)
  return mod(X, Y, T, None)

seed = 42
key = random.PRNGKey(seed)

# Splitting the key for generating different random numbers
key, subkey1, subkey2 = random.split(key, 3)

forward = hk.transform(forward_fn)
# Generating the random arrays
# X = random.normal(subkey1, (1,3,2))
X = jnp.ones((1,3,2))
# Y = random.normal(subkey2, (1,3,1))
Y = jnp.ones((1,3,1))
T = jnp.asarray([3])
# T = random.randint(key, (1,), 0, 10)
rng = None
params = forward.init(key, X, Y, T)
# print(X)
# print(Y)
# print(T)
print(forward.apply(params, key, X,Y,T))

new_attn_model = NewAttentionModel(4, 8, 8, 1)
X = torch.ones((1,3,2))
# Y = random.normal(subkey2, (1,3,1))
Y = torch.ones((1,3,1))
T = torch.as_tensor([3])
print(new_attn_model(X, Y, T, None))



