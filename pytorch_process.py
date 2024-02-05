from neural_diffusion_processes.new_process import *
from neural_diffusion_processes.new_model import NewAttentionModel
import torch

new_attn_model = NewAttentionModel(4, 8, 8, 1)
X = torch.ones((2,3,2))
Y = torch.ones((2,3,1))
# T = torch.as_tensor([3])

beta_t = cosine_schedule(3e-4, 0.5, 500)
process = GaussianDiffusion(beta_t)

loss(process, new_attn_model, X, Y, None, 500, "l1")

# print(process.alpha_bars)

