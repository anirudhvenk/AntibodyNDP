import torch
from neural_diffusion_processes.new_model import NewAttentionModel
from neural_diffusion_processes.new_process import *
from data import CustomImageDataset, get_image_grid_inputs
import pickle
from torch.utils.data import DataLoader

new_attn_model = NewAttentionModel(
                                    n_layers=4, 
                                    hidden_dim=64, 
                                    num_heads=8, 
                                    input_dim=2, 
                                    output_dim=1
                                    )

beta_t = cosine_schedule(3e-4, 0.5, 500)
process = GaussianDiffusion(beta_t)

with open('./data/MNIST/mnist.pkl', 'rb') as f:
    mnist_data = pickle.load(f)
    
mnist_data = torch.stack(mnist_data[20000:20500]).squeeze(1)
dataset = CustomImageDataset(mnist_data, 28, 1)
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

KEEP = 33  # random number
NOT_KEEP = 44  # random number

def get_context_mask(image_size, context_type):
    x = get_image_grid_inputs(image_size)
    if context_type == "horizontal":
        condition = x[..., 1] > 0.0
    elif context_type == "vertical":
        condition = x[..., 0] < 0.0
    elif isinstance(context_type, float):
        p = context_type
        condition = torch.randn(size=(len(x),)) <= p
    else:
        raise ValueError(f"Unknown context type {context_type}")

    return torch.where(
        condition,
        KEEP * torch.ones_like(x[..., 0]),
        NOT_KEEP * torch.ones_like(x[..., 0]),
    )
    
# def sample_conditional():
    # x = torch.linspace(-2, 2, 57)[:, None]
    # xc = jnp.array([-1., 0., 1.]).reshape(-1, 1)
    # yc = jnp.array([0., -1., 1.]).reshape(-1, 1)

percentage = 0.5
get_idx_keep = lambda x: torch.where(x == KEEP, torch.ones(x.shape, dtype=bool), torch.zeros(x.shape, dtype=bool))
context_mask = get_context_mask(28, "horizontal")
num_context_points = torch.where(context_mask == KEEP, torch.ones_like(context_mask), torch.zeros_like(context_mask)).sum()
batch0 = next(iter(test_dataloader))

with torch.no_grad():
    samples = process.conditional_sample(
        x=batch0[0][0],
        mask=None,
        x_context=batch0[0][:, get_idx_keep(context_mask)][0],
        y_context=batch0[1][:, get_idx_keep(context_mask)][0],
        mask_context=torch.zeros_like(batch0[0][:, get_idx_keep(context_mask)][..., 0][0]),
        model_fn=new_attn_model
    )
