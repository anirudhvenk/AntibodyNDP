from neural_diffusion_processes.new_process import *
from neural_diffusion_processes.new_model import NewAttentionModel
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from data import CustomImageDataset

import torch

new_attn_model = NewAttentionModel(4, 8, 8, 2, 1)
optimizer = torch.optim.Adam(new_attn_model.parameters())

beta_t = cosine_schedule(3e-4, 0.5, 500)
process = GaussianDiffusion(beta_t)

data = torch.randn((100, 28, 28))
dataset = CustomImageDataset(data, 28, 1)
train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

for e in range(100):
    cum_loss = 0
    for batch in train_dataloader:
        X, Y = batch
        optimizer.zero_grad()
        loss_val = loss(process, new_attn_model, X, Y, None, 500, "l1")
        loss_val.backward()
        optimizer.step()
        cum_loss += loss_val
    print(loss_val/len(train_dataloader))
