from neural_diffusion_processes.new_process import *
from neural_diffusion_processes.new_model import NewAttentionModel
from torch.utils.data import DataLoader
from data import CustomImageDataset
import torch
import pickle
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

new_attn_model = NewAttentionModel(
                                    n_layers=4, 
                                    hidden_dim=64, 
                                    num_heads=8, 
                                    input_dim=2, 
                                    output_dim=1
                                    )

optimizer = torch.optim.Adam(new_attn_model.parameters(), lr=1e-5)
beta_t = cosine_schedule(3e-4, 0.5, 500)
process = GaussianDiffusion(beta_t)

ema_decay = 0.995
ema_model = AveragedModel(new_attn_model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))

with open('./data/MNIST/mnist.pkl', 'rb') as f:
    mnist_data = pickle.load(f)
    
mnist_data = torch.stack(mnist_data).squeeze(1)
dataset = CustomImageDataset(mnist_data, 28, 1)
train_dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

for e in range(20):
    cum_loss = 0
    for batch in tqdm(train_dataloader):
        X, Y = batch
        optimizer.zero_grad()
        loss_val = loss(process, new_attn_model, X, Y, None, 500, "l1")
        loss_val.backward()
        optimizer.step()
        ema_model.update_parameters(new_attn_model)
        cum_loss += loss_val
        
    print(loss_val/len(train_dataloader))
