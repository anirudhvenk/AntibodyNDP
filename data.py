import torch
from torch.utils.data import Dataset
# import tensorflow as tf
import numpy as np

def get_image_grid_inputs(size, a=2):
    x1 = torch.linspace(a, -a, size)
    x2 = torch.linspace(-a, a, size)
    x1, x2 = torch.meshgrid(x2, x1)
    return torch.stack([x1.ravel(), x2.ravel()], axis=-1)


# def get_image_grid_inputs(size, a):
#     x1 = np.linspace(a, -a, size)
#     x2 = np.linspace(-a, a, size)
#     x1, x2 = np.meshgrid(x2, x1)
#     return np.stack([x1.ravel(), x2.ravel()], axis=-1)

# input_ = get_image_grid_inputs(28, 2)
# x = tf.repeat(input_[None, ...], 100, axis=0)
# print(x)

# def get_rescale_function_fwd_and_inv(config: Config):
#     mean = torch.zeros((1,)).reshape(1, 1)
#     std = torch.ones((1,)).reshape(1, 1)
    
#     def fwd(y):
#         """y: [N, H*W, C] """
#         return (y - mean[None]) / std[None]
    
#     def inv(y):
#         """y: [N, H*W, C] """
#         y = y * std[None] + mean[None]
#         y = torch.clip(y, 0., 1.)
#         return y
    
#     return fwd, inv


# def get_image_data(
#     config: Config,
#     train: bool = True,
#     batch_size: int = 1024,
#     num_epochs: int = 1,
# ) -> Dataset:
#     size = config.image_size
#     output_dim = config.output_dim
#     input_ = get_image_grid_inputs(size)
#     rescale = get_rescale_function_fwd_and_inv(config)[0]

#     def preprocess(batch) -> Mapping[str, ndarray]:
#         y = batch["image"]
#         batch_size = len(y)
#         x = tf.repeat(input_[None, ...], batch_size, axis=0)
#         y = tf.image.resize(y[..., None] if output_dim == 1 else y, (size, size))
#         y = tf.cast(y, tf.float32) / 255.  # rescale
#         y = tf.reshape(y, (batch_size, size*size, output_dim))
#         y = rescale(y)
#         return dict(
#             x_target=x,
#             y_target=y,
#         )

#     images_dataset = datasets.load_dataset(config.hf_dataset_name)
#     if "celeba" in config.dataset:
#         images_dataset = images_dataset["train"].train_test_split(test_size=0.05)
#     images_dataset.set_format('tensorflow')
#     images_dataset = images_dataset.select_columns("image")
#     subset = "train" if train else "test"
#     ds = images_dataset[subset].to_tf_dataset(batch_size=batch_size, shuffle=True, drop_remainder=True)
#     if train:
#         ds = ds.repeat(count=num_epochs)
#     ds = ds.map(preprocess)
#     ds = ds.as_numpy_iterator()
#     return map(lambda d: Batch(**d), ds)



class CustomImageDataset(Dataset):
    def __init__(self, data, size, output_dim, a=2):
        input_ = get_image_grid_inputs(size, a).unsqueeze(0)
        mean = torch.zeros((1,)).reshape(1, 1) # To change
        std = torch.ones((1,)).reshape(1, 1) # To change
        
        self.X = torch.repeat_interleave(input_, len(data), dim=0)
        self.Y = data.reshape(len(data), size*size, output_dim)        
        self.Y = self.Y / 255
        self.Y = (self.Y - mean[None]) / std[None]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]