import torch

def get_image_grid_inputs(size: int, a=2):
    x1 = np.linspace(a, -a, size)
    x2 = np.linspace(-a, a, size)
    x1, x2 = np.meshgrid(x2, x1)
    return np.stack([x1.ravel(), x2.ravel()], axis=-1)


def get_rescale_function_fwd_and_inv(config: Config):
    if config.dataset == "mnist":
        mean = jnp.zeros((1,)).reshape(1, 1)
        std = jnp.ones((1,)).reshape(1, 1)
    elif config.dataset == "celeba32":
        mean = np.array([0.5066832 , 0.4247095 , 0.38070202]).reshape(1, 3)
        std = np.array([0.30913046, 0.28822428, 0.2866247]).reshape(1, 3)
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")
    
    def fwd(y):
        """y: [N, H*W, C] """
        return (y - mean[None]) / std[None]
    
    def inv(y):
        """y: [N, H*W, C] """
        y = y * std[None] + mean[None]
        y = jnp.clip(y, 0., 1.)
        return y
    
    return fwd, inv


def get_image_data(
    config: Config,
    train: bool = True,
    batch_size: int = 1024,
    num_epochs: int = 1,
) -> Dataset:
    size = config.image_size
    output_dim = config.output_dim
    input_ = get_image_grid_inputs(size)
    rescale = get_rescale_function_fwd_and_inv(config)[0]

    def preprocess(batch) -> Mapping[str, ndarray]:
        y = batch["image"]
        batch_size = len(y)
        x = tf.repeat(input_[None, ...], batch_size, axis=0)
        y = tf.image.resize(y[..., None] if output_dim == 1 else y, (size, size))
        y = tf.cast(y, tf.float32) / 255.  # rescale
        y = tf.reshape(y, (batch_size, size*size, output_dim))
        y = rescale(y)
        return dict(
            x_target=x,
            y_target=y,
        )

    images_dataset = datasets.load_dataset(config.hf_dataset_name)
    if "celeba" in config.dataset:
        images_dataset = images_dataset["train"].train_test_split(test_size=0.05)
    images_dataset.set_format('tensorflow')
    images_dataset = images_dataset.select_columns("image")
    subset = "train" if train else "test"
    ds = images_dataset[subset].to_tf_dataset(batch_size=batch_size, shuffle=True, drop_remainder=True)
    if train:
        ds = ds.repeat(count=num_epochs)
    ds = ds.map(preprocess)
    ds = ds.as_numpy_iterator()
    return map(lambda d: Batch(**d), ds)


import matplotlib.pyplot as plt

config: Config = setup_config(Config)
key = jax.random.PRNGKey(config.seed)

KEEP = 33
NOT_KEEP = 44
CONTEXT_TYPES = [
    "horizontal",
    "vertical",
    "10percent",
    "25percent",
    "50percent",
    "75percent",
]

def get_context_mask(key, config: Config, context_type: str = "horizontal") -> ndarray:
    x = get_image_grid_inputs(config.image_size)
    if context_type == "horizontal":
        condition = x[..., 1] > 0.0
    elif context_type == "vertical":
        condition = x[..., 0] < 0.0
    elif "percent" in context_type:
        p = float(context_type[:2]) / 100.
        condition = jax.random.uniform(key, shape=(len(x),)) < p
    else:
        raise ValueError(f"Unknown context type {context_type}")

    return jnp.where(
        condition,
        KEEP * jnp.ones_like(x[..., 0]),
        NOT_KEEP * jnp.ones_like(x[..., 0]),
    )

get_idx_keep = lambda x: jnp.where(x == KEEP, jnp.ones(x.shape, dtype=bool), jnp.zeros(x.shape, dtype=bool))
