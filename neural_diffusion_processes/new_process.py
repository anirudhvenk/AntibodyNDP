import torch

def expand_to(a, b):
    new_shape = a.shape + (1,) * (b.ndim - a.ndim)
    return a.reshape(new_shape)

def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)

    return carry, None

def cosine_schedule(beta_start, beta_end, timesteps, s=0.008, **kwargs):
    x = torch.linspace(0, timesteps, timesteps + 1)
    ft = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = ft / ft[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    return betas * (beta_end - beta_start) + beta_start

class GaussianDiffusion:
    def __init__(self, betas):
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(1.0 - betas, dim=0)

    def pt0(self, y0, t):        
        alpha_bars = expand_to(self.alpha_bars[t[None]][0], y0)
        m_t0 = torch.sqrt(alpha_bars) * y0
        v_t0 = (1.0 - alpha_bars) * torch.ones_like(y0)
        return m_t0, v_t0

    def forward(self, y0, t):
        m_t0, v_t0 = self.pt0(y0, t)
        noise = torch.normal(mean=0, std=1, size=y0.shape)
        yt = m_t0 + torch.sqrt(v_t0) * noise
        return yt, noise

    def ddpm_backward_step(self, noise, yt, t):
        beta_t = expand_to(self.betas[t], yt)
        alpha_t = expand_to(self.alphas[t], yt)
        alpha_bar_t = expand_to(self.alpha_bars[t], yt)

        z = (t > 0) * torch.normal(mean=0, std=1, size=yt.shape, dtype=yt.dtype)

        a = 1.0 / torch.sqrt(alpha_t)
        b = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        yt_minus_one = a * (yt - b * noise) + torch.sqrt(beta_t) * z
        return yt_minus_one

    def ddpm_backward_mean_var(self, noise, yt, t):
        beta_t = expand_to(self.betas[t], yt)
        alpha_t = expand_to(self.alphas[t], yt)
        alpha_bar_t = expand_to(self.alpha_bars[t], yt)

        a = 1.0 / torch.sqrt(alpha_t)
        b = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        m = a * (yt - b * noise)
        v = beta_t * torch.ones_like(yt) * (t > 0)
        v = torch.maximum(v, torch.ones_like(v) * 1e-3)
        return m, v
    
    def conditional_sample(
        self,
        x,
        mask,
        *,
        x_context,
        y_context,
        mask_context,
        model_fn,
        num_inner_steps=5
    ):
        if mask is None:
            mask = torch.zeros_like(x[:, 0])

        if mask_context is None:
            mask_context = torch.zeros_like(x_context[:, 0])

        x_augmented = torch.concatenate([x_context, x], axis=0)
        mask_augmented = torch.concatenate([mask_context, mask], axis=0)
        num_context = len(x_context)

        def repaint_inner(yt_target, t):
            # one step backward: t -> t-1
            yt_context = self.forward(y_context, t)[0]
            # print(yt_context.shape, yt_target.shape)
            y_augmented = torch.concatenate([yt_context, yt_target], axis=0)
            # print("T shape: ", t.shape)
            # print("Y shape: ", y_augmented.shape)
            # print("X shape: ", x_augmented.shape)
            # print("mask shape: ", mask_augmented.shape)
            noise_hat = model_fn(x_augmented.unsqueeze(0), y_augmented.unsqueeze(0), t.unsqueeze(0), mask_augmented.unsqueeze(0)).squeeze(0)
            # print(noise_hat.shape)
            y = self.ddpm_backward_step(noise=noise_hat, yt=y_augmented, t=t)
            y = y[num_context:]
            # one step forward: t-1 -> t
            z = torch.randn(size=y.shape)
            beta__t_minus_1 = expand_to(self.betas[t - 1], y)
            y = torch.sqrt(1.0 - beta__t_minus_1) * y + torch.sqrt(beta__t_minus_1) * z
            return y, None

        def repaint_outer(y, t):
            # loop
            # print(y.shape)
            # print("Timestep: ", t.item())
            ts = torch.ones((num_inner_steps,), dtype=torch.int64) * t
            y, _ = scan(repaint_inner, y, ts)

            # step backward: t -> t-1
            yt_context = self.forward(y_context, t)[0]
            y_augmented = torch.concatenate([yt_context, y], axis=0)
            noise_hat = model_fn(x_augmented.unsqueeze(0), y_augmented.unsqueeze(0), t.unsqueeze(0), mask_augmented.unsqueeze(0)).squeeze(0)
            y = self.ddpm_backward_step(noise=noise_hat, yt=y_augmented, t=t)
            y = y[num_context:]
            return y, None
        
        ts = torch.flip(torch.arange(len(self.betas)), [0])
        yT_target = torch.randn(len(x), y_context.shape[-1])

        y, _ = scan(repaint_outer, yT_target, ts[:-1])
        return y

def loss(process, network, X, Y, mask, num_timesteps, loss_type):
    if loss_type == "l1":
        def loss_metric(a, b):
            return torch.abs(a - b)
    elif loss_type == "l2":
        def loss_metric(a, b):
            return (a - b) ** 2
    else:
        raise ValueError(f"Unknown loss type {loss_type}")

    def loss_fn(t, y, x, mask):
        yt, noise = process.forward(y, t)
        noise_hat = network(x, yt, t[None], mask)
        loss_value = torch.sum(loss_metric(noise, noise_hat), axis=1)  # [N,]
        loss_value = loss_value * (1.0 - mask)
        num_points = len(mask) - torch.count_nonzero(mask)
        return torch.sum(loss_value) / num_points

    batch_size = len(X)
    t = torch.rand(batch_size) * (num_timesteps / batch_size)
    t = t + (num_timesteps / batch_size) * torch.arange(batch_size)
    t = t.to(torch.int64)

    if mask is None:
        # consider all points
        mask_target = torch.zeros_like(X[..., 0])
    else:
        mask_target = mask

    losses = torch.vmap(loss_fn, randomness='same')(t, Y, X, mask_target)
    return torch.mean(losses)
