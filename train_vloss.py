from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import time
import torchvision

results_path = "checkpoints/diffusion"

fmnist = torchvision.datasets.FashionMNIST(
    'data/', train=True, download=True)#, transform = torchvision.transforms.ToTensor()

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4),
    channels=1,
    full_attn = (False, False, True), # use full attention on the final U-Net layer.
    flash_attn=True,  # use attention on all U-Net levels.
)

diffusion = GaussianDiffusion(
    model,
    image_size=28,
    timesteps=1000,  # number of steps
    # sampling_timesteps = 250,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    objective="pred_v",
)

trainer = Trainer(
    diffusion,
    "data/",
    train_batch_size=64,
    train_lr=1e-3,
    train_num_steps=12000,  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=False,                       # turn on mixed precision
    logger_tags=["debug"],
    calculate_fid=False,  # whether to calculate fid during training
    results_folder=results_path + time.strftime("%Y%m%d_%H%M%S")
)

if __name__ == "__main__":
    trainer.train()
