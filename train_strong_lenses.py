from matplotlib.image import imread
from torchvision import datasets, transforms
def custom_loader_16bit(path):
    with open(path, 'rb') as f:
        img = imread(f)
    # Augmentation
    if random.choice([0,1]):
        img = np.flip(img, axis=1)
    if random.choice([0,1]):
        img = np.flip(img, axis=0)
    if random.choice([0,1]):
        img = img.T
    img = transforms.ToTensor()(img.copy())
    img = transforms.CenterCrop(image_size)(img)
    img = transforms.Normalize(mean=[0.5], std=[0.5])(img)
    return img

import numpy as np
import torch

from torch import optim
import copy
from tqdm import tqdm
import torch.nn as nn
import torchvision.utils as vutils

import wandb
import json

def load_api_key():
    with open('~/.wandb_api.json') as f:
        data = json.load(f)
        api_key = data['api_key']
    return api_key

data_dir = "~/data/strong_lenses/"
result_dir = "~/results/strong_lenses/"

run_name = "test"
epochs = 50000
steps_eval = 100  ## number of epochs between evals
batch_size = 2
image_size = 64
device = "cuda"
lr = 3e-4
noise_steps = 1000
num_classes = 1
log_wandb = True

lr_step_epochs = 200
lr_step = 0.5
lr_min = 1e-7




from diffusion_galaxy.unet import UNet_conditional
from diffusion_galaxy.update_parameters import EMA
from diffusion_galaxy.diffusion import Diffusion_cond
from diffusion_galaxy.dataset import create_pytorch_dataloader
from diffusion_galaxy.evaluate import psnr



device = device
unet = UNet_conditional(image_size=image_size, num_classes=num_classes, device=device).to(device)
optimizer = optim.AdamW(unet.parameters(), lr=lr)
diffusion = Diffusion_cond(img_size=image_size, device=device, noise_steps=noise_steps)
ema = EMA(0.995)
ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
dataloader = create_pytorch_dataloader(data_dir, image_size, custom_loader=custom_loader_16bit, batch_size=batch_size, num_workers=2)

mse = nn.MSELoss()


def training(epochs, steps_eval=1):
    if log_wandb:
        wandb_api = load_api_key()
        wandb.login(key=wandb_api)
        wandb.init(project="DDPM_strong_lenses", name="test")

    unet.train()
    losses = {"psnr": [], "mse": []}
    for epoch in range(epochs):
        for images, labels in tqdm(dataloader, desc=f"epoch {epoch+1}"):
            training_step(images, labels)
        plot = not epoch % steps_eval
        psnr_val, mse_val = evaluate(epoch, plot=plot)
        losses["psnr"].append(psnr_val)
        losses["mse"].append(mse_val)
        torch.save(unet.state_dict(), f"unet_{run_name}.pt")
        if not (epoch+1) % lr_step_epochs:
            global lr
            global optimizer
            if lr > lr_min:
                lr = lr * lr_step
                print(lr)
                optimizer = optim.AdamW(unet.parameters(), lr=lr)
    if log_wandb:
        wandb.finish()
    return losses

def training_step(images, labels):
    """ train diffusion model on batch of samples and associated labels."""
    images = images.to(device)
    labels = labels.to(device)
    predicted_noise, noise = predict_noise(images, labels)
    loss = mse(noise, predicted_noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ema.step_ema(ema_model, unet)

def predict_noise(images, labels, p=0.0):
    """ predict noise from images and labels
    set p=0 for unconditional training"""
    images = images.to(device)
    labels = labels.to(device)
    t = diffusion.sample_timesteps(images.shape[0]).to(device)
    x_t, noise = diffusion.noise_images(images, t)
    if np.random.random() < p:
        labels = None
    predicted_noise = unet(x_t, t, labels)
    return predicted_noise, noise

@torch.no_grad()
def evaluate(epoch, plot=False):
    """ evaluate the diffusion model """
    unet.eval()
    psnr_val = 0.0
    mse_val = 0.0
    for i, (images, labels) in enumerate(dataloader):
        predicted_noise, noise = predict_noise(images, labels)
        psnr_val += psnr(predicted_noise, noise, torch.max(predicted_noise))
        mse_val += mse(predicted_noise, noise)
    if plot:
        images = diffusion.sample(unet, 16, None, cfg_scale=0)
#        save_images(images, f"epoch {epoch}", f"generated_images{run_name}_{epoch}.png")
        if log_wandb:
    #        images = (255*(images+1)/2).to(torch.uint8)
            # create a grid of images
            print(images.shape, images.min(), images.max())
            grid = vutils.make_grid(images, nrow=4)
            # convert the grid to a numpy array
            grid_np = grid.permute(1,2,0).cpu().numpy()
            # create a wandb Image object from the numpy array
            image_grid = wandb.Image(grid_np)
            image_grid = wandb.Image(image_grid)
            wandb.log({"Generated Images": image_grid})
    psnr_val /= i + 1
    mse_val /= i + 1
    if log_wandb:
        wandb.log({"PSNR": psnr_val.item(), "MSE": mse_val.item()})
    return psnr_val, mse_val



def save_images(images, title, path):
    """ plot several images and save them to path """
    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i, 0].cpu(), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.suptitle(title)
    plt.savefig(path)
    plt.show()
    plt.close()


training(epochs, steps_eval=steps_eval)