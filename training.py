import os

import numpy as np
import torch

from torch import optim
import copy
from tqdm import tqdm
import torch.nn as nn

import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.run_name = "DDPM_strong_lenses_test"
args.epochs = 1
args.batch_size = 12
args.image_size = 64
args.device = "cuda"
args.lr = 3e-4
args.num_classes = 0

from unet import UNet_conditional
from update_parameters import EMA
from diffusion import Diffusion_cond
from dataset import create_pytorch_dataloader
from evaluate import psnr

root = "~/PycharmProjects/diffusion/"
path_to_images = root+"data_strong_lenses"
batch_size = 64
loader = create_pytorch_dataloader(path_to_images, args.image_size, batch_size=args.batch_size, shuffle=True, num_workers=4)

device = args.device
unet = UNet_conditional(num_classes=args.num_classes, image_size=int(64)).to(device)
optimizer = optim.AdamW(unet.parameters(), lr=args.lr)
diffusion = Diffusion_cond(img_size=args.image_size, device=device)
ema = EMA(0.995)
ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

mse = nn.MSELoss()


def training(epochs):
    unet.train()
    losses = {"psnr": [], "mse": []}
    for epoch in range(epochs):
        for images, labels in tqdm(loader):
            training_step(images, labels)
        psnr_val, mse_val = evaluate(epoch, plot=True)
        losses["psnr"].append(psnr_val)
        losses["mse"].append(mse_val)
        torch.save(unet.state_dict(), f"models/unet_{args.run_name}.pt")
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
    for i, (images, labels) in enumerate(loader):
        predicted_noise, noise = predict_noise(images, labels)
        psnr_val += psnr(predicted_noise, noise, torch.max(predicted_noise))
        mse_val += mse(predicted_noise, noise)
    if plot:
        images = diffusion.sample(unet, 16, None, cfg_scale=0)
        save_images(images, f"epoch {epoch}", root+f"output/generated_images{args.run_name}_{epoch}.png")
    psnr_val /= i + 1
    mse_val /= i + 1
    return psnr_val, mse_val


def save_images(images, title, path):
    """ plot several images and save them to path """
    plt.figure(figsize=(10, 10))
    for i in range(images.shape[0]):
        plt.subplot(8, 8, i + 1)
        plt.imshow(images[i, 0], cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.savefig(path)
    plt.close()

training(args.epochs)