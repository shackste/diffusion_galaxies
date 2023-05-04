

import torch
from tqdm import tqdm
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion_cond:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps # timestesps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps) # linear variance schedule as proposed by Ho et al 2020

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ # equation in the paper from Ho et al that describes the noise processs

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample_timesteps_sde(self, n):
        return torch.rand(n) ## put on device???


    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval() # evaluation mode
        with torch.no_grad(): # algorithm 2 from DDPM
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): # reverse loop from T to 1
                t = (torch.ones(n) * i).long().to(self.device) # create timesteps tensor of length n
                predicted_noise = model(x, t, labels) # feed into the model
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale) #linear interpolation out = start + w*(end-start)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None] # this is noise, created in one
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train() # it goes back to training mode
        x = (x.clamp(-1, 1) + 1) / 2 # to be in [-1, 1], the plus 1 and the division by 2 is to bring back values to [0, 1]
        x = (x * 255).type(torch.uint8) # to bring in valid pixel range
        return x

    def sample_sde(self, model, n, labels, cfg_scale=3):
        model.eval() # evaluation mode
        with torch.no_grad(): # algorithm 2 from DDPM
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): # reverse loop from T to 1
                ## !!! let this go from 1 to 0 instead
                t = (torch.ones(n) * i).long().to(self.device) # create timesteps tensor of length n
                predicted_noise = model(x, t, labels) # feed into the model
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale) #linear interpolation out = start + w*(end-start)
                ## !!! replace by beta and sigma
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None] # this is noise, created in one
                beta = self.beta[t][:, None, None, None]
                noise = torch.randn_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train() # it goes back to training mode
        x = (x.clamp(-1, 1) + 1) / 2 # to be in [-1, 1], the plus 1 and the division by 2 is to bring back values to [0, 1]
        x = (x * 255).type(torch.uint8) # to bring in valid pixel range
        return x















