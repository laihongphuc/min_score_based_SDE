import random 

import numpy as np 
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm


from unet import ScoreNet
from vp_sde import VP
from solver import Euler_Maruyama_sampler


# Hyperparameters
n_epochs = 50          # number of epochs 
batch_size = 64        # batch size
lr = 1e-4              # learning rate
num_steps=1000         # number of steps for discrete solve SDE
beta_min = 0.01        # \bar{beta_min}
beta_max = 10          # \bar{beta_max}
x_shape = (1, 28, 28)  # shape of image


def visualize_images(batch_images, nrow=4):
    grid_image = make_grid(batch_images, nrow=nrow) 
    # convert to numpy format 
    grid_image_np = grid_image.permute(1, 2, 0).numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_image_np)
    plt.axis('off')
    plt.show()
    
    
def loss_fn(model, x, sde, eps=1e-5):
    """ Inputs:
          model: score model (i.e. diffusion model)
          x: batch of images
          sde: instance of VP class
          eps: parameter for numerical stability (1e-5 for learning, 1e-3 for sampling)
    Pipeline:
    - init a batch size of x shape (batch_size, x_shape) from gaussian
    - loop from t = 1 -> 0 with num_steps:
        - compute current t = 1 - delta_t * i
        - compute the current score, drift, diffusion term 
        - generate noise z_t with shape (batch_size, x_shape)
        - compute x_{t - delta_t}
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
    z = torch.randn_like(x)
    mean, std = sde.marginal_proba(x, random_t)
    perturbed_x = mean + z * std[:, None, None, None]

    # predict the score function for each perturbed x in the batch and its corresponding random t
    score = model(perturbed_x, random_t)
    
    # compute loss
    losses = score * std[:, None, None, None] + z
    loss = torch.mean(torch.sum(losses**2, dim=(1,2,3)))
    return loss


def train_model(model, sde, train_loader, optimizer, epochs, device='cuda'):
    for epoch in range(epochs):
        loader = tqdm(train_loader)
        losses = []
        best_avg_loss = np.inf
        for x, _ in loader:
            x = x.to(device)
            loss = loss_fn(model, x, sde)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item() / x.shape[0])
            loader.set_description(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
        print(f'Epoch: {epoch}, Loss: {np.mean(losses):.4f}')
        current_avg_loss = np.mean(losses)
        if best_avg_loss > current_avg_loss:
            best_avg_loss = current_avg_loss
            torch.save(model.state_dict(), './checkpoint/vp_sde_ckpt_best.pt')
        if epoch % 10 == 0:
            sample_test = Euler_Maruyama_sampler(model, sde, 16, x_shape, device=device)
            sample_test = sample_test.detach().cpu()
            visualize_images(sample_test, nrow=4)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # set seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # create model
    sde = VP(beta_min, beta_max, num_steps)
    score_model = ScoreNet(marginal_proba=sde.marginal_proba)

    # create dataloader 
    ## for training SDE we only need to scale image from 0-1
    train_transforms = transforms.Compose([transforms.ToTensor()])
    train_dataset = FashionMNIST('.', train=True, transform=train_transforms, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = torch.optim.AdamW(score_model.parameters(), lr=lr)
    train_model(score_model, sde, train_loader, optimizer, epochs=n_epochs, device=device)