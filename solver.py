import math

from tqdm import tqdm 
import torch 

def Euler_Maruyama_sampler(score_model,
                           sde,
                           batch_size, 
                           num_steps=1000, 
                           device='cuda', 
                           eps=1e-3):

    # compute std at time-step 1 and initialize x
    init_t = torch.ones(batch_size, device=device)
    std = torch.sqrt(1 - torch.exp(2 * sde._c_t(init_t)))
    init_x = torch.randn(batch_size, 1, 28, 28).to(device) * std[:, None, None, None]

    # create a sequence of time_steps from 1 to very smoll
    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    # solve Reverse SDE
    x = init_x
    with torch.no_grad():
        for time_step in tqdm(time_steps):  
            current_t = torch.ones(batch_size, device=device) * time_step
            drift_term = sde.drift(x, current_t)
            diffusion_term = sde.diffusion(current_t)[0]
            score = score_model(x, current_t)
            x_ = x - drift_term * step_size + (diffusion_term ** 2 * step_size) * score 
            z = torch.randn_like(x)
            x = x_ + diffusion_term * z * math.sqrt(step_size)
    # Do not include any noise in the last sampling step.
    return x_