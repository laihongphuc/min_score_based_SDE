import torch

class VP():
    def __init__(self, beta_min, beta_max, num_steps):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_steps = num_steps
        self.discrete_betas = torch.linspace(beta_min / num_steps, beta_max / num_steps, num_steps)
        self.alphas = 1. - self.discrete_betas

    def _beta_t(self, t):
        """
        Compute beta(t) for t in [0, 1]
        """
        return (self.beta_min + t * (self.beta_max - self.beta_min))
    
    def _c_t(self, t):
        """
        Compute c(t) for t in [0, 1]
        """
        return -1 / 2 * (self.beta_min * t + (self.beta_max - self.beta_min) * t**2 /2)
    
    def marginal_proba(self, x, t):
        """
        Return the mean and standared deviation of marginal prob p(x_t|x_0)
        """
        mean = torch.exp(self._c_t(t))[:, None, None, None] * x
        std = torch.sqrt(1 - torch.exp(2 * self._c_t(t)))
        return mean, std

    def drift(self, x, t):
        """
        Compute VP drift coefficient f(x, t)
        """
        return -1 / 2 * self._beta_t(t)[:, None, None, None] * x
    
    def diffusion(self, t):
        """
        Compute VP diffusion coefficient g(t)
        """
        return torch.sqrt(self._beta_t(t))
        