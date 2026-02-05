import torch
import numpy as np


class DDPMSampler:
    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 10000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ) -> None:
        self.beta = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32
            )
            ** 2
        )
        self.alpha = 1.0 - self.beta
        self.generator = generator
        self.alphacum = torch.cumprod(self.alpha, dim=0)
        self.num_training_steps = num_training_steps
        self.timestep = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
