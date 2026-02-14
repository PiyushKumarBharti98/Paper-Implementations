import torch
import numpy as np
from torch._C import device, dtype


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
        self.one = torch.tensor(1.0)
        self.alphacum = torch.cumprod(self.alpha, dim=0)
        self.num_training_steps = num_training_steps
        self.timestep = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_timestep = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_timestep

    def _get_variance(self, timestep: int):
        previous_t = self._get_previous_timestep(timestep)
        alpha_t = self.alphacum[timestep]
        alpha_prev_t = self.alphacum[previous_t] if previous_t >= 0 else self.one
        current_beta_t = 1 - alpha_t / alpha_prev_t

        variance = ((1 - alpha_prev_t) / (1 - alpha_t)) * current_beta_t
        variance = torch.clamp(variance, min=1e-10)

        return variance

    def set_strength(self, strength: float = 1.0):
        start_step = self.num_training_steps - (self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_outputs: torch.Tensor):
        t = timestep
        previous_t = self._get_previous_timestep(t)

        alpha_t = self.alphacum[timestep]
        alpha_prev_t = self.alphacum[previous_t] if previous_t >= 0 else self.one
        beta_t = 1 - alpha_t
        beta_prev_t = 1 - alpha_prev_t
        curr_alpha_t = alpha_t / alpha_prev_t
        curr_beta_t = 1 - curr_alpha_t

        # x0 calculation
        prediction_original_sample = (
            latents - (beta_t ** (0.5)) * model_outputs
        ) / alpha_t**0.5

        predicted_sample_coeff = ((beta_prev_t ** (0.5)) * curr_beta_t) / beta_t
        current_sample_coeff = (curr_alpha_t) ** 0.5 * (beta_prev_t) / curr_beta_t

        predicted_prev_samples = (
            predicted_sample_coeff * prediction_original_sample
            + current_sample_coeff * latents
        )

        variance = 0
        if t > 0:
            device = model_outputs.device
            noise = torch.randn(
                model_outputs.shape,
                generator=self.generator,
                device=device,
                dtype=model_outputs.dtype,
            )

            variance = (self._get_variance(t) ** 0.5) * noise
            predicted_prev_samples += variance

        return predicted_prev_samples

    def add_noise(
        self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor
    ):
        cumprod_add = self.alphacum.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha = self.alphacum[timesteps] ** 0.5
        sqrt_alpha = sqrt_alpha.flatten()
        if len(sqrt_alpha.shape) < len(original_samples.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)

        sqrt_alpha_minus = (1 - self.alphacum[timesteps]) ** 0.5
        sqrt_alpha_minus = sqrt_alpha_minus.flatten()
        if len(sqrt_alpha_minus.shape) < len(original_samples.shape):
            sqrt_alpha_minus = sqrt_alpha_minus.unsqueeze(-1)

        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )

        final_noise = sqrt_alpha * original_samples + sqrt_alpha_minus * noise
        return final_noise
