import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPTokenizer


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def pipe_func(
    prompt: str,
    negative_prompt: str = "",
    strength: float = 0.8,
    input_image=None,
    cfg: bool = True,
    cfg_scale: float = 7.5,
    n_inference_steps: int = 50,
    models: dict = {},
    sampler_name: str = "ddpm",
    seed=None,
    device="cuda",
    idle_device=None,
    tokenizer=CLIPTokenizer,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        to_idle = lambda x: x.to(idle_device) if idle_device else x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if cfg:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus(
                [negative_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)

        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")

        latent_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            image_tensor = input_image.resize((WIDTH, HEIGHT))
            image_tensor_np = np.array(image_tensor)
            # CRITICAL: Images must be float32 for math, not long!
            image_tensor_torch = torch.tensor(
                image_tensor_np, dtype=torch.float32, device=device
            )
            image_tensor_rescale = rescale(image_tensor_torch, (0, 255), (-1, 1))
            image_tensor_final = image_tensor_rescale.unsqueeze(0)
            image_tensor_final = image_tensor_final.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(
                latent_shape, generator=generator, device=device
            )
            latents = encoder(image_tensor_final, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            latents = torch.randn(latent_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)
            model_input = latents

            if cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion(model_input, context, time_embedding)

            if cfg:
                cond_output, uncond_output = model_output.chunk(2)
                model_output = cfg_scale * (cond_output - uncond_output) + uncond_output

            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
