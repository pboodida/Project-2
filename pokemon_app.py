# ============================================================
# pokemon_app.py
# Gradio app for generating new Pokemon sprites
# Uses Run 6 trained model weights
# Features: seed slider, guidance scale, PNG export, gallery
# ============================================================

import os
import torch
import numpy as np
import gradio as gr
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler


# ============================================================
# SECTION 1 - Device Setup
# ============================================================

def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running on device : {device}")
    return device

DEVICE = setup_device()


# ============================================================
# SECTION 2 - Load Run 6 Model from Checkpoint
# ============================================================

def load_run6_model(checkpoint_path="checkpoints/final_run6_50epochs.pt"):
    """
    Loads the Run 6 UNet model from saved checkpoint.
    Run 6 config: UNet + Attention 16x16 + Cosine + 1000 timesteps + EMA 0.999
    """
    print("Loading Run 6 model from checkpoint ...")

    model = UNet2DModel(
        sample_size        = 64,
        in_channels        = 3,
        out_channels       = 3,
        block_out_channels = (64, 128, 128, 256),
        down_block_types   = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types     = ("UpBlock2D",   "AttnUpBlock2D", "UpBlock2D",     "UpBlock2D"),
        attention_head_dim = 8,
        norm_num_groups    = 32,
    )

    checkpoint    = torch.load(checkpoint_path, map_location="cpu")
    ema_state     = checkpoint.get("ema_state", None)
    shadow_params = ema_state.get("shadow_params", None) if ema_state else None

    if shadow_params is not None:
        for param, shadow in zip(model.parameters(), shadow_params):
            param.data.copy_(shadow)
        print("EMA weights loaded successfully.")
    else:
        model.load_state_dict(checkpoint["model_state"])
        print("Model weights loaded successfully.")

    model = model.to(DEVICE)
    model.eval()
    return model


def load_scheduler():
    return DDPMScheduler(
        num_train_timesteps = 1000,
        beta_schedule       = "squaredcos_cap_v2",
        clip_sample         = True,
    )

MODEL     = load_run6_model()
SCHEDULER = load_scheduler()
print("Model and scheduler ready.")


# ============================================================
# SECTION 3 - Classifier Free Guidance Generation
# ============================================================

def generate_with_cfg(seed, guidance_scale, num_images, num_steps):
    """
    Generates Pokemon images using classifier free guidance.
    CFG formula: noise_pred = uncond + scale * (cond - uncond)
    Higher guidance_scale gives sharper but less diverse output.
    Recommended range 1.0 to 3.0 for best results.
    """
    torch.manual_seed(int(seed))
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(int(seed))

    images = []

    with torch.no_grad():
        noise = torch.randn(
            num_images, 3, 64, 64,
            device = DEVICE,
            dtype  = torch.float32,
        )

        SCHEDULER.set_timesteps(num_steps)

        for t in SCHEDULER.timesteps:
            t_batch         = torch.tensor([t] * num_images, device=DEVICE).long()
            noise_pred_cond = MODEL(noise, t_batch).sample

            if guidance_scale > 1.0:
                zero_input        = torch.zeros_like(noise)
                noise_pred_uncond = MODEL(zero_input, t_batch).sample
                noise_pred        = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred = noise_pred_cond

            noise = SCHEDULER.step(noise_pred, t, noise).prev_sample

        for i in range(num_images):
            img_tensor = (noise[i] * 0.5 + 0.5).clamp(0, 1)
            img_np     = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            images.append(Image.fromarray(img_np))

    return images


# ============================================================
# SECTION 4 - Save PNG and Return Paths
# ============================================================

def save_images(images, seed):
    """
    Saves generated images as PNG files.
    Returns list of file paths for download.
    """
    output_dir = "generated_pokemon"
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_dir, f"pokemon_seed{seed}_{i+1}.png")
        img.save(path, format="PNG")
        paths.append(path)
    return paths


# ============================================================
# SECTION 5 - Gradio Generation Function
# ============================================================

def gradio_generate(seed, guidance_scale, num_images, num_steps):
    """
    Main Gradio function. Generates Pokemon and returns
    gallery images and saved PNG file paths for download.
    """
    print(f"Generating {num_images} Pokemon  seed={seed}  guidance={guidance_scale}  steps={num_steps}")
    images = generate_with_cfg(int(seed), float(guidance_scale), int(num_images), int(num_steps))
    paths  = save_images(images, int(seed))
    print(f"Done. Saved {len(paths)} images.")
    return images, paths


# ============================================================
# SECTION 6 - Gradio Interface
# ============================================================

with gr.Blocks(title="Pokemon Sprite Generator") as app:

    gr.Markdown("# Pokemon Sprite Generator")
    gr.Markdown(
        "Generates new Pokemon-style sprites using a DDPM diffusion model "
        "trained on 7300 Pokemon images using Run 6 "
        "(UNet + Cosine schedule + 1000 timesteps + EMA 0.999)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Settings")

            seed_slider = gr.Slider(
                minimum = 0,
                maximum = 9999,
                value   = 42,
                step    = 1,
                label   = "Seed",
            )
            guidance_slider = gr.Slider(
                minimum = 1.0,
                maximum = 5.0,
                value   = 1.5,
                step    = 0.1,
                label   = "Guidance Scale",
            )
            num_images_slider = gr.Slider(
                minimum = 1,
                maximum = 8,
                value   = 4,
                step    = 1,
                label   = "Number of Pokemon",
            )
            num_steps_slider = gr.Slider(
                minimum = 50,
                maximum = 1000,
                value   = 200,
                step    = 50,
                label   = "Denoising Steps",
            )
            generate_btn = gr.Button("Generate Pokemon", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Generated Pokemon Gallery")
            gallery = gr.Gallery(
                label      = "Generated Pokemon",
                columns    = 4,
                rows       = 2,
                object_fit = "contain",
                height     = "auto",
            )
            gr.Markdown("### Download PNG Files")
            download = gr.Files(label="Download Generated Pokemon")

    gr.Markdown(
        "**Tips:** "
        "Try different seeds to get different Pokemon styles. "
        "Guidance scale 1.0 to 2.0 gives the best results. "
        "200 denoising steps balances speed and quality."
    )

    generate_btn.click(
        fn      = gradio_generate,
        inputs  = [seed_slider, guidance_slider, num_images_slider, num_steps_slider],
        outputs = [gallery, download],
    )


# ============================================================
# SECTION 7 - Launch
# ============================================================

if __name__ == "__main__":
    app.launch(
        share       = True,
        server_name = "0.0.0.0",
        server_port = 7860,
        show_error  = True,
    )