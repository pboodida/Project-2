import torch
import gradio as gr
import numpy as np
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler
from tqdm.auto import tqdm
print("Initializing Production Model A...")
#1.Setup Apple Silicon (MPS) Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#2.Rebuilding the exact UNet Architecture
model_A = UNet2DModel(
    sample_size=64,  
    in_channels=3,
    out_channels=3,
    layers_per_block=2,  
    block_out_channels=(64, 128, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
)
#3.Loading the trained weights
model_A.load_state_dict(torch.load("model_A_weights.pth", map_location=device, weights_only=False))
model_A.to(device)
model_A.eval()
model_A.to(torch.float32) # Ensuring stable Mac math
#4.Rebuilding the Model A Scheduler(100 steps, linear)
noise_scheduler_A = DDPMScheduler(num_train_timesteps=100, beta_schedule="linear")
#5.Defining the Generation Function
def generate_pokemon(seed):
    if seed:
        torch.manual_seed(int(seed))      
    #Start with pure noise
    gen_image = torch.randn((1, 3, 64, 64), dtype=torch.float32, device=device)
    noise_scheduler_A.set_timesteps(100)
    #Denoising loop
    for t in noise_scheduler_A.timesteps:
        with torch.no_grad():
            residual = model_A(gen_image, t, return_dict=False)[0]
        gen_image = noise_scheduler_A.step(residual, t, gen_image).prev_sample  
    #Converting Tensor to PIL Image
    gen_image = ((gen_image[0] + 1.0) / 2.0).clamp(0.0, 1.0).cpu() 
    gen_image_np = (gen_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  
    return Image.fromarray(gen_image_np)
#6.Build and Launch the Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👾 Custom Pokemon Generator (Model A Production)")
    gr.Markdown("Running locally on Apple Silicon (MPS) using a 100-step linear diffusion schedule.")  
    with gr.Row():
        with gr.Column():
            seed_input = gr.Number(label="Random Seed (Optional)", precision=0)
            generate_btn = gr.Button("Generate Pokemon", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Generated Sprite")
    generate_btn.click(fn=generate_pokemon, inputs=seed_input, outputs=output_image)
if __name__ == "__main__":
    # Bypass the Mac proxy by generating a public share link
    demo.launch(share=True)