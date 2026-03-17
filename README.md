# Pokemon Diffusion Model from Scratch

An end-to-end Machine Learning pipeline that trains a custom Unconditional Image Diffusion Model to generate brand new Pokémon sprites. Built with PyTorch and Hugging Face `diffusers`, optimized for Apple Silicon (MPS), and deployed with a local Gradio Web UI.

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Models_&_Datasets-FFD21E?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Deployed_with-Gradio-ff69b4?style=for-the-badge)

## 📖 Project Overview

The goal of this project was to build and train a Generative AI diffusion model from the ground up, rather than pulling an off-the-shelf pre-trained model. This repository contains the complete pipeline: from raw dataset preprocessing and custom PyTorch training loops, to hyperparameter A/B testing, mathematical evaluation (FID scoring), and a standalone production web app.

### ✨ Key Features
* **Custom Data Pipeline:** Automatically processes raw RGBA Pokémon sprites, fixing transparency bugs by standardizing pure white backgrounds to prevent tensor distortion.
* **Math-Verified Normalization:** Includes statistical pixel distribution checks to ensure data is perfectly mapped to the `[-1.0, 1.0]` range required by the UNet.
* **Apple Silicon Optimized:** Fully integrated with `accelerate` and `MPS` (Metal Performance Shaders) for fast local training on Mac hardware.
* **Hyperparameter A/B Testing:** Simultaneously trains and evaluates two distinct models to measure the impact of timesteps and noise schedulers on short-cycle training runs.
* **Decoupled Production UI:** A standalone `app.py` script that loads the compiled binary weights and launches a clean Gradio interface, bypassing complex local network/proxy restrictions.

---

## 🏗️ Architecture & A/B Testing

I trained two competing `UNet2DModel` architectures simultaneously for **50 epochs** to test the efficiency of different noise schedulers in a compute-constrained environment.

| Feature | Model A | Model B |
| :--- | :--- | :--- |
| **Inference Steps** | 100 | 1000 |
| **Noise Schedule** | Linear | Cosine |
| **Final FID Score** | **7.20** 🏆 | 9.71 |

*(Note: Fréchet Inception Distance (FID) measures mathematical similarity to the real dataset. Lower is better. A score of 7.20 indicates highly realistic, production-grade fidelity).*

---

## 🛠️ Installation & Quick Start

### 1. Clone the repository and switch to the project branch
```bash
git clone [https://github.com/griddynamics/ds_interns_project_2_2026.git](https://github.com/griddynamics/ds_interns_project_2_2026.git)
cd ds_interns_project_2_2026
git checkout Pavan_Boodida_Pokemon_Generator

```
### 2. Set up the Environment
It is highly recommended to use a virtual environment to prevent dependency conflicts.
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

```
### 3. Install Dependencies
Install the required machine learning and web deployment libraries.
```bash
pip install -r requirements.txt
```
### 4. Run the Web UI
Ensure model_A_weights.pth is in your root directory alongside app.py, then launch the Gradio app:
```bash
python app.py
```

📝 Project Conclusion & Learnings

Looking back at this project, building a custom diffusion model from scratch was one of the most challenging but rewarding things I’ve coded. I didn't just want to pull an off-the-shelf model; I wanted to build the math, process the raw pixels, and actually understand how a UNet hallucinates a Pokémon out of pure static.

Training this locally on my Mac's Apple Silicon (MPS) was a journey. Between fixing the pure-black background bugs in the image tensors and surviving an absolute dependency hell between Pydantic, FastAPI, and Gradio at the very end, I definitely earned my deployment stripes.

But the biggest takeaway for me was the A/B test. Going into this, I fully expected Model B (1000 inference steps with a Cosine schedule) to completely outclass Model A (100 steps with a Linear schedule). In theory, more steps and a smoother noise curve should equal a better image.

Instead, Model A absolutely crushed it, hitting a highly realistic FID score of 7.20.

It taught me a massive lesson about compute constraints: because I was only training for 50 epochs, the aggressive, blocky noise removal of the Linear schedule actually forced the model to learn bold shapes and vibrant colors much faster. The Cosine model was too gentle, and just didn't have the runway (enough epochs) to reach its full potential. It was a great reminder that in machine learning, the "fancier" hyperparameter isn't always the right choice for your specific hardware and time constraints.
```
```
🔮 Future Experiments

Now that the foundational pipeline is locked in and deployed via Gradio, I already have a few ideas for where I want to take this next:

Letting the Cosine Model "Cook": Now I know that 50 epochs isn't enough for a Cosine schedule. My next experiment is to leave my Mac running overnight, push Model B to 300 or 500 epochs, and see if that slow-burn training finally surpasses the Linear model's fidelity. I want to see how smooth those textures can actually get.

Adding Class Conditioning (Type-to-Image): Right now, my Gradio app takes a random seed and generates a totally random Pokémon. The next architectural step is to add an Embedding layer to the UNet so I can pass in text labels. I want to be able to hit a dropdown menu, select "Fire/Dragon", and have the model deliberately generate a custom sprite that fits that elemental typing.

Permanent Cloud Deployment: I successfully got the Gradio app running locally and bypassing my corporate proxy using temporary .live links. The next engineering step is to wrap this entire application in a Docker container and push it to Hugging Face Spaces or AWS. I want to turn this from a local script into a permanent web app I can drop in my portfolio.
```
```
Live Site-https://9e0b56b1cc43e9b399.gradio.live
