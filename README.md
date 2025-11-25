# DiT-Latent-Generator

A sleek latent-space generative model inspired by **SANA**, built with a DiT-style transformer + VAE. It predicts noise in latent space, conditioned on text, and decodes back into images via a VAE.

---

## 🚀 Architecture Overview

- **VAE Component:**  
  - Uses a pretrained `AutoencoderDC` (diffusers) as the VAE.  
  - Latents are scaled, noised, and reconstructed.  
  - Clean latent space via re-encoding and denoising.

- **Transformer (DiT):**  
  - A DiT-like transformer that predicts noise, given current latent and a timestep.  
  - Uses RoPE positional embeddings in 2D, AdaLN-Zero for time conditioning, and SwiGLU for MLP.  
  - Cross-attends to text embeddings so the generation is **text-conditioned**.

- **Text Encoder (LLM):**  
  - Lightweight transformer / LLM to embed prompts.  
  - These embeddings are projected and injected into DiT, enabling **classifier-free guidance (CFG)**.

---

## 📁 File Map

- `model.py` — defines the DiT architecture, time embeddings, attention blocks, and VAE interface.  
- `train.py` — training loop: add noise to latent, predict it, compute loss (MSE + maybe regularization), scheduler + optimizer, checkpointing, and sample generation.  
- `test.py` — inference: given a prompt, run the generation loop, decode latent with VAE, and save or display images.

## you can find a checkpoint there
- `firdavsus/text2Image` <-huggingface
---
<p align="center">
  <img src="examples/result_row (3).png">
  <img src="examples/result_row (4).png">
  <img src="examples/result_row (5).png">
  <img src="examples/result_row (6).png">
  <img src="examples/result_row (7).png">
  <img src="examples/result_row (8).png">
  <img src="examples/result_row (10).png">
  <img src="examples/result_row (11).png">
  <img src="examples/result_row (12).png">
</p>

