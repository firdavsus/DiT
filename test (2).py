import torch
from diffusers import AutoencoderDC
import os
import torch
import torchvision.transforms as T

from model import DiT, config

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
model_repo = "Efficient-Large-Model/Sana_600M_1024px_diffusers"

dcae = AutoencoderDC.from_pretrained(
    model_repo,
    subfolder="vae",
    torch_dtype=dtype
).to(device)

# stop complaining
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def add_random_noise(latents, timesteps=1000, dist="uniform"):
    assert dist in ["normal", "uniform"], f"Requested sigma dist. {dist} not supported"

    # batch size
    bs = latents.size(0)

    # gaussian noise
    noise = torch.randn_like(latents)

    # normal distributed sigmas
    if dist == "normal":
        sigmas = torch.randn((bs,)).sigmoid().to(latents.device)
    else:
        sigmas = torch.rand((bs,)).to(latents.device)
    # sigmas = torch.randn((bs,)).sigmoid().to(latents.device)

    timesteps = (sigmas * timesteps).to(latents.device)   # yes, `timesteps = sigmas * 1000`, let's keep it simple
    sigmas = sigmas.view([latents.size(0), *([1] * len(latents.shape[1:]))])

    latents_noisy = (1 - sigmas) * latents + sigmas * noise # (1-noise_level) * latent + noise_level * noise

    return latents_noisy.to(latents.dtype), timesteps, noise

def encode_prompt(prompt, tokenizer, text_encoder, max_length=50, add_special_tokens=False, **kwargs):
    # lower case prompt! took a long time to find that this is necessary: https://github.com/huggingface/diffusers/blob/e8aacda762e311505ba05ae340af23b149e37af3/src/diffusers/pipelines/sana/pipeline_sana.py#L433
    tokenizer.padding_side = "right"
    if isinstance(prompt, list):
        prompt = [p.lower().strip() for p in prompt]
    elif isinstance(prompt, str):
        prompt = prompt.lower().strip()
    else:
        raise Exception(f"Unknown prompt type {type(prompt)}")
    prompt_tok = tokenizer(prompt, return_tensors="pt", return_attention_mask=True, padding="max_length", truncation=True, max_length=max_length, add_special_tokens=add_special_tokens).to(text_encoder.device)
    with torch.no_grad():
        prompt_encoded=text_encoder(**prompt_tok)
    return prompt_encoded.last_hidden_state, prompt_tok.attention_mask

def latent_to_PIL(latent, ae):
    with torch.no_grad():
        image_out = ae.decode(latent).sample.to("cpu")

    if image_out.size(0) == 1:
        # Single image processing
        image_out = torch.clamp_(image_out[0,:], -1, 1)
        image_out = image_out * 0.5 + 0.5
        return T.ToPILImage()(image_out.float())
    else:
        images = []
        for img in image_out:
            img = torch.clamp_(img, -1, 1)
            img = img * 0.5 + 0.5
            images.append(T.ToPILImage()(img.float()))
        return images
from transformers import AutoModel, AutoTokenizer

te_repo = "HuggingFaceTB/SmolLM2-360M"

tokenizer = AutoTokenizer.from_pretrained(te_repo, torch_dtype=dtype)
text_encoder = AutoModel.from_pretrained(te_repo, torch_dtype=dtype).to(device)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


# transformer = torch.compile(transformer_og)
import torch
from typing import List, Union

def generate(
    prompt: str,
    transformer,
    tokenizer,
    text_encoder,
    dcae,
    num_steps: int = 10,
    latent_dim: Union[List[int], tuple] = (1, 32, 8, 8),
    guidance_scale: float = 4.0,
    neg_prompt: str = "",
    seed: int = None,
    max_prompt_tok: int = 50,
    add_special_tokens: bool = False,
    batch_size: int = 4,   # <-- number of variants to produce in parallel
):
    device, dtype = "cuda", torch.float32
    do_cfg = guidance_scale is not None

    # Encode prompt once. For CFG we expect prompt_encoded to contain two entries:
    # [cond_emb, uncond_emb] (same ordering you used previously).
    prompt_encoded, prompt_atnmask = encode_prompt(
        [prompt, neg_prompt] if do_cfg else prompt,
        tokenizer,
        text_encoder,
        max_length=max_prompt_tok,
        add_special_tokens=add_special_tokens
    )
    # Ensure tensors are on correct device/dtype
    prompt_encoded = prompt_encoded.to(device=device, dtype=dtype)
    prompt_atnmask = prompt_atnmask.to(device=device)

    # Adjust latent_dim so first dim == batch_size
    latent_shape = list(latent_dim)
    latent_shape[0] = batch_size

    # Generator for reproducibility on CUDA
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    # Sample initial latents: shape (batch_size, C, H, W) (or whatever your latent_shape means)
    latent = torch.randn(*latent_shape, generator=gen, device=device, dtype=dtype)

    # Prepare timesteps / sigmas
    timesteps = torch.linspace(1000, 0, num_steps + 1, device=device, dtype=dtype)
    sigmas = timesteps / 1000.0

    # If using CFG we need to expand prompt embeddings & masks so they match the
    # 2 * batch_size inputs we'll pass (cond repeated batch_size, then uncond repeated batch_size).
    if do_cfg:
        # Expect prompt_encoded shape: (2, seq_len, dim) and prompt_atnmask (2, seq_len) — adapt if different.
        cond_emb = prompt_encoded[0:1]  # (1, L, D)
        uncond_emb = prompt_encoded[1:2]  # (1, L, D)
        cond_mask = prompt_atnmask[0:1]  # (1, L)
        uncond_mask = prompt_atnmask[1:2]  # (1, L)

        # Repeat each embedding batch_size times and concatenate so indexing matches chunking later
        prompt_encoded_batched = torch.cat(
            [cond_emb.repeat(batch_size, 1, 1), uncond_emb.repeat(batch_size, 1, 1)],
            dim=0
        )  # shape (2*batch_size, L, D)

        prompt_atnmask_batched = torch.cat(
            [cond_mask.repeat(batch_size, 1), uncond_mask.repeat(batch_size, 1)],
            dim=0
        )  # shape (2*batch_size, L)
    else:
        # Single prompt repeated for each sample in batch
        prompt_encoded_batched = prompt_encoded.repeat(batch_size, 1, 1)
        prompt_atnmask_batched = prompt_atnmask.repeat(batch_size, 1)

    # Diffusion loop
    for t, sigma_prev, sigma_next, steps_left in zip(
        timesteps,
        sigmas[:-1],
        sigmas[1:],
        range(num_steps, 0, -1)
    ):
        t = t[None].to(device)  # shape (1,)
        # Build t batch matching transformer input size
        if do_cfg:
            t_batch = t.repeat(2 * batch_size)
        else:
            t_batch = t.repeat(batch_size)

        # Run model (no grad)
        with torch.no_grad():
            if do_cfg:
                # duplicate latents: cond block then uncond block (same order as prompt_encoded_batched)
                model_input = torch.cat([latent, latent], dim=0)  # shape (2*batch_size, ...)
                noise_pred = transformer(
                    model_input,
                    t_batch,
                    prompt_encoded_batched,
                    prompt_atnmask_batched
                )
                # noise_pred shape = (2*batch_size, ...)
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
                # guided noise: uncond + scale*(cond - uncond)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                # noise_pred now shape (batch_size, ...)
            else:
                noise_pred = transformer(
                    latent,
                    t_batch,
                    prompt_encoded_batched,
                    prompt_atnmask_batched
                )
                # noise_pred shape (batch_size, ...)

        # Denoise/update latent
        latent = latent + (sigma_next - sigma_prev) * noise_pred

    # Convert latents to images. If latent_to_PIL accepts batch, you can pass entire tensor;
    # otherwise convert per-sample.
    try:
        # try batch conversion
        images = latent_to_PIL(latent / dcae.config["scaling_factor"], dcae)
        # ensure images is a list (wrap single image results)
        if not isinstance(images, (list, tuple)):
            images = [images]
    except Exception:
        # fallback: convert each latent to an image individually
        images = []
        scaled = (latent / dcae.config["scaling_factor"])
        for i in range(scaled.shape[0]):
            images.append(latent_to_PIL(scaled[i : i + 1], dcae))

    return images  # list of PIL.Image objects, length == batch_size





transformer = DiT()
transformer.load_state_dict(torch.load("model/real_model-50.pth", map_location=device))

# Then move to device and set dtype explicitly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = transformer.to(device=device, dtype=torch.float32)
text_encoder = text_encoder.to(device=device, dtype=torch.float32)
dcae = dcae.to(device=device, dtype=torch.float32)


transformer.to(device)
transformer = transformer.to(dtype=torch.float32)
transformer.eval()

from PIL import Image

while True:
    prompt = input("Prompt: ")
    if prompt == "stop":
        break
    img = generate(
                    prompt,
                    transformer,
                    tokenizer,
                    text_encoder,
                    dcae,
                    latent_dim=[1, 32, 8, 8],
                    num_steps=40,
                    guidance_scale = 5.0,
    )
                
    imgs = [im.resize((256, 256)) for im in img]

    # Combine in a row
    w, h = imgs[0].size
    out = Image.new("RGB", (w * len(imgs), h))
    
    for i, im in enumerate(imgs):
        out.paste(im, (i * w, 0))
    
    out.save("result_row.png")
