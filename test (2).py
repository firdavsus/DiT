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

def generate(prompt, transformer, tokenizer, text_encoder, dcae, num_steps = 20, latent_dim = [1, 32, 8, 8], guidance_scale = None, neg_prompt = "", seed=None, max_prompt_tok=50, add_special_tokens=False):
    device, dtype = transformer.device, transformer.dtype
    do_cfg = guidance_scale is not None

    # Encode the prompt, +neg. prompt if classifier free guidance (CFG)
    prompt_encoded, prompt_atnmask = encode_prompt(
        [prompt, neg_prompt] if do_cfg else prompt,
        tokenizer,
        text_encoder,
        max_length = max_prompt_tok,
        add_special_tokens = add_special_tokens
    )

    # Divide 1000 -> 0 in equally sized steps
    timesteps = torch.linspace(1000, 0, num_steps + 1, device=device, dtype=dtype)

    # Noise level. 1.0 -> 0.0 in equally sized steps
    sigmas = timesteps / 1000

    latent = torch.randn(
        latent_dim,
        generator=torch.manual_seed(seed) if seed else None
    ).to(dtype).to(device)

    for t, sigma_prev, sigma_next, steps_left in zip(
        timesteps,
        sigmas[:-1],
        sigmas[1:],
        range(num_steps, 0, -1)
    ):
        t = t[None].to(device)

        # DiT predicts noise
        with torch.no_grad():
            noise_pred = transformer(
                hidden_states = torch.cat([latent] * 2) if do_cfg else latent,
                timestep = torch.cat([t] * 2) if do_cfg else t,
                encoder_hidden_states=prompt_encoded,
                encoder_attention_mask=prompt_atnmask,
                return_dict=False
            )[0]

        if do_cfg:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Remove noise from latent
        latent = latent + (sigma_next - sigma_prev) * noise_pred

    return latent_to_PIL(latent / dcae.config["scaling_factor"], dcae)

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
def generate(prompt, transformer, tokenizer, text_encoder, dcae, num_steps = 10, latent_dim = [1, 32, 8, 8], guidance_scale = 4.0, neg_prompt = "", seed=None, max_prompt_tok=50, add_special_tokens=False):
    device, dtype = "cuda", torch.float32
    do_cfg = guidance_scale is not None

    # Encode the prompt, +neg. prompt if classifier free guidance (CFG)
    prompt_encoded, prompt_atnmask = encode_prompt(
        [prompt, neg_prompt] if do_cfg else prompt,
        tokenizer,
        text_encoder,
        max_length = max_prompt_tok,
        add_special_tokens = add_special_tokens
    )

    # Divide 1000 -> 0 in equally sized steps
    timesteps = torch.linspace(1000, 0, num_steps + 1, device=device, dtype=dtype)

    # Noise level. 1.0 -> 0.0 in equally sized steps
    sigmas = timesteps / 1000

    latent = torch.randn(
        latent_dim,
        generator=torch.manual_seed(seed) if seed else None
    ).to(dtype).to(device)

    for t, sigma_prev, sigma_next, steps_left in zip(
        timesteps,
        sigmas[:-1],
        sigmas[1:],
        range(num_steps, 0, -1)
    ):
        t = t[None].to(device)

        # DiT predicts noise
        with torch.no_grad():
            noise_pred = transformer(torch.cat([latent] * 2) if do_cfg else latent,torch.cat([t] * 2) if do_cfg else t,
                prompt_encoded,
                prompt_atnmask
            )

        if do_cfg:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Remove noise from latent
        latent = latent + (sigma_next - sigma_prev) * noise_pred

    return latent_to_PIL(latent / dcae.config["scaling_factor"], dcae)




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
                    num_steps=500,
                    guidance_scale = 3.4,
    )
                
    img = img.resize((256, 256))
    img.save(f"cat.png")