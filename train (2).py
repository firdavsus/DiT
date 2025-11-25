import torch
from datasets import load_dataset
from torchvision.utils import make_grid
from torchvision import transforms
from diffusers import AutoencoderDC
import os
import torch
import torchvision.transforms as T
import time
import platform
import random
from torchvision.transforms.functional import pil_to_tensor
from torchmetrics.functional.multimodal import clip_score
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader
from operator import itemgetter
from datasets import load_dataset
import torch.nn.functional as F

from model import DiT, config
from tqdm import tqdm

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

def load_imagenet_1k_vl_enriched_recaped():
    import requests, gzip, json
    from io import BytesIO
    
    # URL of the gzipped JSON file
    url = "https://huggingface.co/datasets/g-ronimo/imagenet-1k-vl-enriched-recaped/resolve/main/captions.json.gz"
    
    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    
    with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz:
        data = json.loads(gz.read().decode('utf-8'))
    return data

def load_IN1k256px_AR(batch_size=512, batch_size_eval=256, label_dropout=0.1):
    splits_train = ["train_AR_1_to_1", "train_AR_3_to_4", "train_AR_4_to_3"]
    splits_eval = ["validation_AR_1_to_1", "validation_AR_3_to_4", "validation_AR_4_to_3"]

    ds = load_dataset("g-ronimo/IN1k256-AR-buckets-bfl16latents_dc-ae-f32c32-sana-1.0")

    dataloader_train = ImageNetARDataset(
        ds, 
        splits=splits_train, 
        bs=batch_size, 
        label_dropout=label_dropout,
        ddp=False,
    )

    dataloader_eval = ImageNetARDataset(
        ds, 
        splits=splits_eval, 
        bs=batch_size, 
        label_dropout=None,
        ddp=False
    )

    return dataloader_train, dataloader_eval

class ImageNetARDataset(torch.utils.data.Dataset):
    def __init__(
        self, hf_dataset, splits, bs, label_dropout=None, ddp=False, col_id="image_id", col_label="label", col_latent="latent"
    ):
        self.hf_dataset = hf_dataset
        self.bs = bs
        # each split is one aspect ratio
        self.splits = splits  
        self.col_label, self.col_latent, self.col_id = col_label, col_latent, col_id
        self.label_dropout = label_dropout

        # load md2, qwen2 and smolvlm captions
        self.in1k_recaps = load_imagenet_1k_vl_enriched_recaped()

        seed = 42

        # Create a dataloader for each split (=aspect ratio)
        self.dataloaders = {}
        self.samplers = {}
        for split in splits:
            if ddp: 
                self.samplers[split] = DistributedSampler(hf_dataset[split], shuffle=True, seed=seed)
            else: 
                self.samplers[split] = RandomSampler(hf_dataset[split], generator=torch.manual_seed(seed))
            self.dataloaders[split] = DataLoader(
                hf_dataset[split], sampler=self.samplers[split], collate_fn=self.collate, batch_size=bs, num_workers=4, prefetch_factor=2
            )

    def collate(self, items):
        labels = [
            # random pick between md2, qwen2 and smolvlm
            self.in1k_recaps[i[self.col_id]][random.randint(0, 2)]
            for i in items
        ]

        # drop 10% of the labels
        if self.label_dropout:
            labels = [ label if random.random() > self.label_dropout else "" for label in labels ]

        # latents shape [B, 1, 32, W, H] -> squeeze [B, 32, W, H]
        latents = torch.Tensor([i[self.col_latent] for i in items]).squeeze()

        return labels, latents
  
    def __iter__(self):
        # Reset iterators at the beginning of each epoch
        iterators = { split: iter(dataloader) for split, dataloader in self.dataloaders.items() }
        active_dataloaders = set(self.splits)  # Track exhausted dataloaders
        current_split_index = -1
        
        while active_dataloaders:
            # Round robin: change split on every iteration (=after every batch OR after we unsucc. tried to get a batch) 
            current_split_index = (current_split_index + 1) % len(self.splits)
            split = self.splits[current_split_index]

            # Skip if this dataloader is exhausted
            if split not in active_dataloaders: continue
            
            # Try to get the next batch
            try:
                labels, latents = next(iterators[split]) 

                yield labels, latents
            # dataloader is exhausted
            except StopIteration: active_dataloaders.remove(split)

    def set_epoch(self, epoch):
        for split in self.splits:
            sampler = self.samplers[split]
    
            if isinstance(sampler, torch.utils.data.DistributedSampler):
                sampler.set_epoch(epoch)
            else:
                # recreate RandomSampler with new seed
                g = torch.Generator()
                g.manual_seed(42 + epoch)
    
                self.samplers[split] = RandomSampler(
                    self.hf_dataset[split],
                    generator=g
                )
    
                # rebuild dataloader using the new sampler
                self.dataloaders[split] = DataLoader(
                    self.hf_dataset[split],
                    sampler=self.samplers[split],
                    collate_fn=self.collate,
                    batch_size=self.bs,
                    num_workers=4,
                    prefetch_factor=2,
                    shuffle=False
                )

    def __len__(self):
        return sum([len(self.samplers[split]) for split in self.splits]) // self.bs


transformer = DiT()
transformer.load_state_dict(torch.load("model/real_model-49.pth", map_location=device))
# transformer = torch.compile(transformer_og)


# transformer = torch.compile(transformer_og)
def generate(prompt, transformer, tokenizer, text_encoder, dcae, num_steps = 10, latent_dim = [1, 32, 8, 8], guidance_scale = 3.4, neg_prompt = "", seed=None, max_prompt_tok=50, add_special_tokens=False):
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

dataloader_train, dataloader_eval = load_IN1k256px_AR(
      batch_size=256, label_dropout=0.30
)

# Then move to device and set dtype explicitly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = transformer.to(device=device, dtype=torch.float32)
text_encoder = text_encoder.to(device=device, dtype=torch.float32)
dcae = dcae.to(device=device, dtype=torch.float32)

optimizer = torch.optim.AdamW(
    transformer.parameters(), 
    lr=0.0002, 
    betas=(0.9, 0.95),  # Better betas for transformers
    weight_decay=0.02   # Mild weight decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5004*50)
transformer.to(device)
transformer = transformer.to(dtype=torch.float32)

epoch = 50

from torch.cuda.amp import autocast, GradScaler

accum_steps = 1             
scaler = GradScaler()

for ep in range(epoch):
    dataloader_train.set_epoch(ep)
    pbar = tqdm(dataloader_train, desc=f"Epoch {ep+1}/{epoch}", leave=True)

    optimizer.zero_grad()

    for step, (labels, latents) in enumerate(pbar):

        prompts_emb, prompts_atnmask = encode_prompt(
            labels, max_length=50,
            tokenizer=tokenizer,
            text_encoder=text_encoder
        )

        prompts_emb = prompts_emb.to(device, dtype=torch.float32)
        prompts_atnmask = prompts_atnmask.to(device, dtype=torch.float32)

        latents = latents.to(device, dtype=torch.float32)
        latents *= dcae.config["scaling_factor"]

        latents_noisy, timestep, noise = add_random_noise(latents)

        # --- FP16 forward pass ---
        with autocast(dtype=torch.float16):

            noise_pred = transformer(
                latents_noisy,
                timestep,
                prompts_emb,
                prompts_atnmask
            )

            loss = F.mse_loss(noise_pred, noise - latents)
            loss = loss / accum_steps          # scale for accumulation

        # --- backward using scaler ---
        scaler.scale(loss).backward()

        # every accum_steps → update weights
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() 
            optimizer.zero_grad()

        pbar.set_postfix({"loss": f"{loss.item()*accum_steps:.4f}"})
    
        if (step + 1) % 500 == 0:
            print(step, loss.item())
            img = generate(
                "a pirate ship",
                transformer,
                tokenizer,
                text_encoder,
                dcae,
                latent_dim=[1, 32, 8, 8],
                num_steps=20,
            )
            
            img = img.resize((256, 256))
            img.save(f"logs_tr/pirate-{ep}.png")

            img = generate(
                "a cat",
                transformer,
                tokenizer,
                text_encoder,
                dcae,
                latent_dim=[1, 32, 8, 8],
                num_steps=20,
            )
            
            img = img.resize((256, 256))
            img.save(f"logs_tr/cat-{ep}.png")
            
            torch.save(transformer.state_dict(), f"save_tr/real_model-{ep+1}.pth")