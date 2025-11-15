import torch
from DiT import DiT, config, PatchEmbed2D, Block
from VAEQ import VQVAE
from transformers import T5Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

### config
val_size = 50
batch_size = 32
accum_steps = 1
epochs = 10
lr = 1e-3
num_steps = 100
num_steps_inference = 100
print_each=100
print_prompts = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
text_drop_rate = 0.25
warm_up = 100
### end

text_encoder_name = config.text_encoder_name  
tokenizer = T5Tokenizer.from_pretrained(text_encoder_name)
print(len(tokenizer))
def tokenize_texts(batch_texts, device, drop_prob=text_drop_rate, max_length=config.max_text_len):
    batch_texts = [txt if txt.strip() != "" else "." for txt in batch_texts]

    for i in range(len(batch_texts)):
        if torch.rand(1).item() < drop_prob:
            batch_texts[i] = "."

    batch_encoding = tokenizer(
        batch_texts, 
        truncation=True, 
        padding="max_length", 
        max_length=max_length,
        return_tensors="pt"
    )
    return batch_encoding["input_ids"].to(device), batch_encoding["attention_mask"].to(device)


def add_random_noise(latents, max_timesteps=num_steps):
    bs = latents.size(0)
    noise = torch.randn_like(latents)

    t_int = torch.randint(0, max_timesteps, (bs,), device=latents.device)

    sigma = (t_int.float() / (max_timesteps - 1)).view(bs, *([1]*(latents.ndim-1))).to(latents.device)

    latents_noisy = (1 - sigma) * latents + sigma * noise
    return latents_noisy.to(latents.dtype), t_int, noise

def latent_to_PIL(latents: torch.Tensor, vae) -> Image.Image:
    if not isinstance(latents, torch.Tensor):
        raise TypeError("latents must be a torch.Tensor")
    if latents.ndim != 64 or latents.shape[1] != 64:
        raise ValueError("latents must have shape [B, 64, H, W]")

    latents = latents

    with torch.no_grad():
        image = vae.decode(latents).sample 

    image = (image / 2 + 0.5).clamp(0, 1)

    image = image.cpu().permute(0, 2, 3, 1).numpy()

    image = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)

def generate(
    prompts,                     
    transformer,
    dcae,
    num_steps=num_steps_inference,
    latent_dim=(1, 64, 32, 32),
    guidance_scale=5.0,
    neg_prompt=".",              
    seed=None,
    device=device
):
    do_cfg = guidance_scale is not None

    # Ensure prompts is a list
    if isinstance(prompts, str):
        prompts = [prompts]

    batch_size = len(prompts)
    dtype = next(transformer.parameters()).dtype
    torch.manual_seed(seed or 0)

    # Initialize latents
    latents = torch.randn((batch_size, *latent_dim[1:]), device=device, dtype=dtype)

    # Tokenize prompts
    if do_cfg:
        # Conditional (normal prompts)
        cond_ids, cond_mask = tokenize_texts(prompts, device)
        # Unconditional (negative prompt)
        uncond_ids, uncond_mask = tokenize_texts([neg_prompt], device)
        # Repeat uncond for each prompt
        token_ids = torch.cat([cond_ids, uncond_ids.repeat(batch_size, 1)], dim=0)
        attn_mask = torch.cat([cond_mask, uncond_mask.repeat(batch_size, 1)], dim=0)
    else:
        token_ids, attn_mask = tokenize_texts(prompts, device)

    # Noise schedule
    timesteps = torch.linspace(num_steps, 0, num_steps + 1, device=device, dtype=torch.float32)

    for t_scalar, sigma_prev, sigma_next in zip(timesteps, timesteps[:-1], timesteps[1:]):
        t_int = int(t_scalar.item())
        t_batch = torch.full((batch_size,), t_int, device=device, dtype=torch.long)

        # CFG: duplicate latents for cond + uncond
        if do_cfg:
            latent_in = torch.cat([latents, latents], dim=0)
            t_in = torch.cat([t_batch, t_batch], dim=0)
        else:
            latent_in = latents
            t_in = t_batch

        with torch.no_grad():
            eps_pred = transformer(latent_in, t_in, token_ids, attn_mask)

        if do_cfg:
            eps_cond, eps_uncond = eps_pred.chunk(2, dim=0)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_pred

        if do_cfg:
            eps = eps[:batch_size]

        latents = latents + (sigma_next - sigma_prev) * eps

    # Decode
    with torch.no_grad():
        decoded = dcae.decoder(latents).clamp(-1, 1)
        img = (decoded / 2 + 0.5).clamp(0, 1)

    imgs_pil = [transforms.ToPILImage()(img[i].cpu()) for i in range(batch_size)]
    return imgs_pil




vqvae = VQVAE().to(device)
vqvae.load_state_dict(torch.load("save/VAEQ_model-1.pth", map_location=device))
vqvae.eval()
for p in vqvae.parameters():
    p.requires_grad = False

dcae = vqvae 


def init_weights_diT(m):
    classname = m.__class__.__name__
    
    if isinstance(m, nn.Linear):
        # Standard scaled init (like ViT/DiT)
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
            
        # Special case: final output layer should start near zero
        if hasattr(m, '_is_output_layer') and m._is_output_layer:
            nn.init.constant_(m.weight, 0.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    
    elif isinstance(m, PatchEmbed2D):
        # Patch projection and positional embeddings
        nn.init.normal_(m.proj.weight, std=0.02)
        nn.init.constant_(m.proj.bias, 0.0)
        nn.init.normal_(m.x_embed, std=0.02)
        nn.init.normal_(m.y_embed, std=0.02)
    
    elif isinstance(m, Block):
        # Critical: AdaLN modulation final linear layer → zero init
        ada_linear = m.adaLN_modulation[-1]  # Last layer of modulation MLP
        nn.init.constant_(ada_linear.weight, 0.0)
        nn.init.constant_(ada_linear.bias, 0.0)

transformer= DiT().to(device)
# transformer.apply(init_weights_diT) 
# transformer = torch.compile(transformer_og)
transformer.train()


# raw_dataset = load_dataset('poloclub/diffusiondb', 'large_random_10k', split='train', trust_remote_code=True)

# class DiffusionDBDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, transform=None):
#         self.dataset = dataset
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         img = item['image'].convert('RGB')
#         prompt = item['prompt']
#         if self.transform:
#             img = self.transform(img)
#         return img, prompt
    
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# full_dataset = DiffusionDBDataset(raw_dataset, transform=transform)

raw_dataset = load_dataset("mnist", split="train")

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        img = item["image"].convert("RGB")
        label = str(item["label"]) 
        if self.transform:
            img = self.transform(img)
        return img, label

# Same transform you had, resized to 256x256
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create dataset
full_dataset = MNISTDataset(raw_dataset, transform=transform)

# tiny validation to keep it fast

train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

step = 0 
loss_history = []
scaler = GradScaler()  


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*epochs-warm_up, eta_min=1e-6)

for e in range(epochs):
    epoch_pbar = tqdm(train_loader, desc=f"Epoch {e+1}/{epochs}", leave=False)
    for i, (image, labels) in enumerate(epoch_pbar, start=1):
        step += 1
        epoch = step / len(train_loader)
        image = image.to(device, non_blocking=True)

        # Encode prompts
        prompts_emb, prompts_atnmask = tokenize_texts(labels, device)

        # Latents
        with torch.no_grad():
            z_q = vqvae.encoder(image)
            z, _, _ = vqvae.codebook(z_q)  

        latents = z.to(device)
        latents_noisy, timestep, noise = add_random_noise(latents)

        # Mixed precision forward
        with autocast():  
            noise_pred = transformer(latents_noisy, timestep, prompts_emb, prompts_atnmask)
            loss = F.mse_loss(noise_pred, noise - latents)
            loss = loss / accum_steps

        # Backward with scaler
        scaler.scale(loss).backward()

        # Gradient step
        if i % accum_steps == 0 or i == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            scheduler.step() 

        loss_history.append(loss.item() * accum_steps)
        epoch_pbar.set_postfix({"step": step, "loss": loss.item() * accum_steps})

        # Every 500 steps, save image and loss curve
        if step % print_each == 0:
            pil_img = generate(print_prompts, transformer, dcae)
            widths, heights = zip(*(img.size for img in pil_img))

            total_width = sum(widths)
            max_height = max(heights)
            combined = Image.new('RGB', (total_width, max_height))

            x_offset = 0
            for img in pil_img:
                combined.paste(img, (x_offset, 0))
                x_offset += img.width
            combined.save(f"logs_tr/{step}_combined.png")

            plt.figure(figsize=(6,4))
            plt.plot(loss_history, label="train loss")
            plt.xlabel("steps")
            plt.ylabel("loss")
            plt.title("Training Loss Curve")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"loss_curve.png")
            plt.close()