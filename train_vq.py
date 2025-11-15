from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torch
import os
import json
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import torchvision.utils as vutils

from VAEQ import VQVAE
from torchvision.models import vgg16

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = vgg16(pretrained=True).features.eval().to(device)
        # Use relu3_3 (layer 16) and relu4_3 (layer 23)
        self.layers = nn.Sequential(*list(vgg.children())[:23])  # up to relu4_3
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        return F.l1_loss(self.layers(pred), self.layers(target))

#### some configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
accum_steps = 1
num_epochs = 100



class TextImageDataset(Dataset):
    def __init__(self, images_dir, json_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        with open(json_path, "r", encoding="utf-8") as f:
            self.descriptions = json.load(f)

        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.jpeg','.png', '.PNG', '.webp'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        key = os.path.splitext(image_name)[0]
        description = self.descriptions.get(key, "")
        return image, description

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = TextImageDataset("../all_images", "../captions.json", transform=transform)

old_state_dict = torch.load("save/VAEQ_model-10.pth")

model = VQVAE()
model.load_state_dict(torch.load("save/VAEQ_model-10.pth"))
# new_state_dict = model.state_dict()

# # Copy all compatible weights from old checkpoint
# for key in old_state_dict:
#     if key in new_state_dict:
#         new_state_dict[key] = old_state_dict[key]
#     else:
#         print(f"Skipping unexpected key: {key}")

# # Initialize EMA-specific buffers from embedding weights
# embedding_weight = new_state_dict["codebook.embedding.weight"]
# num_embeddings, embedding_dim = embedding_weight.shape

# # cluster_size: start with uniform usage (or zeros)
# new_state_dict["codebook.cluster_size"] = torch.zeros(num_embeddings)

# # embedding_avg: start as embedding_weight * initial cluster_size
# # Since cluster_size=0, we initialize as embedding_weight (common practice)
# new_state_dict["codebook.embedding_avg"] = embedding_weight.clone()
# model.load_state_dict(new_state_dict, strict=True)

# del old_state_dict
# del new_state_dict
# torch.cuda.empty_cache()
model = model.to(device)

criterion = nn.MSELoss()

scaler = torch.cuda.amp.GradScaler()
opt = torch.optim.Adam(model.parameters(), lr=3e-5)
perceptual_loss = PerceptualLoss(device)

def denormalize(tensor):
    return (tensor + 1) / 2


for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,     
    )

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
    train_loss=0.0
    
    for step, (images, texts) in enumerate(pbar):
        images = images.to(device)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            recon, tokens, z_e, z_q = model(images)
            l1_loss = F.l1_loss(recon, images)  # ← L1 instead of MSE
        

        perceptual_val = perceptual_loss(recon.float(), images.float())
        
        recon_loss = 0.8 * l1_loss + 0.2 * perceptual_val
        
        # VQ losses
        z_e = z_e.float()
        z_q = z_q.float()
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        
        beta = 0.25
        loss = recon_loss + codebook_loss + beta * commitment_loss

        loss = loss / accum_steps

        scaler.scale(loss).backward()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss+=loss.item()
        
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            

        if step%50==0:
            model.eval()
            with torch.no_grad():
                input_denorm = denormalize(images[:4])
                recon_denorm = denormalize(recon[:4])
                comparison = torch.cat([input_denorm, recon_denorm], dim=0)
                vutils.save_image(comparison, "recon_comparison.png", nrow=4)
            model.train()

        if step%500==0:
            torch.save(model.state_dict(), f"save/VAEQ_model-{epoch+1}.pth")


    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss / len(dataloader):.4f}")
    torch.save(model.state_dict(), f"save/VAEQ_model-{epoch+1}.pth")