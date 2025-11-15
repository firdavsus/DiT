from PIL import Image
import torch
import torchvision.utils as vutils
from torchvision import transforms

from VAEQ import VQVAE

#### some configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VQVAE()
model.load_state_dict(torch.load("save/VAEQ_model-1.pth"))
model = model.to(device)

def denormalize(tensor):
    return (tensor + 1) / 2

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load images
imgs = []
for path in ["1.png", "2.png", "3.png"]:
    img = Image.open(path).convert("RGB")
    img = transform(img)
    imgs.append(img)

# Create batch
images = torch.stack(imgs).to(device)   # ✅ [B, 3, 256, 256]

model.eval()
with torch.no_grad():
    recon, tokens, z_e, z_q = model(images)

    input_denorm = denormalize(images)
    recon_denorm = denormalize(recon)

    comparison = torch.cat([input_denorm, recon_denorm], dim=0)
    vutils.save_image(comparison, "recon_comparison.png", nrow=3)