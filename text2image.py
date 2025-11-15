import torch
import torch.nn.functional as F

from VAEQ import VQVAE
from model import Transformer, config
import torchvision.utils as vutils
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()

#### some configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#### tokenizer
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
config.vocab_size = len(tokenizer)

def tokenize_texts(texts, max_len=config.max_text_len, device="cpu", drop_prob=0.0):
    drop_mask = torch.rand(len(texts)) < drop_prob  # [True, False, True, ...]
    
    # Replace dropped texts with empty string
    processed_texts = [
        "" if drop_mask[i] else texts[i] 
        for i in range(len(texts))
    ]
    
    # Tokenize normally
    encoded = tokenizer(
        processed_texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    input_ids = encoded.input_ids.to(device)   
    
    return input_ids

## for loss
vqvae = VQVAE().to(device)
vqvae.load_state_dict(torch.load("save/VAEQ_model-3.pth"))  # or best epoch
vqvae.eval()
print("VQVAE loaded!")


### samele
def denormalize(tensor):
    return (tensor + 1) / 2
def sample_and_save(model, vqvae, text, filename, device, guidance_scale=3.0, temperature=1.5):
    model.eval()
    with torch.no_grad():
        # Tokenize conditional prompt
        cond_tokens= tokenize_texts([text], device=device, drop_prob=0.0)
        
        # Tokenize unconditional (empty) prompt
        uncond_tokens= tokenize_texts([""], device=device, drop_prob=0.0)
        
        # Get logits
        cond_logits = model(cond_tokens)      # [1, 1024, 8192]
        uncond_logits = model(uncond_tokens) 
        
        # Apply CFG
        guided_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
        
        # Sample tokens from guided logits
        probs = F.softmax(guided_logits / temperature, dim=-1)
        generated_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1])
        
        # Decode to image
        image = vqvae.decode(generated_tokens)
        vutils.save_image(denormalize(image), filename, normalize=False)
        image = denormalize(image)
        pil_img = to_pil(image[0].cpu())

        pil_img.show()
    
    model.train()

model = Transformer().to(device)
model.load_state_dict(torch.load("save_tr/transformer_epoch_6.pth"))
model.eval()
print("Trained model loaded!")


if __name__ == "__main__":
    while True:
        prompt = input("Prompt Musie: ")
        sample_and_save(model, vqvae, prompt, f"generated/img-[{prompt}].png", device)
