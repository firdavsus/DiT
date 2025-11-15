import torch
from transformers import GPT2Tokenizer
from model import Transformer, config
import torchvision.utils as vutils
import torch.nn.functional as F
from VAEQ import VQVAE

# --- Tokenizer setup ---
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Add special tokens
special_tokens_dict = {
    'additional_special_tokens': [
        '<|startoftext|>', 
        '<|endoftext|>', 
        '<|startofimage|>'
    ]
}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added_toks} special tokens")

# Update config
config.vocab_size = len(tokenizer)
config.start_text_token = tokenizer.convert_tokens_to_ids('<|startoftext|>')
config.end_text_token = tokenizer.convert_tokens_to_ids('<|endoftext|>')
config.start_image_token = tokenizer.convert_tokens_to_ids('<|startofimage|>')

def tokenize_texts(texts, max_len=config.max_text_len, device="cpu", drop_prob=0.3):
    # Apply dropout (30% empty prompts)
    drop_mask = torch.rand(len(texts)) < drop_prob
    processed_texts = [
        "" if drop_mask[i] else texts[i] 
        for i in range(len(texts))
    ]
    
    # Add boundary tokens to non-empty texts
    final_texts = []
    for text in processed_texts:
        if text == "":
            # Empty prompt: just start/end tokens
            final_texts.append(f"<|startoftext|> <|endoftext|>")
        else:
            # Non-empty: wrap with boundary tokens
            final_texts.append(f"<|startoftext|> {text} <|endoftext|> <|startofimage|>")
    
    # Tokenize with boundary tokens
    encoded = tokenizer(
        final_texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    return encoded.input_ids.to(device)

# --- Device + model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer().to(device)
model.load_state_dict(torch.load("save_tr/transformer_epoch_7.pth", map_location=device))
model.eval()

def denormalize(tensor):
    return (tensor + 1) / 2

def sample_and_save(model, vqvae, text, filename, device, guidance_scale=5.0, temperature=1.0):
    with torch.no_grad():
        # Tokenize conditional prompt
        cond_tokens= tokenize_texts([text], device=device, drop_prob=0.0)
        
        # Tokenize unconditional (empty) prompt
        uncond_tokens= tokenize_texts([""], device=device, drop_prob=0.0)
        
        # Get logits
        cond_logits = model(cond_tokens)      # [1, 1024, 8192]
        uncond_logits = model(uncond_tokens)  # [1, 1024, 8192]
        
        # Apply CFG
        guided_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
        
        # Sample tokens from guided logits
        probs = F.softmax(guided_logits / temperature, dim=-1)
        generated_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1])
        
        # Decode to image
        image = vqvae.decode(generated_tokens)
        vutils.save_image(denormalize(image), filename, normalize=False)

vqvae = VQVAE().to(device)
vqvae.load_state_dict(torch.load("save/VAEQ_model-3.pth"))  # or best epoch
vqvae.eval()

# --- Interactive generation ---
if __name__ == "__main__":
    while True:
        prompt = input("Enter a prompt: ")
        img =sample_and_save(model, vqvae, prompt, "example.png", device)