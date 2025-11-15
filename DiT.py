import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import T5EncoderModel

class Config:
    model_dim = 256
    num_heads = 4
    hidden_dim_mul = 4
    num_layers = 4
    patch_size = 2
    img_size = 32
    num_latents = 64  # VAE latent channels
    text_encoder_name = "t5-small"
    drop_text_prob = 0.1 
    max_text_len = 512
    t5_dim = 512

config = Config()

def get_sinusoidal_timestep_embedding(timesteps, dim):
    device = timesteps.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:  # zero pad
        emb = torch.cat([emb, torch.zeros(timesteps.shape[0], 1, device=device)], dim=-1)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, context=None, mask=None):
        # x: (B, N_q, D)
        B, N_q, _ = x.shape
        if context is None:
            context = x
        N_k = context.shape[1]

        # project
        q = self.q_proj(x)                         # (B, N_q, D)
        k = self.k_proj(context)                   # (B, N_k, D)
        v = self.v_proj(context)                   # (B, N_k, D)

        # reshape -> (B, num_heads, N, head_dim)
        q = q.view(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        # attention scores (B, heads, N_q, N_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # mask: expected shape (B, N_k) or None
        if mask is not None and mask.shape[1] == N_k:
            mask_bool = mask.to(torch.bool)
            # broadcast to (B, 1, 1, N_k)
            attn_scores = attn_scores.masked_fill(~mask_bool[:, None, None, :], float("-inf"))

        # numerically stable softmax in float32, then cast back
        attn = F.softmax(attn_scores.float(), dim=-1).type_as(attn_scores)

        out = torch.matmul(attn, v)                       # (B, heads, N_q, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N_q, self.embed_dim)
        return self.out_proj(out)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.model_dim)
        self.norm2 = nn.LayerNorm(config.model_dim)
        self.norm3 = nn.LayerNorm(config.model_dim)
        
        self.self_attn = MultiHeadAttention(config.model_dim, config.num_heads)
        self.cross_attn = MultiHeadAttention(config.model_dim, config.num_heads)
        
        # Time conditioning for each residual path
        self.t_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.model_dim, 3 * config.model_dim)
        )
        # Zero initialize for stability
        nn.init.zeros_(self.t_emb_proj[-1].weight)
        nn.init.zeros_(self.t_emb_proj[-1].bias)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * config.hidden_dim_mul),
            nn.GELU(),
            nn.Linear(config.model_dim * config.hidden_dim_mul, config.model_dim)
        )

    def forward(self, x, t_emb, context_emb, mask=None):
        # Get time biases (B, 3*D) -> split for each path
        t_biases = self.t_emb_proj(t_emb)  # (B, 3*D)
        t_sa, t_ca, t_mlp = t_biases.chunk(3, dim=-1)
        
        # Self-Attention with proper residual connection
        x_res = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = x_res + x + t_sa.unsqueeze(1)
        
        # Cross-Attention with proper residual connection
        x_res = x
        x = self.norm3(x)
        x = self.cross_attn(x, context=context_emb, mask=mask)
        x = x_res + x + t_ca.unsqueeze(1)
        
        # MLP with proper residual connection
        x_res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x_res + x + t_mlp.unsqueeze(1)
        
        return x

class PatchEmbed2D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=2, image_size=32):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"

        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.in_channels = in_channels
        self.image_size = image_size

        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        # Reshape to extract patches
        x = x.reshape(B, C, self.grid_size, p, self.grid_size, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # B, grid_h, grid_w, C, p, p
        x = x.reshape(B, self.num_patches, self.patch_dim)
        
        return self.proj(x)

    def unpatchify(self, x, out_channels):
        B, N, D = x.shape
        p = self.patch_size
        gs = self.grid_size
        H = W = self.image_size

        # Verify D matches expected
        assert D == p * p * out_channels, f"Expected D={p*p*out_channels}, got {D}"

        x = x.contiguous().view(B, gs, gs, out_channels, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, out_channels, H, W)

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)

        # time projection produces a vector in hidden_size (added to token features)
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # Zero init time proj last linear (stability)
        nn.init.zeros_(self.t_proj[-1].weight)
        nn.init.zeros_(self.t_proj[-1].bias)

        # final linear maps hidden -> output dims
        self.linear = nn.Linear(hidden_size, out_dim)
        # crucial: zero init final linear so model starts near identity for residual-like behavior
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, t_emb):
        # norm, add time to features, then project to output dims
        x = self.norm_final(x)                      # (B, N, D)
        time_bias = self.t_proj(t_emb).unsqueeze(1) # (B, 1, D)
        x = x + time_bias                           # inject time before linear
        x = self.linear(x)                          # (B, N, out_dim)
        return x


class DiT(nn.Module):
    def __init__(self, learn_sigma=False):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.out_channels = config.num_latents * (2 if learn_sigma else 1)
        self.patch = PatchEmbed2D(config.num_latents, config.model_dim, config.patch_size, config.img_size)
        
        # Fixed positional embeddings (non-learnable)
        num_patches = self.patch.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.model_dim), requires_grad=False)
        
        self.blocks = nn.ModuleList([Block() for _ in range(config.num_layers)])
        
        # Final layer with time conditioning
        final_out_dim = self.patch.patch_dim * (2 if learn_sigma else 1)
        self.final_layer = FinalLayer(config.model_dim, final_out_dim)
        
        # Time embedding MLP
        self.mlp_time = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * config.hidden_dim_mul),
            nn.SiLU(),
            nn.Linear(config.model_dim * config.hidden_dim_mul, config.model_dim)
        )
        
        # Text encoder (frozen)
        self.text_encoder = T5EncoderModel.from_pretrained(config.text_encoder_name)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        
        # Single text projection (shared across all layers)
        self.text_proj = nn.Sequential(
            nn.LayerNorm(config.t5_dim),  # T5-base hidden size
            nn.Linear(config.t5_dim, config.model_dim)
        )
        self.input_norm = nn.LayerNorm(config.model_dim)
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers
        self.apply(self._basic_init)
        
        # Initialize positional embeddings (fixed sine-cosine)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch.grid_size))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize patch projection like nn.Linear
        w = self.patch.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize timestep embedding MLP
        nn.init.normal_(self.mlp_time[0].weight, std=0.02)
        nn.init.normal_(self.mlp_time[2].weight, std=0.02)
        
        # Initialize text projection
        nn.init.normal_(self.text_proj[1].weight, std=0.02)
    
    def _basic_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x, t, text_ids, attn_mask=None):
        # Get timestep embeddings
        t_emb = get_sinusoidal_timestep_embedding(t, config.model_dim)
        t_emb = self.mlp_time(t_emb)  # (B, D)
        
        # Get text embeddings
        with torch.no_grad():
            text_emb = self.text_encoder(input_ids=text_ids, attention_mask=attn_mask).last_hidden_state
        
        # Project text embeddings
        text_emb = self.text_proj(text_emb)  # (B, N_text, D)
        
        # Patchify input and add positional embeddings
        x = self.patch(x)
        x = x + self.pos_embed  # (B, N_patches, D)
        x = self.input_norm(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, text_emb, attn_mask)
        
        # Final layer with time conditioning
        x = self.final_layer(x, t_emb)
        
        # Unpatchify to get final output - PASS OUT_CHANNELS CORRECTLY
        x = self.patch.unpatchify(x, self.out_channels)  # (B, out_channels, H, W)
        
        return x
    