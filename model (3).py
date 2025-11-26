import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import T5EncoderModel

class Config:
    model_dim = 960
    num_heads = 15
    hidden_dim_mul = 4
    num_layers = 16
    patch_size = 2
    num_latents = 32
    drop_text_prob = 0.1
    max_text_len = 50
    t5_dim = 960

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

class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(in_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # RoPE frequencies precomputed for max grid size (32x32)
        self.max_grid_size = 32
        self._create_rope_freqs()
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _create_rope_freqs(self):
        # Just store inv_freq — compute positions dynamically
        dim = self.head_dim // 2  # Half for h, half for w
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def _apply_rope(self, x, grid_h, grid_w):
        B, H, N, D = x.shape
        assert N == grid_h * grid_w
        
        # Create 2D position IDs
        device = x.device
        pos_h = torch.arange(grid_h, device=device).view(-1, 1).repeat(1, grid_w).flatten()
        pos_w = torch.arange(grid_w, device=device).view(1, -1).repeat(grid_h, 1).flatten()
        
        # Compute RoPE frequencies for h and w
        freqs_h = torch.einsum("n,d->nd", pos_h.float(), self.inv_freq)  # [N, D//4]
        freqs_w = torch.einsum("n,d->nd", pos_w.float(), self.inv_freq)  # [N, D//4]
        
        # Combine and expand to full dimension
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)  # [N, D//2]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # [N, D]
        
        cos = emb.cos().view(1, 1, N, D)
        sin = emb.sin().view(1, 1, N, D)
        
        x1, x2 = x.chunk(2, dim=-1)
        x_rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + x_rot * sin

    def forward(self, x, context=None, mask=None, grid_size=None):
        # x: (B, N_q, D)
        B, N_q, _ = x.shape
        if context is None:
            context = x
            is_self_attn = True  
        else:
            is_self_attn = False  
        
        N_k = context.shape[1]

        # Get grid dimensions for RoPE (only needed for self-attention)
        if grid_size is None and is_self_attn:
            grid_h = grid_w = int(math.sqrt(N_q))
        elif grid_size is not None and is_self_attn:
            grid_h, grid_w = grid_size
        else:
            # For cross-attention, we don't need grid_size for RoPE
            grid_h = grid_w = None

        # project
        q = self.q_proj(x)                         # (B, N_q, D)
        k = self.k_proj(context)                   # (B, N_k, D)
        v = self.v_proj(context)                   # (B, N_k, D)

        # reshape -> (B, num_heads, N, head_dim)
        q = q.view(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        if is_self_attn and grid_h is not None and grid_w is not None:
            if grid_h * grid_w == N_q:
                q = self._apply_rope(q, grid_h, grid_w)
            if grid_h * grid_w == N_k:  # Only if context is also spatial
                k = self._apply_rope(k, grid_h, grid_w)

        # attention scores (B, heads, N_q, N_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided (for cross-attention)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[:, None, None, :]  # (B, 1, 1, N_k)
            attn_scores = attn_scores.masked_fill(~mask.bool(), -10000.0)

        # numerically stable softmax in float32, then cast back
        attn = F.softmax(attn_scores.float(), dim=-1).type_as(attn_scores)

        out = torch.matmul(attn, v)                       # (B, heads, N_q, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N_q, self.embed_dim)
        return self.out_proj(out)

class AdaLNZero(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.affine = nn.Linear(embed_dim, 2 * embed_dim)
        # Zero init: critical!
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x, t_emb):
        scale, shift = self.affine(t_emb).chunk(2, dim=-1)  # (B, D)
        x = F.layer_norm(x, x.shape[-1:], weight=None, bias=None)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        # AdaLN-Zero for each pathway (ZERO INITIALIZED)
        self.adaLN_self_attn = AdaLNZero(config.model_dim)
        self.adaLN_cross_attn = AdaLNZero(config.model_dim)
        self.adaLN_mlp = AdaLNZero(config.model_dim)

        self.self_attn = MultiHeadAttention(config.model_dim, config.num_heads)
        self.cross_attn = MultiHeadAttention(config.model_dim, config.num_heads)

        # Time projection (split into 3 conditioning vectors)
        self.t_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.model_dim, 3 * config.model_dim)
        )
        nn.init.zeros_(self.t_emb_proj[-1].weight)
        nn.init.zeros_(self.t_emb_proj[-1].bias)

        # CORRECT SwiGLU usage
        self.mlp = SwiGLU(
            in_dim=config.model_dim,
            out_dim=config.model_dim,
            hidden_dim=config.model_dim * config.hidden_dim_mul
        )

    def forward(self, x, t_emb, context_emb, mask=None, grid_size=None):
        t_biases = self.t_emb_proj(t_emb)
        t_sa, t_ca, t_mlp = t_biases.chunk(3, dim=-1)  # (B, D) each

        # Self-Attention Path (with spatial grid_size)
        x_res = x
        x = self.adaLN_self_attn(x, t_sa)
        x = self.self_attn(x, grid_size=grid_size) 
        x = x_res + x

        # Cross-Attention Path (no grid_size needed)
        x_res = x
        x = self.adaLN_cross_attn(x, t_ca)
        x = self.cross_attn(x, context=context_emb, mask=mask) 
        x = x_res + x

        # MLP Path
        x_res = x
        x = self.adaLN_mlp(x, t_mlp)
        x = self.mlp(x)
        x = x_res + x

        return x


class DiT(nn.Module):
    def __init__(self, learn_sigma=False):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.out_channels = config.num_latents * (2 if learn_sigma else 1)

        self.in_proj = nn.Linear(config.num_latents, config.model_dim)

        self.blocks = nn.ModuleList([Block() for _ in range(config.num_layers)])

        self.final_layer = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, self.out_channels)
        )
        
        # Time embedding MLP
        self.mlp_time = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * config.hidden_dim_mul),
            nn.SiLU(),
            nn.Linear(config.model_dim * config.hidden_dim_mul, config.model_dim)
        )

        # Text projection
        self.text_proj = nn.Sequential(
            RMSNorm(config.t5_dim),
            nn.Linear(config.t5_dim, config.model_dim)
        )
        self.input_norm = RMSNorm(config.model_dim)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._basic_init)
        
        # Initialize in/out projections like nn.Linear
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        
        nn.init.xavier_uniform_(self.final_layer[1].weight)
        nn.init.zeros_(self.final_layer[1].bias)

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
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        N = H * W
        
        # Get timestep embeddings
        t_emb = get_sinusoidal_timestep_embedding(t, config.model_dim)
        t_emb = self.mlp_time(t_emb)

        # Project text embeddings
        text_emb = self.text_proj(text_ids)

        x = x.permute(0, 2, 3, 1).reshape(B, N, C)  # [B, N, C]
        x = self.in_proj(x)  # [B, N, D]
        x = self.input_norm(x)

        # Apply transformer blocks - PASS GRID SIZE FOR RoPE
        for block in self.blocks:
            x = block(x, t_emb, text_emb, attn_mask, grid_size=(H, W))

        # Final projection
        x = self.final_layer(x)  # [B, N, out_channels]

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x
