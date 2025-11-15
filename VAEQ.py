import torch
from torch import nn
import torch.nn.functional as F
import math

# -------------------------
# (your existing blocks)
# -------------------------
class SelfAttention(nn.Module):
  def __init__(self, n_heads, embd_dim, in_proj_bias=True, out_proj_bias=True):
    super().__init__()
    self.n_heads = n_heads
    self.in_proj = nn.Linear(embd_dim, 3 * embd_dim, bias=in_proj_bias)
    self.out_proj = nn.Linear(embd_dim, embd_dim, bias=out_proj_bias)
    self.d_heads = embd_dim // n_heads

  def forward(self, x, casual_mask=False):
    batch_size, seq_len, d_emed = x.shape
    interim_shape = (batch_size, seq_len, self.n_heads, self.d_heads)
    q, k, v = self.in_proj(x).chunk(3, dim=-1)
    q = q.view(interim_shape); k = k.view(interim_shape); v = v.view(interim_shape)
    q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
    weight = q @ k.transpose(-1, -2)
    if casual_mask:
        mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
        weight.masked_fill_(mask, -torch.inf)
    weight /= math.sqrt(self.d_heads)
    weight = F.softmax(weight, dim=-1)
    output = weight @ v
    output = output.transpose(1, 2)
    output = output.reshape((batch_size, seq_len, d_emed))
    output = self.out_proj(output)
    return output

class AttentionBlock(nn.Module):
  def __init__(self, channels):
      super().__init__()
      self.groupnorm = nn.GroupNorm(32, channels)
      self.attention = SelfAttention(1, channels)
  def forward(self, x):
      residual = x.clone()
      x = self.groupnorm(x)
      n, c, h, w = x.shape
      x = x.view((n, c, h * w))
      x = x.transpose(-1, -2)
      x = self.attention(x)
      x = x.transpose(-1, -2)
      x = x.view((n, c, h, w))
      x += residual
      return x

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.groupnorm1 = nn.GroupNorm(32, in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.groupnorm2 = nn.GroupNorm(32, out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    if in_channels == out_channels:
      self.residual_layer = nn.Identity()
    else:
      self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
  def forward(self, x):
    residue = x.clone()
    x = self.groupnorm1(x); x = F.selu(x); x = self.conv1(x)
    x = self.groupnorm2(x); x = self.conv2(x)
    return x + self.residual_layer(residue)

class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 64, kernel_size=3, padding=1),   # → 64 channels for latent
            nn.Conv2d(64, 64, kernel_size=1, padding=0)    # optional 1x1 conv
        )

    def forward(self, x):
        for module in self:
            # Handle uneven padding for stride=2 convs (256 → 128 → 64 → 32)
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))  # pad right and bottom
            x = module(x)
        return x  # [B, 64, 32, 32] for 256x256 input

class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            
            nn.Upsample(scale_factor=2),  # 32 → 64
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            
            nn.Upsample(scale_factor=2),  # 64 → 128
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            
            nn.Upsample(scale_factor=2),  # 128 → 256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),  # ← FIXED: was 64 → now 128
            nn.Tanh()  # ← CRITICAL: output in [-1, 1] to match input
        )

    def forward(self, x):
        for module in self:
            x = module(x)
        return x  # [B, 3, 256, 256]

# -------------------------
# Vector Quantizer + VQVAE
# -------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=8192, embedding_dim=64, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps
        
        # Codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_(0, 1)
        
        # EMA tracking
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embedding_avg', self.embedding.weight.data.clone())

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        # Compute distances
        distances = torch.cdist(z_flat, self.embedding.weight, p=2).pow(2)
        indices = torch.argmin(distances, dim=1)
        
        # Quantize
        z_q_flat = self.embedding.weight[indices]
        z_q = z_q_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # EMA update (only in training)
        if self.training:
            # 🔥 CRITICAL: Detach indices and z_flat to break gradient flow
            indices_det = indices.detach()
            z_flat_det = z_flat.detach()  # ← ADD THIS
            
            encodings = F.one_hot(indices_det, self.num_embeddings).float()
            self.cluster_size = self.decay * self.cluster_size + (1 - self.decay) * encodings.sum(0)
            
            # Update embedding averages
            dw = torch.matmul(encodings.t(), z_flat_det)  # ← use detached z_flat
            self.embedding_avg = self.decay * self.embedding_avg + (1 - self.decay) * dw
            
            # Laplace smoothing
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
            embedding_normalized = self.embedding_avg / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embedding_normalized)
    
        # Straight-through estimator (this part is correct)
        z_q_st = z_e + (z_q - z_e).detach()
        indices = indices.view(B, H * W)
        return z_q_st, indices, z_q

class VQVAE(nn.Module):
    """
    Wraps your Encoder + VectorQuantizer + Decoder.
    Forward returns: recon, tokens, z_e, z_q
    """
    def __init__(self, num_embeddings=8192):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # encoder produces channels C = 4 (by architecture). Keep embedding_dim=C.
        self.embedding_dim = 64
        self.codebook = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=self.embedding_dim)

    def forward(self, x):
        # x: [B, 3, 256, 256]
        # z_e: encoder output (stochastic in your current encoder)
        z_e = self.encoder(x)               # [B, C, H, W], C==4
        # quantize
        z_q_st, indices, z_q = self.codebook(z_e)
        # decode quantized latents
        recon = self.decoder(z_q_st)
        # tokens returned as indices [B, H*W]
        tokens = indices
        return recon, tokens, z_e, z_q

    def encode(self, x):
        # returns discrete tokens
        z_e = self.encoder(x)
        _, indices, _ = self.codebook(z_e)
        return indices

    def decode(self, tokens):
        B, N = tokens.shape
        H = W = int(N**0.5)
        embeds = self.codebook.embedding(tokens)  # [B, N, C]
        z_q = embeds.view(B, H, W, self.embedding_dim).permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)