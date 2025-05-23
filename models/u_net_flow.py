import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def safe_groupnorm(ch: int, desired_groups: int = 32) -> nn.GroupNorm:
    """Applies GroupNorm, finding a valid number of groups."""
    if ch == 0:
        return nn.Identity()
    groups = desired_groups
    while ch % groups != 0 and groups > 1:
        groups -= 1
    if ch % groups != 0:
        groups = 1
    return nn.GroupNorm(groups, ch)


class TimeEmbedding(nn.Module):
    """Embeds scalar time steps into a vector space using sinusoidal embeddings."""

    def __init__(self, dim: int, embed_dim_factor: int = 4):
        super().__init__()
        hidden_dim = dim * embed_dim_factor
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.dim = dim
        freqs = torch.exp(-math.log(10000) * torch.arange(dim // 2) / (dim // 2 - 1))
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t.unsqueeze(-1) * self.freqs
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 != 0:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding)


class PreActResBlock(nn.Module):
    """Pre-Activation Residual Block with FiLM conditioning based on time embedding."""

    def __init__(
        self, ch_in: int, ch_out: int, t_dim: int, drop: float, res_scale: float = 1.0
    ):
        super().__init__()
        self.n1 = safe_groupnorm(ch_in)
        self.n2 = safe_groupnorm(ch_out)
        self.c1 = nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=False)
        self.c2 = nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=False)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.res_scale = nn.Parameter(torch.ones(ch_out).view(1, -1, 1, 1))

        # FiLM layers use the time embedding dimension
        self.film1 = nn.Linear(t_dim, ch_in * 2) if t_dim > 0 else None
        self.film2 = nn.Linear(t_dim, ch_out * 2) if t_dim > 0 else None

        self.skip = (
            nn.Conv2d(ch_in, ch_out, 1, bias=False)
            if ch_in != ch_out
            else nn.Identity()
        )

    def _apply_film(self, x, film_params):
        """Applies Feature-wise Linear Modulation (FiLM)."""
        scale, shift = film_params.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + scale) + shift

    def forward(self, x, t_emb):
        res = x

        h = self.n1(x)
        h = F.silu(h)
        if self.film1 and t_emb is not None:
            film_params1 = self.film1(t_emb)
            h = self._apply_film(h, film_params1)
        h = self.c1(h)

        h = self.n2(h)
        h = F.silu(h)
        if self.film2 and t_emb is not None:
            film_params2 = self.film2(t_emb)
            h = self._apply_film(h, film_params2)
        h = self.c2(h)
        h = self.drop(h)

        skip_out = self.skip(res)
        return skip_out + self.res_scale * h


class SpatialAttention(nn.Module):
    """Simple Self-Attention block for 2D feature maps."""

    def __init__(self, ch: int, num_heads: int = 8, head_dim: int = None):
        super().__init__()
        self.num_heads = num_heads
        calculated_head_dim = ch // num_heads
        self.head_dim = head_dim or (
            calculated_head_dim if calculated_head_dim > 0 else 1
        )
        self.inner_dim = self.head_dim * self.num_heads
        if self.inner_dim > ch:
            self.head_dim = ch // self.num_heads
            if self.head_dim == 0:
                self.num_heads = ch
                self.head_dim = 1
            self.inner_dim = self.head_dim * self.num_heads

        self.scale = self.head_dim**-0.5
        self.norm = safe_groupnorm(ch)
        self.to_qkv = nn.Conv2d(ch, self.inner_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(self.inner_dim, ch, kernel_size=1, bias=False)
        nn.init.zeros_(self.to_out.weight)

    def forward(self, x):
        B, C, H, W = x.shape
        res = x

        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(
                t, "b (h d) x y -> b h (x y) d", h=self.num_heads, d=self.head_dim
            ),
            qkv,
        )

        # Attention mechanism
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(
            out,
            "b h (x y) d -> b (h d) x y",
            x=H,
            y=W,
            h=self.num_heads,
            d=self.head_dim,
        )

        out = self.to_out(out)
        return res + out


class Encoder(nn.Module):
    """U-Net Encoder Path with Time-Conditioned ResBlocks."""

    def __init__(
        self,
        in_ch: int,
        filters: list,
        t_dim: int,
        drop: float,
        depth: int,
        use_attn: list,
        res_scale: float,
    ):
        super().__init__()
        self.filters = filters
        self.depth = depth
        self.blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        current_ch = in_ch

        self.blocks.append(
            nn.Conv2d(in_ch, filters[0], kernel_size=3, padding=1, bias=False)
        )
        current_ch = filters[0]

        for i, ch_out in enumerate(filters):
            for _ in range(depth):
                block_in_ch = current_ch
                block_out_ch = ch_out
                self.blocks.append(
                    PreActResBlock(
                        block_in_ch, block_out_ch, t_dim, drop, res_scale=res_scale
                    )
                )
                current_ch = block_out_ch

            self.attn_blocks.append(
                SpatialAttention(ch_out) if use_attn[i] else nn.Identity()
            )

            if i != len(filters) - 1:
                next_ch = filters[i + 1]
                self.blocks.append(
                    nn.Conv2d(
                        ch_out, next_ch, kernel_size=3, stride=2, padding=1, bias=False
                    )
                )
                current_ch = next_ch

    def forward(self, x, t_emb):
        skips = []
        block_idx = 0
        attn_idx = 0

        x = self.blocks[block_idx](x)
        block_idx += 1

        for i in range(len(self.filters)):
            for _ in range(self.depth):
                x = self.blocks[block_idx](x, t_emb)
                block_idx += 1

            x = self.attn_blocks[attn_idx](x)
            attn_idx += 1

            if i != len(self.filters) - 1:
                skips.append(x)
                x = self.blocks[block_idx](x)
                block_idx += 1

        return x, skips


class Decoder(nn.Module):
    """U-Net Decoder Path with Time-Conditioned ResBlocks and Skip Connections."""

    def __init__(
        self,
        filters: list,
        t_dim: int,
        drop: float,
        depth: int,
        use_attn: list,
        res_scale: float,
    ):
        super().__init__()
        self.filters = filters
        self.depth = depth
        self.blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        rev_filters = list(reversed(filters))
        rev_use_attn = list(reversed(use_attn))
        current_ch = rev_filters[0]

        for i, ch_level in enumerate(rev_filters[:-1]):
            ch_skip = rev_filters[i + 1]
            ch_upsampled = ch_skip

            self.blocks.append(
                nn.ConvTranspose2d(current_ch, ch_upsampled, kernel_size=2, stride=2)
            )
            current_ch = ch_upsampled

            res_in_ch = current_ch + ch_skip

            for j in range(depth):
                block_in = res_in_ch if j == 0 else ch_upsampled
                block_out = ch_upsampled
                self.blocks.append(
                    PreActResBlock(
                        block_in, block_out, t_dim, drop, res_scale=res_scale
                    )
                )
                current_ch = block_out
            self.attn_blocks.append(
                SpatialAttention(current_ch) if rev_use_attn[i + 1] else nn.Identity()
            )

    def forward(self, x, skips, t_emb):
        skips = list(reversed(skips))
        block_idx = 0
        attn_idx = 0

        for i in range(len(self.filters) - 1):
            x = self.blocks[block_idx](x)
            block_idx += 1

            skip = skips[i]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

            for _ in range(self.depth):
                x = self.blocks[block_idx](x, t_emb)
                block_idx += 1

            x = self.attn_blocks[attn_idx](x)
            attn_idx += 1

        return x


class ResBlock(nn.Module):
    """Simple Residual Block for non-time-conditioned features."""

    def __init__(self, ch_in: int, ch_out: int, drop: float):
        super().__init__()
        self.n1 = safe_groupnorm(ch_in)
        self.n2 = safe_groupnorm(ch_out)
        self.c1 = nn.Conv2d(ch_in, ch_out, 3, 1, 1, bias=False)
        self.c2 = nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=False)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.skip = (
            nn.Conv2d(ch_in, ch_out, 1, bias=False)
            if ch_in != ch_out
            else nn.Identity()
        )

    def forward(self, x):
        res = x
        h = self.n1(x)
        h = F.silu(h)
        h = self.c1(h)
        h = self.n2(h)
        h = F.silu(h)
        h = self.c2(h)
        h = self.drop(h)
        return self.skip(res) + h


class UNetSR(nn.Module):
    """
    U-Net for Super-Resolution using spatial LR guidance and time conditioning.
    """

    def __init__(
        self,
        in_ch: int = 6,
        out_ch: int = 3,
        filters: list = None,
        t_dim: int = 256,
        drop: float = 0.1,
        depth: int = 3,
        attn_levels: list = None,
        num_attn_heads: int = 8,
        use_bottleneck_attn: bool = True,
        res_block_scale: float = 1.0,
    ):
        super().__init__()
        filters = filters or [64, 128, 256, 512]
        self.filters = filters
        self.depth = depth
        self.t_dim = t_dim
        attn_levels = attn_levels or [
            False,
            False,
            True,
            True,
        ]
        if len(attn_levels) != len(filters):
            raise ValueError("Length of attn_levels must match length of filters")

        self.time_mlp = TimeEmbedding(t_dim)
        self.lr_res = ResBlock(3, 3, drop=drop)

        self.encoder = Encoder(
            in_ch, filters, t_dim, drop, depth, attn_levels, res_block_scale
        )
        self.decoder = Decoder(
            filters, t_dim, drop, depth, attn_levels, res_block_scale
        )

        self.bottleneck_blocks = nn.ModuleList(
            [
                PreActResBlock(filters[-1], filters[-1], t_dim, drop),
                (
                    SpatialAttention(filters[-1], num_heads=num_attn_heads)
                    if use_bottleneck_attn
                    else nn.Identity()
                ),
                nn.Conv2d(filters[-1], filters[-1], kernel_size=1),
                nn.SiLU(),
                nn.Dropout2d(drop),
                PreActResBlock(filters[-1], filters[-1], t_dim, drop, res_scale=0.0),
            ]
        )

        self.final_norm = safe_groupnorm(filters[0])
        self.final_conv = nn.Conv2d(filters[0], out_ch, kernel_size=3, padding=1)

        self.apply(self._initialize_weights)
        nn.init.zeros_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        attn_enabled_str = f"AttnLvls={attn_levels}, BottleneckAttn={use_bottleneck_attn}, Heads={num_attn_heads}"
        print(
            f"UNetSR Initialized: {num_params/1e6:.2f}M params (depth={depth}, filters={filters}, {attn_enabled_str})"
        )

    def _initialize_weights(self, m):
        """Initializes weights using Kaiming normal/uniform and standard norm init."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            try:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            except ValueError:
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_t, t, lr_up):
        """
        Forward pass through the U-Net. lr16 is unused in this version.

        Args:
            x_t (torch.Tensor): Noisy high-res image [B, 3, H, W].
            t (torch.Tensor): Timesteps [B].
            lr_up (torch.Tensor): Upscaled low-res image [B, 3, H, W].

        Returns:
            torch.Tensor: Predicted output (velocity/noise) [B, out_ch, H, W].
        """
        t = t.to(x_t.device)
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(x_t.size(0))
        elif t.ndim == 1 and t.size(0) == 1 and x_t.size(0) > 1:
            t = t.expand(x_t.size(0))
        elif t.ndim != 1 or t.size(0) != x_t.size(0):
            raise ValueError(
                f"Time tensor shape {t.shape} incompatible with batch size {x_t.size(0)}"
            )
        t_emb = self.time_mlp(t)

        # Process upscaled LR condition
        lr_up_feat = self.lr_res(lr_up)
        unet_input = torch.cat([x_t, lr_up_feat], dim=1)

        # U-Net Forward Pass
        x, skips = self.encoder(unet_input, t_emb)
        for block in self.bottleneck_blocks:
            if isinstance(block, PreActResBlock):
                x = block(x, t_emb)
            else:
                x = block(x)
        x = self.decoder(x, skips, t_emb)

        # Final processing
        x = self.final_norm(x)
        x = F.silu(x)
        output = self.final_conv(x)

        return output
