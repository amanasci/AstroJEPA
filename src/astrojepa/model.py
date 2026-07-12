"""
Core JEPA model for AstroJEPA.

Architecture:
    - ContextEncoder:  ViT that processes visible (context) patches
    - TargetEncoder:   EMA copy of ContextEncoder, processes all patches
    - Predictor:       Small ViT that predicts target embeddings from context

References:
    - I-JEPA: Assran et al., 2023 (arXiv:2301.08243)
    - V-JEPA: Bardes et al., 2024 (arXiv:2404.08471)
"""

import math
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def new_gelu(x: torch.Tensor) -> torch.Tensor:
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
            )
        )
    )


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class MLP(nn.Module):
    def __init__(self, n_embd: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attn_mask is not None:
                att = att.masked_fill(attn_mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class CrossAttention(nn.Module):
    """Cross-attention: queries attend to external keys/values."""

    def __init__(self, n_embd: int, n_head: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, kv: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B_q, T_q, C = x.size()
        B_k, T_k, _ = kv.size()
        q = self.q_proj(x).view(B_q, T_q, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.k_proj(kv).view(B_k, T_k, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.v_proj(kv).view(B_k, T_k, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attn_mask is not None:
                att = att.masked_fill(attn_mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B_q, T_q, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.ln1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, bias=bias, dropout=dropout)
        self.ln2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias=bias, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CrossAttnBlock(nn.Module):
    """Transformer block with cross-attention: self-attn then cross-attn then MLP."""

    def __init__(self, n_embd: int, n_head: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.ln1 = LayerNorm(n_embd, bias=bias)
        self.self_attn = CausalSelfAttention(n_embd, n_head, bias=bias, dropout=dropout)
        self.ln2 = LayerNorm(n_embd, bias=bias)
        self.cross_attn = CrossAttention(n_embd, n_head, bias=bias, dropout=dropout)
        self.ln3 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias=bias, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x), attn_mask=self_attn_mask)
        x = x + self.cross_attn(self.ln2(x), kv=kv, attn_mask=cross_attn_mask)
        x = x + self.mlp(self.ln3(x))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder used for Context and Target encoders."""

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 3,
        n_embd: int = 768,
        n_head: int = 12,
        n_layer: int = 12,
        bias: bool = False,
        dropout: float = 0.0,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_embd = n_embd
        self.use_cls_token = use_cls_token

        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = in_chans * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, n_embd, bias=bias)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        else:
            self.cls_token = None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + (1 if use_cls_token else 0), n_embd))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, bias=bias, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = LayerNorm(n_embd, bias=bias)

        self._init_weights()

    def _init_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, LayerNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_fn)
        # Re-apply the special inits set in __init__ (apply() only touches submodules)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode patches.

        Args:
            x: (B, N, P) patch tokens where N = num_patches, P = patch_size**2 * in_chans.
            mask: (B, N) bool – True for patches to keep (used to zero out masked patches).
                  If None, all patches are used.
            attn_mask: optional attention mask for the transformer blocks.

        Returns:
            (B, M, C) embeddings where M = num_patches (+1 for cls if used).
        """
        B, N, P = x.shape
        x = self.patch_embed(x)  # (B, N, C)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        return x


class Predictor(nn.Module):
    """Predictor network that predicts target embeddings from context embeddings.

    Uses cross-attention: learnable query tokens attend to context embeddings.
    """

    def __init__(
        self,
        n_embd: int = 384,
        n_head: int = 6,
        n_layer: int = 4,
        bias: bool = False,
        dropout: float = 0.0,
        num_queries: int = 16,
        out_dim: int | None = None,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.n_embd = n_embd
        self.out_dim = out_dim if out_dim is not None else n_embd
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_queries, n_embd))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            CrossAttnBlock(n_embd, n_head, bias=bias, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = LayerNorm(n_embd, bias=bias)
        if self.out_dim != n_embd:
            self.out_proj = nn.Linear(n_embd, self.out_dim, bias=False)
        else:
            self.out_proj = nn.Identity()
        self._init_weights()

    def _init_weights(self):
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, LayerNorm):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_fn)
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, context_embeds: torch.Tensor) -> torch.Tensor:
        """Predict target embeddings from context.

        Args:
            context_embeds: (B, N_ctx, C) context encoder output.

        Returns:
            (B, num_queries, out_dim) predicted target embeddings.
        """
        B = context_embeds.size(0)
        queries = self.query_tokens.expand(B, -1, -1) + self.pos_embed
        x = self.drop(queries)

        for block in self.blocks:
            x = block(x, kv=context_embeds)

        x = self.ln_f(x)
        x = self.out_proj(x)
        return x


@dataclass
class JEPAConfig:
    img_size: int = 512
    patch_size: int = 16
    in_chans: int = 3
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    predictor_n_embd: int = 384
    predictor_n_head: int = 6
    predictor_n_layer: int = 4
    predictor_num_queries: int = 16
    bias: bool = False
    dropout: float = 0.0
    use_cls_token: bool = True
    # Masking
    num_target_blocks: int = 4
    min_target_block_size: int = 16
    max_target_block_size: int = 64
    context_scale: float = 2.0
    # Loss
    use_vicreg: bool = False
    vicreg_lambda: float = 1.0
    vicreg_mu: float = 1.0
    vicreg_nu: float = 0.1


class JEPA(nn.Module):
    """Joint-Embedding Predictive Architecture for galaxy images."""

    def __init__(self, config: JEPAConfig, master_process: bool = True):
        super().__init__()
        self.config = config
        self.master_process = master_process

        # --- Encoders ---
        self.context_encoder = ViTEncoder(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_layer=config.n_layer,
            bias=config.bias,
            dropout=config.dropout,
            use_cls_token=config.use_cls_token,
        )
        # Target encoder is a deep copy, EMA-updated during training
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor: one query token per target block (I-JEPA style)
        self.num_target_blocks = config.num_target_blocks
        self.predictor = Predictor(
            n_embd=config.predictor_n_embd,
            n_head=config.predictor_n_head,
            n_layer=config.predictor_n_layer,
            bias=config.bias,
            dropout=config.dropout,
            num_queries=config.num_target_blocks,
            out_dim=config.n_embd,
        )
        # Project context encoder output to predictor dimension if they differ
        if config.n_embd != config.predictor_n_embd:
            self.ctx_proj = nn.Linear(config.n_embd, config.predictor_n_embd, bias=False)
        else:
            self.ctx_proj = nn.Identity()

        if self.master_process:
            total_params = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"JEPA model initialized:")
            print(f"  Total params: {total_params / 1e6:.2f}M")
            print(f"  Trainable:    {trainable / 1e6:.2f}M")
            print(f"  ContextEncoder layers: {config.n_layer}")
            print(f"  Predictor layers:      {config.predictor_n_layer}")

    @torch.no_grad()
    def update_target_encoder(self, m: float = 0.996):
        """EMA update of target encoder from context encoder.

        target = m * target + (1 - m) * context
        """
        for param_t, param_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            param_t.data.mul_(m).add_(param_c.data, alpha=1 - m)

    def forward(
        self,
        patches: torch.Tensor,
        masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Implements I-JEPA style block-wise prediction: each of the
        ``num_target_blocks`` masked blocks is predicted separately by its own
        predictor query token, and the loss is averaged over blocks. This
        prevents the predictor from collapsing to a constant (which would happen
        if a single global-mean target were predicted).

        Args:
            patches: (B, N, P) patch tokens.
            masks:   dict with:
                      'context'      (B, N) bool – True where patch is context
                      'target_blocks' (B, K, N) bool – per-block target masks
                      (optionally 'target' (B, N) union, used for logging only)

        Returns:
            dict with 'loss', 'pred', 'target', optionally 'vicreg_loss'.
        """
        if masks is None:
            raise ValueError("masks must be provided for JEPA forward pass")

        B, N, P = patches.shape
        device = patches.device

        context_mask = masks["context"]          # (B, N)
        target_blocks = masks["target_blocks"]   # (B, K, N)
        K = target_blocks.size(1)

        # --- Context encoder: zero out non-context patches, run ViT ---
        ctx_embeds = self.context_encoder(patches, mask=context_mask)
        # (B, N+1, C) if cls_token, else (B, N, C)
        ctx_embeds = self.ctx_proj(ctx_embeds)

        # --- Target encoder: ALL patches (no mask), frozen ---
        with torch.no_grad():
            tgt_embeds = self.target_encoder(patches, mask=None)
        # (B, N+1, C) if cls_token, else (B, N, C)

        # --- Predictor: one prediction per target block (query token k -> block k) ---
        pred_embeds = self.predictor(ctx_embeds)  # (B, K, C)
        assert pred_embeds.size(1) == K, (
            f"predictor produced {pred_embeds.size(1)} queries but {K} target blocks"
        )

        # --- Gather per-block target embeddings from the target encoder ---
        if self.config.use_cls_token:
            tgt_no_cls = tgt_embeds[:, 1:, :]  # (B, N, C)
        else:
            tgt_no_cls = tgt_embeds

        mask_f = target_blocks.unsqueeze(-1).to(tgt_no_cls.dtype)  # (B, K, N, 1)
        weighted = (mask_f * tgt_no_cls.unsqueeze(1)).sum(dim=2)   # (B, K, C)
        counts = target_blocks.sum(dim=2, keepdim=True).clamp(min=1).to(tgt_no_cls.dtype)  # (B, K, 1)
        target_block_repr = weighted / counts                     # (B, K, C)

        # --- Loss: mean over blocks of per-block MSE ---
        per_block = ((pred_embeds - target_block_repr) ** 2).mean(dim=2)  # (B, K)
        valid = target_blocks.sum(dim=2) > 0                              # (B, K)
        loss = (per_block * valid).sum() / valid.sum().clamp(min=1)

        # --- Optional VICReg regularizer on the predictor outputs ---
        vicreg_loss = None
        if self.config.use_vicreg:
            pred_flat = pred_embeds.reshape(-1, pred_embeds.size(-1))  # (B*K, C)
            pred_centered = pred_flat - pred_flat.mean(dim=0, keepdim=True)
            std = pred_flat.std(dim=0, unbiased=False)
            std_loss = F.relu(1 - std).mean()
            cov = (pred_centered.T @ pred_centered) / max(pred_flat.size(0) - 1, 1)
            cov_loss = cov.fill_diagonal_(0).pow(2).sum() / pred_flat.size(1)
            vicreg_loss = (
                self.config.vicreg_lambda * loss
                + self.config.vicreg_mu * std_loss
                + self.config.vicreg_nu * cov_loss
            )

        return {
            "loss": vicreg_loss if vicreg_loss is not None else loss,
            "pred": pred_embeds.mean(dim=1),       # (B, C) for logging
            "target": target_block_repr.mean(dim=1),  # (B, C) for logging
            "vicreg_loss": vicreg_loss,
        }

    @torch.no_grad()
    def get_embeddings(self, patches: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Extract embeddings from the target encoder for downstream tasks.

        Args:
            patches: (B, N, P) patch tokens.
            reduction: 'mean', 'cls', or 'none'.

        Returns:
            (B, C) embeddings if reduction is 'mean'/'cls', else (B, N, C).
        """
        self.target_encoder.eval()
        embeds = self.target_encoder(patches, mask=None)
        if self.config.use_cls_token:
            if reduction == "cls":
                return embeds[:, 0]
            elif reduction == "none":
                return embeds[:, 1:]
            else:
                return embeds[:, 1:].mean(dim=1)
        else:
            if reduction == "none":
                return embeds
            return embeds.mean(dim=1)

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ):
        """AdamW optimizer with weight decay on 2D params only."""
        import inspect
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=fused)
