"""AuraXR production models.

Active runtime model:
  SDFLSTMModel
    frame_feat: 25 core features + 4 local SDF features = 29 dims
    obj_embed: 32-dim object SDF embedding
    state:     LSTM (h, c), 2 layers, hidden size 256
    output:    MANO PCA pose (15), wrist rotation 6D (6), contact probability (1)

SDFEncoder is used offline to build object embeddings. GraspFlowModel is kept as
an optional contact-pose extension, but the deployed hand pose path is LSTM only.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDFTransformerModel(nn.Module):
    """Causal Transformer: trains all T steps in parallel, infers with rolling context cache.

    Training:  forward_sequence processes the full window in one batched pass → ~3x faster
               than LSTM scheduled-sampling (no sequential loop).
    Inference: forward appends each frame to a context buffer (KV cache analogue) and runs
               the transformer over the accumulated context, taking the last output.
               Equivalent to LSTM stateful inference but bounded by max_seq_len.
    """

    def __init__(
        self,
        feat_dim: int = 29,
        embed_dim: int = 32,
        proj_dim: int = 64,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        orientation_aware_sdf: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.orientation_aware_sdf = orientation_aware_sdf

        sdf_extra = 3 if orientation_aware_sdf else 0
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        self.obj_inj = nn.Sequential(
            nn.Linear(proj_dim + embed_dim + sdf_extra, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)

        self.pose_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 15),
        )
        self.wrist_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )
        self.contact_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _project_seq(self, feat: torch.Tensor, obj_embed: torch.Tensor) -> torch.Tensor:
        """Project (B, T, F) + (B, E) → (B, T, d_model) without positional embedding."""
        B, T, _ = feat.shape
        frame = self.feat_proj(feat)
        emb_exp = obj_embed.unsqueeze(1).expand(-1, T, -1)
        parts = [frame, emb_exp]
        if self.orientation_aware_sdf:
            parts.append(F.normalize(feat[..., 3:6], dim=-1, eps=1e-6))
        return self.obj_inj(torch.cat(parts, dim=-1))

    def _project_frame(self, feat: torch.Tensor, obj_embed: torch.Tensor) -> torch.Tensor:
        """Project (B, F) + (B, E) → (B, d_model) without positional embedding."""
        frame = self.feat_proj(feat)
        parts = [frame, obj_embed]
        if self.orientation_aware_sdf:
            parts.append(F.normalize(feat[..., 3:6], dim=-1, eps=1e-6))
        return self.obj_inj(torch.cat(parts, dim=-1))

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward_sequence(
        self,
        feat_seq: torch.Tensor,
        obj_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parallel training path — all T steps computed in one pass.

        Args:
            feat_seq:  (B, T, feat_dim)
            obj_embed: (B, embed_dim)
        Returns:
            pose (B,T,15), wrist (B,T,6), contact (B,T,1)
        """
        B, T, _ = feat_seq.shape
        x = self._project_seq(feat_seq, obj_embed)
        positions = torch.arange(T, device=feat_seq.device)
        x = x + self.pos_embed(positions)
        mask = self._causal_mask(T, feat_seq.device)
        h = self.transformer(x, mask=mask)
        return (
            self.pose_head(h),
            self.wrist_head(h),
            torch.sigmoid(self.contact_head(h)),
        )

    def forward(
        self,
        frame_feat: torch.Tensor,
        obj_embed: torch.Tensor,
        h_0: torch.Tensor,
        c_0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-frame stateful inference.

        h_0: (B, T_cache, d_model) — past context tokens (with positional embeddings baked in).
        c_0: dummy, returned unchanged for API compatibility with LSTM.
        """
        device = frame_feat.device
        t_cache = h_0.shape[1]

        new_token = self._project_frame(frame_feat, obj_embed)
        pos_idx = torch.tensor([min(t_cache, self.max_seq_len - 1)], device=device)
        new_token_emb = new_token + self.pos_embed(pos_idx)

        if t_cache == 0:
            context = new_token_emb.unsqueeze(1)
        else:
            context = torch.cat([h_0, new_token_emb.unsqueeze(1)], dim=1)

        T_total = context.shape[1]
        mask = self._causal_mask(T_total, device)
        out = self.transformer(context, mask=mask)

        step = out[:, -1, :]

        new_h = context
        if new_h.shape[1] > self.max_seq_len:
            new_h = new_h[:, -self.max_seq_len:, :]

        return (
            self.pose_head(step),
            self.wrist_head(step),
            torch.sigmoid(self.contact_head(step)),
            new_h,
            c_0,
        )

    def initial_state(self, batch_size: int = 1, device=None):
        d = device or next(self.parameters()).device
        h = torch.zeros(batch_size, 0, self.d_model, device=d)
        c = torch.zeros(2, batch_size, self.d_model, device=d)
        return h, c

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SDFLSTMModel(nn.Module):
    """Stateful SDF-conditioned LSTM used by training, evaluation, and Unity."""

    def __init__(
        self,
        feat_dim: int = 29,
        embed_dim: int = 32,
        proj_dim: int = 64,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.25,
        orientation_aware_sdf: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.orientation_aware_sdf = orientation_aware_sdf

        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        self.obj_inj = nn.Sequential(
            nn.Linear(proj_dim + embed_dim + (3 if orientation_aware_sdf else 0), proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 15),
        )
        self.wrist_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )
        self.contact_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _prepare_step(self, frame_feat: torch.Tensor, obj_embed: torch.Tensor) -> torch.Tensor:
        frame = self.feat_proj(frame_feat)
        parts = [frame, obj_embed]
        if self.orientation_aware_sdf:
            parts.append(torch.nn.functional.normalize(frame_feat[..., 3:6], dim=-1, eps=1e-6))
        combined = torch.cat(parts, dim=-1)
        return self.obj_inj(combined)

    def forward_sequence(
        self,
        feat_seq: torch.Tensor,
        obj_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training path for batches of temporal windows.

        Args:
            feat_seq:  (B, T, 29)
            obj_embed: (B, 32)

        Returns:
            pose_pca:     (B, T, 15)
            wrist_rot:    (B, T, 6)
            contact_prob: (B, T, 1)
        """
        _, T, _ = feat_seq.shape
        frame = self.feat_proj(feat_seq)
        emb = obj_embed.unsqueeze(1).expand(-1, T, -1)
        parts = [frame, emb]
        if self.orientation_aware_sdf:
            parts.append(torch.nn.functional.normalize(feat_seq[..., 3:6], dim=-1, eps=1e-6))
        lstm_in = self.obj_inj(torch.cat(parts, dim=-1))
        h_out, _ = self.lstm(lstm_in)
        return (
            self.pose_head(h_out),
            self.wrist_head(h_out),
            torch.sigmoid(self.contact_head(h_out)),
        )

    def forward(
        self,
        frame_feat: torch.Tensor,
        obj_embed: torch.Tensor,
        h_0: torch.Tensor,
        c_0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-frame stateful path exported to ONNX."""
        lstm_in = self._prepare_step(frame_feat, obj_embed).unsqueeze(1)
        h_out, (h_n, c_n) = self.lstm(lstm_in, (h_0, c_0))
        step = h_out.squeeze(1)
        return (
            self.pose_head(step),
            self.wrist_head(step),
            torch.sigmoid(self.contact_head(step)),
            h_n,
            c_n,
        )

    def initial_state(self, batch_size: int = 1, device=None):
        d = device or next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=d)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=d)
        return h, c

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SDFEncoder(nn.Module):
    """Offline SDF grid encoder used to compute 32-dim object embeddings."""

    def __init__(self, grid_res: int = 32, embed_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.Linear(64, embed_dim), nn.LayerNorm(embed_dim))

    def forward(self, sdf_grid: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(sdf_grid))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GraspFlowModel(nn.Module):
    """Optional contact pose denoiser conditioned on object embedding and LSTM state."""

    def __init__(
        self,
        pose_dim: int = 15,
        embed_dim: int = 32,
        hidden_dim: int = 256,
        net_hidden: int = 512,
        n_layers: int = 4,
        dropout: float = 0.10,
    ):
        super().__init__()
        in_dim = pose_dim + 1 + embed_dim + hidden_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, net_hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(net_hidden, net_hidden), nn.SiLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(net_hidden, pose_dim))
        self.net = nn.Sequential(*layers)
        self.pose_dim = pose_dim

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        sdf_embed: torch.Tensor,
        lstm_hidden: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([x_t, t, sdf_embed, lstm_hidden], dim=-1))

    @torch.no_grad()
    def sample(self, sdf_embed: torch.Tensor, lstm_hidden: torch.Tensor, n_steps: int = 5):
        x = torch.randn(sdf_embed.shape[0], self.pose_dim, device=sdf_embed.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((sdf_embed.shape[0], 1), i * dt, device=sdf_embed.device)
            x = x + dt * self.forward(x, t, sdf_embed, lstm_hidden)
        return x

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    batch, steps = 4, 16
    frame = torch.randn(batch, 29)
    seq   = torch.randn(batch, steps, 29)
    emb   = torch.randn(batch, 32)

    for ModelClass in [SDFLSTMModel, SDFTransformerModel]:
        m = ModelClass()
        h0, c0 = m.initial_state(batch)
        pose, wrist, contact, hn, cn = m(frame, emb, h0, c0)
        assert pose.shape    == (batch, 15),  pose.shape
        assert wrist.shape   == (batch, 6),   wrist.shape
        assert contact.shape == (batch, 1),   contact.shape

        pose_s, wrist_s, cont_s = m.forward_sequence(seq, emb)
        assert pose_s.shape  == (batch, steps, 15)
        assert wrist_s.shape == (batch, steps, 6)
        assert cont_s.shape  == (batch, steps, 1)
        print(f"{ModelClass.__name__} OK ({m.count_params():,} params)")
