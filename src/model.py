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

import torch
import torch.nn as nn


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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        self.obj_inj = nn.Sequential(
            nn.Linear(proj_dim + embed_dim, proj_dim),
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
        combined = torch.cat([frame, obj_embed], dim=-1)
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
        lstm_in = self.obj_inj(torch.cat([frame, emb], dim=-1))
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
    model = SDFLSTMModel()
    frame = torch.randn(batch, 29)
    seq = torch.randn(batch, steps, 29)
    emb = torch.randn(batch, 32)
    h0, c0 = model.initial_state(batch)

    pose, wrist, contact, hn, cn = model(frame, emb, h0, c0)
    assert pose.shape == (batch, 15)
    assert wrist.shape == (batch, 6)
    assert contact.shape == (batch, 1)
    assert hn.shape == (2, batch, 256)
    assert cn.shape == (2, batch, 256)

    pose_seq, wrist_seq, contact_seq = model.forward_sequence(seq, emb)
    assert pose_seq.shape == (batch, steps, 15)
    assert wrist_seq.shape == (batch, steps, 6)
    assert contact_seq.shape == (batch, steps, 1)
    print(f"SDFLSTMModel OK ({model.count_params():,} params)")
