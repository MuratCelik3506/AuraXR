"""Baseline causal LSTM: Controller-to-Hand (Faz 2A).

Girdi/adim: [feat(13), category_emb(E), prev_pose(15)] -> LSTM -> finger_pca15.
Egitim: teacher-forced (prev = GT[t-1]). Eval: free-running (prev = pred[t-1]).
Bilek tahmin EDILMEZ (controller'dan gelir).
"""
import math
import torch
import torch.nn as nn

FEAT_DIM = 13
POSE_DIM = 45   # tam finger axis-angle (eski: 15 PCA)
NUM_CAT = 4     # hook | power | wide | pinch


# --- Faz 6 alternatif gövdeler (causal, feedforward) -------------------------
class _CausalConvBlock(nn.Module):
    """sol-pad dilated conv1d + ReLU + residual (causal)."""
    def __init__(self, c_in, c_out, k=3, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)
        self.relu = nn.ReLU()
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):  # (B,C,L)
        h = self.conv(nn.functional.pad(x, (self.pad, 0)))
        h = self.relu(h)
        return h + (self.res(x) if self.res is not None else x)


class CausalTCN(nn.Module):
    def __init__(self, in_dim, hidden, n_blocks=4, k=3):
        super().__init__()
        blocks, ch, d = [], in_dim, 1
        for _ in range(n_blocks):
            blocks.append(_CausalConvBlock(ch, hidden, k, d)); ch = hidden; d *= 2
        self.net = nn.ModuleList(blocks)

    def forward(self, x):  # (B,L,C)->(B,L,H)
        h = x.transpose(1, 2)
        for b in self.net:
            h = b(h)
        return h.transpose(1, 2)


class CausalTransformer(nn.Module):
    def __init__(self, in_dim, hidden, n_layers=4, heads=4, maxlen=512):
        super().__init__()
        self.hidden = hidden
        self.inp = nn.Linear(in_dim, hidden)
        self.register_buffer("pe", self._sinusoid(maxlen, hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=hidden * 2,
                                           batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    @staticmethod
    def _sinusoid(L, hidden, device=None):
        pe = torch.zeros(L, hidden, device=device)
        pos = torch.arange(L, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, hidden, 2, device=device).float() * (-math.log(10000.0) / hidden))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, x):  # (B,L,C)->(B,L,H)
        L = x.size(1)
        pos = self.pe[:L] if L <= self.pe.size(0) else self._sinusoid(L, self.hidden, x.device)
        h = self.inp(x) + pos
        mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
        return self.enc(h, mask=mask)


class ControllerToHand(nn.Module):
    def __init__(self, hidden=256, layers=1, cat_emb=8, dropout=0.0, use_prev=True,
                 feat_dim=FEAT_DIM, arch="lstm"):
        super().__init__()
        self.arch = arch
        if arch != "lstm":
            use_prev = False   # feedforward seq modeller: AR/prev yok (saf temporal encoder)
        self.use_prev = use_prev
        self.feat_dim = feat_dim
        self.cat_emb = nn.Embedding(NUM_CAT, cat_emb)
        self.in_dim = feat_dim + cat_emb + (POSE_DIM if use_prev else 0)
        if arch == "lstm":
            self.lstm = nn.LSTM(self.in_dim, hidden, num_layers=layers,
                                batch_first=True, dropout=dropout)
        elif arch == "tcn":
            self.tcn = CausalTCN(self.in_dim, hidden, n_blocks=max(4, 2 * layers))
        elif arch == "transformer":
            self.tr = CausalTransformer(self.in_dim, hidden, n_layers=max(4, 2 * layers))
        else:
            raise ValueError(f"bilinmeyen arch: {arch}")
        self.head = nn.Linear(hidden, POSE_DIM)

    def _encode(self, x):  # (B,L,in_dim) -> (B,L,hidden)
        if self.arch == "lstm":
            out, _ = self.lstm(x)
            return out
        if self.arch == "tcn":
            return self.tcn(x)
        return self.tr(x)

    def forward_teacher(self, feat, cat, target):
        """feat(B,L,13) cat(B,L) target(B,L,15) -> pred(B,L,15). prev = GT shift."""
        B, L, _ = feat.shape
        parts = [feat, self.cat_emb(cat)]
        if self.use_prev:
            prev = torch.zeros(B, L, POSE_DIM, device=feat.device)
            prev[:, 1:] = target[:, :-1]
            parts.append(prev)
        return self.head(self._encode(torch.cat(parts, dim=-1)))

    def forward_scheduled(self, feat, cat, target, ss_prob):
        """AR + scheduled sampling: prev = (1-p) GT[t-1] + p kendi tahmini. p artar."""
        if not self.use_prev:   # feedforward gövde (tcn/transformer): AR yok
            return self.forward_teacher(feat, cat, target)
        B, L, _ = feat.shape
        emb = self.cat_emb(cat)
        h = c = None
        prev = torch.zeros(B, POSE_DIM, device=feat.device)
        preds = []
        for t in range(L):
            x = torch.cat([feat[:, t], emb[:, t], prev], dim=-1).unsqueeze(1)
            out, (h, c) = self.lstm(x, None if h is None else (h, c))
            p = self.head(out[:, 0])
            preds.append(p)
            if t + 1 < L:
                use_own = (torch.rand(B, 1, device=feat.device) < ss_prob).float()
                prev = use_own * p.detach() + (1 - use_own) * target[:, t]
        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def forward_free(self, feat, cat):
        """free-running: use_prev ise prev=kendi tahmini; degilse saf feed-forward."""
        if not self.use_prev:
            return self.head(self._encode(torch.cat([feat, self.cat_emb(cat)], dim=-1)))
        B, L, _ = feat.shape
        h = c = None
        prev = torch.zeros(B, POSE_DIM, device=feat.device)
        preds = []
        emb = self.cat_emb(cat)
        for t in range(L):
            x = torch.cat([feat[:, t], emb[:, t], prev], dim=-1).unsqueeze(1)
            out, (h, c) = self.lstm(x, None if h is None else (h, c))
            p = self.head(out[:, 0])
            preds.append(p)
            prev = p
        return torch.stack(preds, dim=1)


def masked_mse(pred, target, mask):
    """pred/target (B,L,D), mask (B,L). -> skaler."""
    d = ((pred - target) ** 2).mean(-1)         # (B,L)
    return (d * mask).sum() / mask.sum().clamp(min=1)


def masked_vel_mse(pred, target, mask):
    """ardisik fark (hiz) MSE, maskeli."""
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    mm = mask[:, 1:] * mask[:, :-1]
    d = ((dp - dt) ** 2).mean(-1)
    return (d * mm).sum() / mm.sum().clamp(min=1)
