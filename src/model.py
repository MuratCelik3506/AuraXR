"""Baseline causal LSTM: Controller-to-Hand (Faz 2A).

Girdi/adim: [feat(13), category_emb(E), prev_pose(15)] -> LSTM -> finger_pca15.
Egitim: teacher-forced (prev = GT[t-1]). Eval: free-running (prev = pred[t-1]).
Bilek tahmin EDILMEZ (controller'dan gelir).
"""
import torch
import torch.nn as nn

FEAT_DIM = 13
POSE_DIM = 15
NUM_CAT = 3


class ControllerToHand(nn.Module):
    def __init__(self, hidden=256, layers=1, cat_emb=8, dropout=0.0, use_prev=True,
                 feat_dim=FEAT_DIM):
        super().__init__()
        self.use_prev = use_prev
        self.feat_dim = feat_dim
        self.cat_emb = nn.Embedding(NUM_CAT, cat_emb)
        self.in_dim = feat_dim + cat_emb + (POSE_DIM if use_prev else 0)
        self.lstm = nn.LSTM(self.in_dim, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, POSE_DIM)

    def forward_teacher(self, feat, cat, target):
        """feat(B,L,13) cat(B,L) target(B,L,15) -> pred(B,L,15). prev = GT shift."""
        B, L, _ = feat.shape
        parts = [feat, self.cat_emb(cat)]
        if self.use_prev:
            prev = torch.zeros(B, L, POSE_DIM, device=feat.device)
            prev[:, 1:] = target[:, :-1]
            parts.append(prev)
        out, _ = self.lstm(torch.cat(parts, dim=-1))
        return self.head(out)

    def forward_scheduled(self, feat, cat, target, ss_prob):
        """AR + scheduled sampling: prev = (1-p) GT[t-1] + p kendi tahmini. p artar."""
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
            out, _ = self.lstm(torch.cat([feat, self.cat_emb(cat)], dim=-1))
            return self.head(out)
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
