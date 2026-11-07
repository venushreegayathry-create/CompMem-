"""Phase 1, Steps B + C: small Transformer encoder -> 64-d vector -> binary code in {-1,+1}
using Binary Concrete relaxation + Straight-Through Estimator."""
import torch
import torch.nn as nn

from chunker import token_ids, MAX_TOKENS


class HashEncoder(nn.Module):
    def __init__(self, vocab_size=8192, d_model=128, n_heads=4, n_layers=2,
                 n_bits=64, max_len=MAX_TOKENS, tau=1.0):
        super().__init__()
        self.vocab_size, self.n_bits, self.tau = vocab_size, n_bits, tau
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.proj = nn.Linear(d_model, n_bits)   # z_i in R^64

    # ---- Step B: text chunk -> z in R^64 ----
    def forward_z(self, ids):                      # ids: (B, L) int, 0 = pad
        pad = ids.eq(0)
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        x = self.tok(ids) + self.pos(pos)
        x = self.enc(x, src_key_padding_mask=pad)
        mask = (~pad).unsqueeze(-1).float()
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1)   # masked mean pool
        return self.proj(pooled)

    # ---- Step C: Binary Concrete + STE ----
    def binarize(self, z):
        if self.training:
            u = torch.rand_like(z).clamp(1e-6, 1 - 1e-6)
            g = torch.log(u) - torch.log(1 - u)     # Logistic noise
            h_tilde = torch.tanh((z + g) / self.tau)
        else:
            h_tilde = torch.tanh(z / self.tau)      # deterministic at inference
        sign = torch.where(h_tilde >= 0, 1.0, -1.0).to(h_tilde.dtype)  # sign() w/o 0
        # parentheses matter: (h_tilde - h_tilde.detach()) is exactly 0 in the forward pass,
        # so h is exactly +-1; without them float rounding gives 0.9999999
        h = sign + (h_tilde - h_tilde.detach())     # forward = +-1, backward = d tanh
        return h, h_tilde

    def forward(self, ids):
        z = self.forward_z(ids)
        h, h_tilde = self.binarize(z)
        return h, h_tilde

    # ---- helpers ----
    def batch_ids(self, texts, device="cpu"):
        seqs = [token_ids(t, self.vocab_size) or [1] for t in texts]
        L = max(len(s) for s in seqs)
        out = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s)
        return out.to(device)

    @torch.no_grad()
    def encode_texts(self, texts):
        """texts -> int64 python ints (64-bit packed codes) + the +-1 tensor."""
        was_training = self.training
        self.eval()
        h, _ = self(self.batch_ids(texts))
        self.train(was_training)
        return h


def to_int(h_row) -> int:
    """+-1 vector (64,) -> 64-bit python int (bit=1 where +1)."""
    v = 0
    for i, x in enumerate(h_row.tolist()):
        if x > 0:
            v |= (1 << i)
    return v
