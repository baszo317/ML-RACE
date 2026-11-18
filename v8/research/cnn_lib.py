"""cnn_lib: the 1D-CNN member of the fourth round (PREREG_m.md), plugged into transformer/tfmodel.py without editing it.
Importing this module makes tfmodel.make_model accept kind='cnn'."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'transformer'))
import torch, torch.nn as nn
import tfmodel as TM

class CNNClassifier(nn.Module):
    """x: [batch, L, n_feat] -> two temporal convolutions over the window -> [last day, window mean] -> logit"""
    def __init__(self, n_feat, seq_len, ch=32, dropout=0.2):
        super().__init__()
        self.c1 = nn.Conv1d(n_feat, ch, kernel_size=3, padding=1)
        self.c2 = nn.Conv1d(ch, ch, kernel_size=3, padding=2, dilation=2)
        self.act = nn.GELU(); self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * ch, 1)
    def forward(self, x):
        h = x.transpose(1, 2)                                   # [batch, n_feat, L]
        h = self.drop(self.act(self.c1(h)))
        h = self.drop(self.act(self.c2(h)))                     # same length L
        return self.head(torch.cat([h[:, :, -1], h.mean(2)], dim=1)).squeeze(-1)

_orig = TM.make_model
def make_model(kind, n_feat, seq_len, **kw):
    if kind == 'cnn':
        return CNNClassifier(n_feat, seq_len, dropout=kw.get('dropout', 0.2))
    return _orig(kind, n_feat, seq_len, **kw)
TM.make_model = make_model
