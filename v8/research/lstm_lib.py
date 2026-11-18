"""lstm_lib: the LSTM member of the seventh round (PREREG_m.md), plugged into transformer/tfmodel.py without editing it.
Importing this module makes tfmodel.make_model accept kind='lstm' (and keeps earlier patches, e.g. rnn_lib for kind='rnn')."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'transformer'))
import torch, torch.nn as nn
import tfmodel as TM

class LSTMClassifier(nn.Module):
    """x: [batch, L, n_feat] -> 2-layer LSTM over the window -> [last hidden state, window mean] -> logit"""
    def __init__(self, n_feat, seq_len, hidden=32, layers=2, dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 1)
    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(self.drop(torch.cat([h[:, -1], h.mean(1)], dim=1))).squeeze(-1)

_prev = TM.make_model
def make_model(kind, n_feat, seq_len, **kw):
    if kind == 'lstm':
        return LSTMClassifier(n_feat, seq_len, dropout=kw.get('dropout', 0.2))
    return _prev(kind, n_feat, seq_len, **kw)
TM.make_model = make_model
