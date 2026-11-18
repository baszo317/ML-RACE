r"""
v11b_lib.py -- v11 part B (research\PREREG_v11b.md): v9's walk-forward predictions with the candidate changes to the market view
(T1 pbar over the Top50, T2 night-session reversal, T3 month start) and to the stock view (X1 second LightGBM on the xrN9
features, X2 LightGBM with a return-size label, X3 z-score combination, X4 smoothing across days), scored with v9's rule.
The decision follows v10lib.decide (= v9_eval.decide): ex-dividend day / day before / 2 days after -> 0, beta-proportional
hedge lambda 0.75; T3 shifts the scores before the hedge so that 5 more Top50 stocks (outside the ex-dividend rule) get 1.
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); V11 = os.path.dirname(HERE); ROOT = os.path.dirname(V11)
for p in (os.path.join(ROOT, 'v10', 'research'), os.path.join(ROOT, 'v9', 'research'), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import v10lib as V
import v11b_train as TB

W_T, LAM, NIGHT_W, SMOOTH_A, EXTRA_ONES = 0.5, 0.75, 0.5, 0.5, 5

class Data:
    """everything the candidates need, computed once"""
    def __init__(self):
        self.W9, self.TF, self.LGB, self.idx = V.load_v9()
        self.P = V.panel()
        self.INFO = V.info_table(self.P, self.idx)
        self.top50 = self.P.set_index(['date', 'stock_id']).top50.reindex(self.idx).astype(bool)
        self.tr = self.P.set_index(['date', 'stock_id']).target_r.reindex(self.idx)
        self.raw = pd.concat([pd.concat([self.W9.mlp_raw_B9, self.W9.mlp_raw_xr9, self.W9.mlp_raw_w4_9], axis=1).mean(axis=1), self.TF],
                             axis=1).mean(axis=1)
        self.mlp = [self.W9.mlp_res_B9, self.W9.mlp_res_xr9, self.W9.mlp_res_BN9, self.W9.mlp_res_xrN9]
        self.x1 = TB.load_member('lgb_res_xrN'); self.x2 = TB.load_member('lgb_res_soft')
        if self.x1 is not None:
            self.x1 = self.x1.reindex(self.idx)
        if self.x2 is not None:
            self.x2 = self.x2.reindex(self.idx)
        import v9lib
        N = v9lib.night_table().set_index('date').night_ret.sort_index()
        sd = N.rolling(250, min_periods=60).std().shift(1)            # past days only
        self.z_night = (N / sd).clip(-3, 3)
        days = pd.DatetimeIndex(sorted(self.P.date.unique()))
        first3 = pd.Series(days, index=days).groupby(days.to_period('M')).transform(lambda s: s.rank(method='first') <= 3)
        self.month_start = set(days[first3.values.astype(bool)])
        self.dates = pd.DatetimeIndex(sorted(set(self.idx.get_level_values('date'))))
        v9 = self.stock_view()
        self.v9_sd = v9.groupby(level='date').std()                   # v9's daily scale of the centred stock view

    # ------------------------------------------------------------ views
    def pbar(self, top50=False, night=False):
        r = self.raw[self.top50] if top50 else self.raw
        pb = r.groupby(level='date').mean()
        if night:
            sd = pb.rolling(250, min_periods=20).std().shift(1)
            z = self.z_night.reindex(pb.index)
            adj = (NIGHT_W * sd * z).fillna(0.0)
            pb = pb - adj
        return pb

    def stock_view(self, x1=False, x2=False, zscore=False, smooth=False):
        """centred p_e (Series on idx); X3 / X4 are rescaled to v9's daily sd"""
        c = lambda s: s - s.groupby(level='date').transform('mean')
        sdd = lambda s: s.groupby(level='date').transform('std')
        lg = [self.LGB]
        if x1:
            lg.append(self.x1)
        if x2:
            s2 = c(self.x2)
            lg.append(self.LGB.groupby(level='date').transform('mean') + s2 * sdd(c(self.LGB)) / sdd(s2))
        if zscore:
            z = lambda s: c(s) / sdd(s)
            mz = pd.concat([z(s) for s in self.mlp], axis=1).mean(axis=1)
            lz = pd.concat([z(s) for s in lg], axis=1).mean(axis=1)
            pe = c((mz + lz) / 2)
        else:
            pe = c((pd.concat(self.mlp, axis=1).mean(axis=1) + pd.concat(lg, axis=1).mean(axis=1)) / 2)
        if zscore or smooth:
            if smooth:
                U = pe.unstack('stock_id').sort_index()
                out = U.copy()
                prev = None
                for i, d in enumerate(U.index):
                    row = U.loc[d]
                    if prev is not None:
                        row = (SMOOTH_A * row + (1 - SMOOTH_A) * prev.where(prev.notna(), row)).where(row.notna())
                    out.loc[d] = row; prev = row
                pe = out.stack().reindex(self.idx)
                pe = c(pe)
            pe = pe / sdd(pe) * self.v9_sd.reindex(pe.index.get_level_values('date')).values
        return pe

    def score(self, cfg):
        pb = self.pbar(top50='T1' in cfg, night='T2' in cfg)
        pe = self.stock_view(x1='X1' in cfg, x2='X2' in cfg, zscore='X3' in cfg, smooth='X4' in cfg)
        s = W_T * (pb.reindex(pe.index.get_level_values('date')).values - 0.5) + pe
        return s, pb, pe

    # ------------------------------------------------------------ decision (v10lib.decide + T3)
    def decide(self, score, month_start=False, lam=LAM, ex_cols=V.EX4):
        X = self.INFO.join(score.rename('score'), how='inner')
        rows = []
        for d, g in X.groupby(level='date'):
            sc, b, r = g.score.values.copy(), g.beta.values, g.target_r.values
            ex = np.zeros(len(g), bool)
            for col in ex_cols:
                ex |= g[col].values == 1
            if month_start and d in self.month_start:
                cand = np.sort(sc[~ex & (sc <= 0)])[::-1]
                if len(cand) > EXTRA_ONES:
                    sc = sc - (cand[EXTRA_ONES - 1] + cand[EXTRA_ONES]) / 2
                elif len(cand):
                    sc = sc - cand[-1] + 1e-9
            s = np.where(ex, -1, np.where(sc > 0, 1, -1))
            if lam < 1.0:
                sb, bb = s.mean(), b.mean(); B0 = ((s - sb) * (b - bb)).mean(); Et = sb * bb + lam * B0
                S = (sc[None, :] + V.CGRID[:, None] * b[None, :] > 0) & ~ex[None, :]
                E = ((2 * S - 1) @ b) / len(b)
                dd = np.abs(E - Et); cnd = np.flatnonzero(dd <= dd.min() + 1e-12); j = cnd[np.argmin(np.abs(V.CGRID[cnd]))]
                s = np.where(S[j], 1, -1)
            m = r.mean()
            rows.append((d, (s * r).mean() * 1e4, s.mean() * m * 1e4, m * 1e4, s.mean()))
        D = pd.DataFrame(rows, columns=['date', 'AR', 'timing', 'm', 'sbar']).set_index('date'); D['xsec'] = D.AR - D.timing
        return D

    def run(self, cfg):
        s, pb, pe = self.score(cfg)
        D = self.decide(s, month_start='T3' in cfg)
        return D, pb, pe

    def ic(self, pe, top50=False):
        """daily rank IC of the centred stock view with the intraday return (all stocks or the Top50)"""
        X = pd.DataFrame({'p': pe, 'r': self.tr}).dropna()
        if top50:
            X = X[self.top50.reindex(X.index).values]
        g = X.groupby(level='date')
        R = pd.DataFrame({'p': g.p.rank(), 'r': g.r.rank()})
        g = R.groupby(level='date')
        return g.apply(lambda x: np.corrcoef(x.p, x.r)[0, 1])
