r"""
agglib.py -- race_agg (research\PREREG_race_agg.md): paths, data, unit files, candidate scores, decision SD, statistics.
Imports the frozen part-1 code (..\..\race_std\research\rslib.py, learners.py) read-only; nothing there is changed.
Only the D folds (2022, 2023, 2024) are used; every series is cut to 2022-01-01 .. 2024-12-31.
"""
import os, sys, json, math, time, hashlib
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AGG = os.path.dirname(HERE)                                   # ...\race ver\race_agg
RV = os.path.dirname(AGG)                                     # ...\race ver
STD = os.path.join(RV, 'race_std')
if os.path.join(STD, 'research') not in sys.path:
    sys.path.insert(0, os.path.join(STD, 'research'))
import rslib as L                                             # noqa: E402

CACHE = os.path.join(AGG, 'cache')
UNITS = os.path.join(CACHE, 'units_agg')
RES = os.path.join(AGG, 'results')
PREREG = os.path.join(HERE, 'PREREG_race_agg.md')
PREREG_V = os.path.join(HERE, 'PREREG_race_agg_V.md')          # the V stage (user request 11/18 18:12) needs this file
D_FOLDS = L.D_FOLDS
D_A, D_B = L.PER['D']
YEARS = [('2022', '2022-01-01', '2023-01-01'), ('2023', '2023-01-01', '2024-01-01'), ('2024', '2024-01-01', '2025-01-01')]
FOLDS_OF = {'D': L.D_FOLDS, 'V': L.V_FOLDS}
PER_OF = {'D': L.PER['D'], 'V': L.PER['V']}
SEGMENTS = {'D': YEARS, 'V': [('2025', '2025-01-01', '2026-01-01'), ('2026-01..05', '2026-01-01', '2026-06-01'),
                              ('2026-06..', '2026-06-01', '2100-01-01')]}
ES_DAYS = 120

FAMILY = {'A1': 'A', 'A2': 'A', 'A3': 'A', 'A4': 'A', 'B1': 'B', 'B2': 'B', 'C1': 'C', 'C2': 'C', 'C3': 'C',
          'D1': 'D', 'D2': 'D', 'E1': 'E', 'E2': 'E', 'E4': 'E', 'F1': 'F', 'F2': 'F'}
CANDS = list(FAMILY)
LABEL = {'A1': 'uniform blend C.LU + C.LN', 'A2': 'uniform blend of the 5 C learners', 'A3': 'linear blending (alpha >= 0)',
         'A4': 'any blending (stacking GBDT)', 'B1': 'LU, 10 seeds', 'B2': 'day-bootstrap bagging of LU, 10 bags',
         'C1': 'AdaBoost-Stump', 'C2': 'AdaBoost-DTree (depth 3)', 'C3': 'GBDT, exponential loss',
         'D1': 'single C&RT, cost-complexity pruned', 'D2': 'LU, categorical dow / month', 'E1': 'random forest',
         'E2': 'uniform blend C.LU + RF', 'E4': 'LU on permutation-selected features', 'F1': 'GBDT, squared loss',
         'F2': 'LU score x absres_60', 'SD': 'race version (C.LU, Top25 + ex-div)', 'COMBO': 'family winners combined'}
# trained units of this experiment (seeds); B1 seeds 1-3 are the cached C.LU units
SEEDS = {'B1': tuple(range(4, 11)), 'B2': tuple(range(1, 11)), 'C1': (0,), 'C2': (0,), 'C3': (1, 2, 3), 'D1': (1,),
         'D2': (1, 2, 3), 'E1': (1,), 'E4': (1, 2, 3), 'F1': (1, 2, 3)}
MEMBERS5 = ['C.LU', 'C.LN', 'C.LP', 'C.LC', 'C.LM']
GM_UNITS = ['GM.' + m.split('.')[1] for m in MEMBERS5]       # g^- of the 5 learners on the last 120 days (A3, A4)


def log(*a):
    s = time.strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, 'race_agg.log'), 'a', encoding='utf-8') as fh:
        fh.write(s + '\n')


def sha256(path):
    return L.sha256(path)


def code_hashes():
    return {f: sha256(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith('.py')}


# ------------------------------------------------------------------ data
def load_panel():
    P = pd.read_pickle(os.path.join(STD, 'cache', 'panel.pkl'))
    P['date'] = pd.to_datetime(P.date).astype('datetime64[ns]')
    P['stock_id'] = P.stock_id.astype(str)
    return P


def feats_C():
    return L.feature_list('C')


def masks(P, fa, fb):
    """training rows (labelled, 2020-03-02 <= date < fold start) and the fold's rows, as part-1 train.run_one"""
    if pd.Timestamp(fb) > pd.Timestamp(D_B):
        assert os.path.exists(PREREG_V), 'the V folds need research\\PREREG_race_agg_V.md'
    d = P['date'].to_numpy()
    lab = P['target_r'].notna().to_numpy() & P['resid_r'].notna().to_numpy()
    tr = lab & (d >= np.datetime64(L.TRAIN_START)) & (d < np.datetime64(pd.Timestamp(fa)))
    te = (d >= np.datetime64(pd.Timestamp(fa))) & (d < np.datetime64(pd.Timestamp(fb)))
    assert d[tr].max() < np.datetime64(pd.Timestamp(fa)), 'training rows reach the fold'
    return tr, te


def split_last(d, tr, k=ES_DAYS):
    """fit part and the last k training days (= learners._split_last_days)"""
    days = np.unique(d[tr])
    cut = days[-k] if len(days) > k else days[len(days) // 2]
    return tr & (d < cut), tr & (d >= cut)


def label_weight(P, tr):
    e = P['resid_r'].to_numpy(float)
    y = (e > 0).astype(float)
    w = np.minimum(np.abs(e), 0.05)
    w = w / np.nanmean(w[tr])
    return e, y, w


# ------------------------------------------------------------------ unit files
def unit_file(cand, fa, seed):
    return os.path.join(UNITS, cand, f'{fa}_seed{seed}.npz')


def cut_D(s):
    dts = s.index.get_level_values('date')
    return s[(dts >= pd.Timestamp(D_A)) & (dts < pd.Timestamp(D_B))]


def cut(s, period):
    a, b = PER_OF[period]
    dts = s.index.get_level_values('date')
    return s[(dts >= pd.Timestamp(a)) & (dts < pd.Timestamp(b))]


MEMBER_V_TRAINED = ('C.LN', 'C.LP', 'C.LM')                    # not finalists in part 1: their V units are trained here


def load_member(unit, period):
    """a C-feature learner (seed mean) on a period: part-1 units where they exist, otherwise this experiment's V units"""
    if period == 'D' or unit not in MEMBER_V_TRAINED:
        s = L.load_unit(unit, FOLDS_OF[period])
    else:
        s = load_agg_unit(unit, seeds=L.SEEDS[unit.split('.')[1]], folds=FOLDS_OF[period])
    if s is None:
        raise FileNotFoundError(f'{unit} {period}')
    return cut(s, period)


def load_period_unit(cand, period, seeds=None):
    s = load_agg_unit(cand, seeds=seeds, folds=FOLDS_OF[period])
    return None if s is None else cut(s, period)


def load_std_unit(unit):
    """a cached part-1 unit (seed mean), D folds only"""
    s = L.load_unit(unit, D_FOLDS)
    if s is None:
        raise FileNotFoundError(unit)
    return cut_D(s)


def load_agg_unit(cand, seeds=None, folds=None):
    """seed mean of this experiment's units; None if a file is missing"""
    seeds = SEEDS[cand] if seeds is None else seeds
    parts = []
    for fa, _ in (folds or D_FOLDS):
        ss = []
        for sd in seeds:
            f = unit_file(cand, fa, sd)
            if not os.path.exists(f):
                return None
            ss.append(L.load_npz(f)[0])
        parts.append(pd.concat(ss, axis=1).mean(axis=1))
    s = pd.concat(parts).sort_index()
    return cut_D(s) if folds is None else s


def load_metas(cand, seeds=None):
    out = []
    for fa, _ in D_FOLDS:
        for sd in (SEEDS[cand] if seeds is None else seeds):
            f = unit_file(cand, fa, sd)
            if os.path.exists(f):
                m = L.load_npz(f)[1]
                m.update(fold=fa, seed=sd)
                out.append(m)
    return out


def zday(s):
    return L.zday(s)


def zavg(series):
    return pd.concat([zday(s) for s in series], axis=1).mean(axis=1)


# ------------------------------------------------------------------ decision and statistics
def load_INFO():
    return L.info_frame(L.load_info())


def decide(score, INFO, period='D'):
    a, b = PER_OF[period]
    X = INFO[(INFO.index.get_level_values('date') >= pd.Timestamp(a)) & (INFO.index.get_level_values('date') < pd.Timestamp(b))]
    D = L.decide_sd(score, X)
    return D[(D.index >= pd.Timestamp(a)) & (D.index < pd.Timestamp(b))]


def stats_p(D, base, period):
    """stats() with the segments of the period (D: 3 years; V: 2025, 2026-01..05, 2026-06..)"""
    d = (D.AR - base.AR).dropna()
    seg = {y: float(d[(d.index >= a) & (d.index < b)].mean()) for y, a, b in SEGMENTS[period]}
    t = float(L.nw_t(d.values))
    roll = D.AR.rolling(19).mean().dropna()
    dic = (D.ic - base.ic).dropna()
    return dict(mean=float(D.AR.mean()), diff=float(d.mean()), t=t, se=float(L.nw_se(d.values)), segments=seg,
                pos_segments=int(sum(v > 0 for v in seg.values())), sd=float(D.AR.std()), p19_neg=float((roll < 0).mean()),
                ic=float(D.ic.mean()), ic_diff=float(dic.mean()), ic_t=float(L.nw_t(dic.values)), days=int(len(D)))


def stats(D, base):
    d = (D.AR - base.AR).dropna()
    yrs = {y: float(d[(d.index >= a) & (d.index < b)].mean()) for y, a, b in YEARS}
    t = float(L.nw_t(d.values))
    roll = D.AR.rolling(19).mean().dropna()
    return dict(mean=float(D.AR.mean()), diff=float(d.mean()), t=t, se=float(L.nw_se(d.values)), years=yrs,
                pos_years=int(sum(v > 0 for v in yrs.values())), sd=float(D.AR.std()), p19_neg=float((roll < 0).mean()),
                ic=float(D.ic.mean()), ones=float(D.ones.mean()), days=int(len(D)),
                passes=bool(d.mean() > 0 and t >= 2.0 and sum(v > 0 for v in yrs.values()) >= 2))


def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out = {}; run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); out[k] = run
    return out


def p_one_sided(t):
    return 1 - 0.5 * (1 + math.erf(t / math.sqrt(2))) if t == t else float('nan')


def daily_rank_corr(a, b):
    """mean over days of the Spearman correlation between two scores (all stocks with both values)"""
    X = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
    r = X.groupby(level='date').rank()
    return float(r.groupby(level='date').apply(lambda g: g['a'].corr(g['b'])).mean())


def top25_changes(score, base_score, INFO):
    """average number of Top25 names (before the ex-dividend rule) that differ from the base, per day"""
    X = INFO[[]].join(score.rename('s'), how='left').join(base_score.rename('b'), how='left')
    X = X[(X.index.get_level_values('date') >= pd.Timestamp(D_A)) & (X.index.get_level_values('date') < pd.Timestamp(D_B))]
    out = []
    for d, g in X.groupby(level='date', sort=True):
        sid = g.index.get_level_values('stock_id').to_numpy()
        def top(v):
            v = np.where(np.isnan(v), -np.inf, v)
            return set(sid[np.lexsort((sid, -v))[:25]])
        out.append(len(top(g.s.to_numpy(float)) - top(g.b.to_numpy(float))))
    return float(np.mean(out))
