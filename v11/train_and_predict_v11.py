r"""
train_and_predict_v11.py -- v11 daily (live) prediction (README_v11.md part B, research\PREREG_v11b.md)

v11 = v9 + the three changes selected by the user's rule (val bp > 0) that passed the pre-registered check (hold, post > 0):
  T2 night-session reversal (market view): pbar' = pbar - 0.5 * sd(pbar of the previous 250 days) * clip(z_night, -3, 3),
     z_night = night_ret / sd(night_ret of the previous 250 sessions). The night session closes at 05:00, so the evening
     (prelim) run has no night session and T2 adds nothing; the morning run adds it.
  T3 month start: on the first 3 trading days of a month all scores are shifted up so that 5 more Top50 stocks (outside the
     ex-dividend rules) get 1; done before the beta hedge.
  X1 a second LightGBM in the stock view: lgb_res_xrN (LightGBM as lgb_res, with the features of mlp_res_xrN9); the LightGBM part
     of p_e is the mean of lgb_res and lgb_res_xrN (rolling9.combine averages the members of each kind).
Everything else is v9 (w_T 0.5, ex-dividend day / day before / 2 days after -> 0, beta hedge 0.75) and v9's members and
model pool are used as they are: v11 reads ..\v9\models_v9 (trained by v9's own run) and only trains lgb_res_xrN into
models_v11\ (1 model per day, about 40 s on the laptop; the pool was started in the cloud with 5 models trained up to 10/22).

Run v9 first (daily_v11.bat / evening_v11.bat do both):
  python train_and_predict_v11.py --dates auto            morning, after ..\v9\train_and_predict_v9.py --dates auto
  python train_and_predict_v11.py --dates auto --prelim   evening; 2026-10-28: when the zip has no usable day to predict (no -1
                                                          row, or a -1 row without the U.S. close) it only trains lgb_res_xrN
                                                          and exits normally, like v9 (..\v9\train_and_predict_v9.py --prelim)
  --as-v9    the same program without T2, T3, X1: must reproduce v9's file exactly (check)
Your own parameters (README_v11.md section 8; custom_v11.bat asks for them):
  --w-t 0.5        market-view weight w_T (0 = market view off)
  --lam 0.75       beta hedge lambda (1 = no hedge, smaller = hedge more)
  --night-w 0.5    T2 weight of the night-session reversal (0 = T2 off)
  --extra-ones 5   T3: how many more 1s on the first 3 trading days of a month (0 = T3 off)
  --no-x1          without the second LightGBM (X1)
  --balanced       F2 (README_v11 section 10): every day half of the Top50 get 1 by the stock view alone (no market view, T3 or
                   hedge). Tested 2026-10-27: NOT lower risk - daily sd 45 vs 38 for w_T 0 / lambda 0.5, because it is short
                   the market (correlation with the day's market move -0.37); it only looks good on big down days such as 10/27.
  A run with any non-default parameter writes submission_v11_MMDD_<parameters>.csv next to the normal file (it never replaces
  submission_v11_MMDD.csv) and does not touch history_predictions.csv or data\pbar_history.csv. --tag sets the name yourself.
Outputs (v11\out\): submission_v11_MMDD.csv (upload if you choose v11), probabilities_v11_MMDD.csv (with pbar_v9, night_z,
  night_adj, month_start), checks_v11_MMDD.txt, history_predictions.csv; final runs add the day's pbar to data\pbar_history.csv.
"""
import os, sys, time, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); V9DIR = os.path.join(ROOT, 'v9')
if V9DIR not in sys.path:
    sys.path.insert(0, V9DIR)
import rolling9 as V
import train_and_predict_v9 as T9
E, R, G = V.E, V.R, T9.G
log = T9.log

X1 = V.mem('lgb_res_xrN', 'lgb', 'B9+XR+NEW+XRN+SIG', 'resid', train='lgb')
ROOT9 = os.path.join(V9DIR, 'models_v9')
ROOT11 = os.path.join(HERE, 'models_v11')
PBAR_HIST = os.path.join(HERE, 'data', 'pbar_history.csv')
NIGHT_W, EXTRA_ONES = 0.5, 5

# ------------------------------------------------------------------ T2
def night_z(core, d):
    """night_ret of day d divided by the sd of the previous 250 night sessions (>= 60), clipped to +-3; nan if missing"""
    x = core.loc[core.date == d, 'night_ret'].dropna()
    if x.empty:
        return np.nan
    N = V.night_table().dropna(subset=['night_ret'])
    past = N.loc[N.date < d, 'night_ret'].tail(250)
    if len(past) < 60 or not past.std() > 0:
        return np.nan
    return float(np.clip(x.iloc[0] / past.std(), -3, 3))

def pbar_sd(d, path=PBAR_HIST):
    """sd of the daily pbar of the previous 250 days (>= 20) from data\\pbar_history.csv; nan if not enough"""
    if not os.path.exists(path):
        return np.nan
    H = pd.read_csv(path, parse_dates=['date'])
    past = H[H.date < d].sort_values('date').pbar.dropna().tail(250)
    return float(past.std()) if len(past) >= 20 else np.nan

def save_pbar(ens, path=PBAR_HIST):
    new = ens.groupby(level='date').pbar_v9.first().rename('pbar').reset_index()
    new['date'] = pd.to_datetime(new.date).dt.strftime('%Y-%m-%d'); new['source'] = 'live (v9 members, final run)'
    H = pd.read_csv(path, dtype={'date': str}) if os.path.exists(path) else pd.DataFrame(columns=['date', 'pbar', 'source'])
    H = pd.concat([H[~H.date.isin(new.date)], new], ignore_index=True).sort_values('date')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    H.to_csv(path + '.tmp', index=False, float_format='%.6f'); os.replace(path + '.tmp', path)

# ------------------------------------------------------------------ T3
def trading_days(data_dir):
    closed = set(V.closed_days(data_dir).strftime('%Y-%m-%d'))
    try:
        cal = pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(os.path.join(data_dir, 'ref', 'calendar.parquet')).date)).sort_values()
    except Exception:
        return None
    return pd.DatetimeIndex([d for d in cal if d.strftime('%Y-%m-%d') not in closed])

def month_start_days(data_dir, dates):
    """dates that are among the first 3 trading days of their month (official calendar without the closed days; Mon-Fri if the
    calendar is unreadable)"""
    cal = trading_days(data_dir); out = set()
    for d in dates:
        d = pd.Timestamp(d)
        if cal is not None:
            same = cal[(cal.year == d.year) & (cal.month == d.month) & (cal <= d)]
        else:
            same = pd.bdate_range(d.replace(day=1), d)
        if len(same) and same[-1] == d and len(same) <= 3:
            out.add(d)
    return out

def shift_month_start(ens, core, stocks, days, extra=EXTRA_ONES):
    """shift every score of a month-start day so that `extra` more Top50 stocks outside the ex-dividend rules get 1"""
    flags = core.set_index(['date', 'stock_id'])[list(V.EX4)]
    ens = ens.copy(); ens['month_start'] = 0; ens['ms_shift'] = 0.0
    for d in days:
        sl = ens.index.get_level_values('date') == d
        g = ens[sl]
        top = g[g.index.get_level_values('stock_id').isin(stocks)]
        ex = (flags.reindex(top.index).fillna(0.0) == 1).any(axis=1).values
        sc = top.score.values
        cand = np.sort(sc[~ex & (sc <= 0)])[::-1]
        if extra <= 0:
            th = 0.0
        elif len(cand) > extra:
            th = (cand[extra - 1] + cand[extra]) / 2
        elif len(cand):
            th = cand[-1] - 1e-9
        else:
            th = 0.0
        ens.loc[sl, 'score'] = ens.loc[sl, 'score'] - th
        ens.loc[sl, 'month_start'] = 1; ens.loc[sl, 'ms_shift'] = -th
    return ens

# ------------------------------------------------------------------ F2
def balance_half(X):
    """F2 (research\\PREREG_v11f.md): per day, the Top50 outside the ex-dividend rules ranked by the stock view p_e_c; the
    floor(n/2) highest get 1, every other stock 0 (ex-dividend-rule stocks count among the 0s) -> net exposure 0"""
    X = X.copy()
    for d, g in X.groupby('date'):
        k = len(g) // 2
        ok = g[~g.ex_any.astype(bool)]
        top = ok.p_e_c.sort_values(ascending=False, kind='stable').index[:k]
        X.loc[g.index, 'pred'] = 0
        X.loc[top, 'pred'] = 1
    return X

# ------------------------------------------------------------------ main
def compare_with_v9(sub, sub_dates, stocks, tag):
    msgs = []
    f = os.path.join(V9DIR, 'out', f'submission_v9_{tag}.csv')
    if not os.path.exists(f):
        return f'no ..\\v9\\out\\{os.path.basename(f)} to compare with'
    S9 = pd.read_csv(f, dtype={'id': str}).set_index('id'); S = sub.set_index('id')
    for d in sub_dates:
        col = pd.Timestamp(d).strftime('%Y-%m-%d')
        if col in S9:
            n = int((S9[col].reindex(stocks).values != S[col].reindex(stocks).values).sum())
            msgs.append(f'{col}: {n} of {len(stocks)} predictions differ from v9 ({os.path.basename(f)}); share of 1 v9 '
                        f'{S9[col].reindex(stocks).mean():.2f}, v11 {S[col].reindex(stocks).mean():.2f}')
    return '; '.join(msgs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=V.DATA)
    ap.add_argument('--template', default=os.path.join(E.ROOT, 'sample_submission.csv'))
    ap.add_argument('--dates', default='auto', help="'auto' = the -1 placeholder day(s); or comma-separated dates")
    ap.add_argument('--retrain', default='auto', choices=['auto', 'none', 'now'], help='lgb_res_xrN only (v9 trains its own members)')
    ap.add_argument('--models-v9', default=ROOT9)
    ap.add_argument('--models-dir', default=ROOT11)
    ap.add_argument('--panel-cache', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--prelim', action='store_true', help='evening run (no night session yet): tag MMDD_prelim')
    ap.add_argument('--as-v9', action='store_true', help='check: without T2, T3, X1 (must equal v9)')
    ap.add_argument('--w-t', type=float, default=V.W_T, help='market-view weight (v11 = v9 = 0.5); a risk setting, see README_v11 6')
    ap.add_argument('--lam', type=float, default=V.LAM, help='beta hedge (v11 = v9 = 0.75; 1 = off)')
    ap.add_argument('--night-w', type=float, default=NIGHT_W, help='T2 night-session reversal weight (v11 0.5; 0 = off)')
    ap.add_argument('--extra-ones', type=int, default=EXTRA_ONES, help='T3 extra 1s on the first 3 trading days of a month (v11 5; 0 = off)')
    ap.add_argument('--no-x1', action='store_true', help='without the second LightGBM lgb_res_xrN')
    ap.add_argument('--balanced', action='store_true', help='F2: half of the Top50 get 1 by the stock view (not lower risk, README_v11 10)')
    a = ap.parse_args(); t0 = time.time()
    if not (0 <= a.w_t <= 3 and 0 <= a.lam <= 1 and 0 <= a.night_w <= 3 and 0 <= a.extra_ones <= 50):
        sys.exit('parameters out of range: 0 <= w-t <= 3, 0 <= lam <= 1, 0 <= night-w <= 3, 0 <= extra-ones <= 50')
    use_x1 = not (a.as_v9 or a.no_x1)
    custom = [p for p, on in ((f'wt{a.w_t:g}', a.w_t != V.W_T), (f'lam{a.lam:g}', a.lam != V.LAM), (f'nw{a.night_w:g}', a.night_w != NIGHT_W),
                              (f'ex{a.extra_ones}', a.extra_ones != EXTRA_ONES), ('nox1', a.no_x1), ('bal', a.balanced)) if on]
    params = (f'w_T {a.w_t:g}, lambda {a.lam:g}, ' + ('as v9 (no T2, T3, X1)' if a.as_v9 else
              f'T2 night weight {a.night_w:g}, T3 extra 1s {a.extra_ones}, X1 {"on" if use_x1 else "off"}'))
    if a.balanced:
        params = f'F2 balanced: half of the Top50 get 1 by the stock view (w_T, lambda, T2, T3 not used), X1 {"on" if use_x1 else "off"}'
    sched = V.SCHED['live']; fs = V.feature_sets()
    members9 = [dict(m) for m in V.MEMBERS]; x1 = dict(X1)
    members = members9 + ([x1] if use_x1 else [])
    out_dir = a.out or os.path.join(HERE, 'out'); os.makedirs(out_dir, exist_ok=True)
    tpl = pd.read_csv(a.template, dtype={'id': str}); stocks = tpl['id'].tolist()
    if a.dates == 'auto':
        dates = E.placeholder_dates(a.data)
        why = T9.evening_no_prediction(a, dates)
        if why:                                                                  # 2026-10-28: evening without a usable day
            log(t0, f'{why}: no preliminary file tonight; training lgb_res_xrN (X1) instead')
            core, ext = T9.load_panels(a, [], t0)
            lab = R.labelled_days(core)
            if use_x1 and a.retrain != 'none':
                V.live_update(core, ext, [x1], a.models_dir, sched, fs, retrain=a.retrain, log=lambda s: log(t0, s))
            log(t0, f'done: lgb_res_xrN trained on data up to {lab[-1].date()}. Next morning, after about 07:00, download the newest zip '
                    f'and run daily_v11.bat: it finds the models and only predicts')
            return
        if not dates:
            sys.exit("no -1 placeholder day in the data: run ..\\transformer\\update_data.py with today's zip first (or ..\\v9\\daily.bat)")
    else:
        dates = [pd.Timestamp(x) for x in a.dates.split(',')]
    base_tag = dates[-1].strftime('%m%d') + ('_prelim' if a.prelim else '')
    tag = a.tag or (base_tag + ('_' + '_'.join(custom) if custom else ''))
    official = not a.tag and not custom and not a.as_v9                        # the normal file: history and pbar history
    core, ext = T9.load_panels(a, dates, t0)
    lab = R.labelled_days(core)
    have = set(core.date.unique())
    model_dates = [d for d in dates if d in have and R.prev_day(lab, d) is not None]
    if not model_dates:
        sys.exit('none of the requested dates can be predicted (no labelled day before them)')
    log(t0, f'labelled data up to {lab[-1].date()}; parameters: {params}' + ('' if official else f' -> files tagged {tag}'))

    rep = G.Report()
    core, nxt, ids = V.fix_ex_next(core, a.data)
    log(t0, f'next trading day {nxt.date()}: ex-dividend announced for {ids if ids else "none of the stocks"}')
    allf = sorted(set(sum((fs[m['feats']] for m in members), [])) - set(V.NIGHT))
    rep.add('INFO', 'environment: ' + T9.env_line())
    rep.add('INFO', 'parameters: ' + params + ('' if not custom else ' (non-default: ' + ', '.join(custom) + ')'))
    T9.check_not_past(model_dates, rep)
    G.check_inputs(core, model_dates, allf, a.data, rep)
    T9.check_exdiv_recent(a.data, rep)
    for d in model_dates:
        nn = core.loc[core.date == d, V.NIGHT].isna().all(axis=1).mean()
        if nn > 0.5:
            rep.add('INFO' if a.prelim else 'WARN', f'{d.date()}: no TX night session ' + ('yet (evening run): T2 adds nothing tonight'
                                                    if a.prelim else '(run ..\\v7\\fetch\\fetch_night_session.py): T2 adds nothing today'))
        else:
            rep.add('OK', f'{d.date()}: TX night session present')
    if rep.fatal() and not a.force:
        rep.show(lambda m: log(t0, m))
        sys.exit('stopped by the input checks above (nothing written). Fix the cause, or rerun with --force after checking.')

    prevd = R.prev_day(lab, model_dates[-1])
    if use_x1 and a.retrain != 'none':
        V.live_update(core, ext, [x1], a.models_dir, sched, fs, retrain=a.retrain, log=lambda s: log(t0, s))
    G.check_pool(lambda m: V.scan_pool(a.models_v9, m), members9, prevd, rep)
    for m in members9:
        pts = V.scan_pool(a.models_v9, m)
        if pts and pd.Timestamp(max(pts)) < pd.Timestamp(prevd):
            rep.add('FATAL', f"{m['name']}: v9's newest model is trained up to {pd.Timestamp(max(pts)).date()}, before the previous trading "
                             f"day {pd.Timestamp(prevd).date()}: run ..\\v9\\daily.bat (or evening.bat) first - v11 uses v9's models")
    if use_x1:
        G.check_pool(lambda m: V.scan_pool(a.models_dir, m), [x1], prevd, rep)
    if rep.fatal() and not a.force:
        rep.show(lambda m: log(t0, m))
        sys.exit('stopped by the model-pool checks above (nothing written).')
    probs, desc = V.live_probs(core, ext, members9, a.models_v9, sched, model_dates, fs=fs, log=lambda s: log(t0, s))
    if use_x1:
        p1, d1 = V.live_probs(core, ext, [x1], a.models_dir, sched, model_dates, fs=fs, log=lambda s: log(t0, s))
        probs.update(p1); desc.update(d1)
    for (name, d), txt in desc.items():
        log(t0, f'{name} for {d.date()}: {txt}')

    ens = V.combine(probs, members, w_t=a.w_t)
    ens = ens[ens.index.get_level_values('date').isin(model_dates)].copy()
    ens['pbar_v9'] = ens.pbar; ens['night_z'] = np.nan; ens['night_adj'] = 0.0
    if not a.as_v9:
        for d in model_dates:                                                    # T2
            z, sd = night_z(core, d), pbar_sd(d)
            adj = a.night_w * sd * z if np.isfinite(z) and np.isfinite(sd) else 0.0
            sl = ens.index.get_level_values('date') == d
            ens.loc[sl, 'night_z'] = z; ens.loc[sl, 'night_adj'] = adj; ens.loc[sl, 'pbar'] = ens.loc[sl, 'pbar_v9'] - adj
            log(t0, f'{d.date()}: T2 night z {z:+.2f}, sd(pbar) {sd:.4f}, pbar {ens.loc[sl, "pbar_v9"].iloc[0]:.4f} -> {ens.loc[sl, "pbar"].iloc[0]:.4f}')
        ens['score'] = a.w_t * (ens.pbar - 0.5) + ens.p_e_c
        ms = month_start_days(a.data, model_dates)                               # T3
        ens = shift_month_start(ens, core, stocks, ms if a.extra_ones > 0 else set(), extra=a.extra_ones)
        for d in sorted(ms):
            log(t0, f'{d.date()}: T3 month start, scores shifted by {ens.loc[ens.index.get_level_values("date") == d, "ms_shift"].iloc[0]:+.4f}')
    else:
        ens['month_start'] = 0; ens['ms_shift'] = 0.0
    X, H = V.decide(ens, core, stocks, lam=a.lam)
    if a.balanced:
        X = balance_half(X); H = H.iloc[0:0]
        rep.add('INFO', 'F2 balanced: half of the Top50 get 1 by the stock view; not lower risk than w_T 0 / lambda 0.5 (README_v11 section 10)')
    miss = X[X.score.isna()]
    if len(miss):
        rep.add('FATAL', f'{len(miss)} Top50 stock-days without a score, e.g. {miss[["date", "stock_id"]].head(3).values.tolist()}')
    G.check_output(X[X.date.isin(model_dates)], set(stocks), rep)

    fb_rows = core[core.target_r.notna()].sort_values('date')
    fb = (fb_rows.groupby('stock_id').target_r.apply(lambda s: s.tail(250).mean()) > 0).astype(int).reindex(stocks).fillna(0).astype(int)
    rep.show(lambda m: log(t0, m))
    name = 'v11asv9' if a.as_v9 else 'v11'
    with open(os.path.join(out_dir, f'checks_{name}_{tag}.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(f'[{l}] {m}' for l, m in rep.items) + '\n')
    if rep.fatal() and not a.force:
        sys.exit(f'stopped by {len(rep.fatal())} FATAL check(s) above: nothing written.')
    sub_dates = [pd.Timestamp(d) for d in (E.placeholder_dates(a.data) if a.dates == 'auto' else model_dates)]
    sub = pd.DataFrame({'id': stocks})
    for d in sub_dates:
        x = X[X.date == d].set_index('stock_id')['pred'].reindex(stocks)
        sub[d.strftime('%Y-%m-%d')] = np.where(x.isna(), fb.values, x.fillna(0).values).astype(int)
    vals = sub.drop(columns='id')
    assert list(sub['id']) == stocks and vals.notna().all().all() and set(np.unique(vals.values)) <= {0, 1}, 'submission format check failed'
    f = os.path.join(out_dir, f'submission_{name}_{tag}.csv')
    sub.to_csv(f, index=False)
    log(t0, f'wrote {f}: {sub.shape[0]} stocks x {sub.shape[1] - 1} date(s), share of 1 = {vals.values.mean():.3f}')
    msg = compare_with_v9(sub, sub_dates, stocks, base_tag)
    log(t0, msg)
    with open(os.path.join(out_dir, f'checks_{name}_{tag}.txt'), 'a', encoding='utf-8') as fh:
        fh.write(f'[INFO] {msg}\n')
    cols = ['date', 'stock_id', 'p_raw', 'pbar_v9', 'night_z', 'night_adj', 'pbar', 'p_e', 'p_e_mlp', 'p_e_lgb', 'p_e_c', 'month_start',
            'ms_shift', 'score'] + list(V.EX4) + ['beta', 'pred_nohedge', 'pred']
    X[[c for c in cols if c in X]].to_csv(os.path.join(out_dir, f'probabilities_{name}_{tag}.csv'), index=False, float_format='%.6f')
    if a.as_v9:
        return
    if len(H):
        H.to_csv(os.path.join(out_dir, f'hedge_v11_{tag}.csv'), index=False, float_format='%.5f')
    if official:
        hist = os.path.join(out_dir, 'history_predictions.csv')
        h = X[[c for c in cols if c in X]].assign(w_t=a.w_t, lam=a.lam, run='prelim' if a.prelim else 'final',
                                               made_at=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
        if os.path.exists(hist):
            old = pd.read_csv(hist, dtype={'stock_id': str}, parse_dates=['date'])
            h = pd.concat([old[~old.date.isin(h.date.unique())], h], ignore_index=True)
        h.sort_values(['date', 'stock_id']).to_csv(hist, index=False, float_format='%.6f')
        if not a.prelim:
            save_pbar(ens)
    for d in model_dates[-3:]:
        x = X[X.date == d]
        log(t0, f"{d.date()}: pbar v9 {x.pbar_v9.iloc[0]:.3f} -> v11 {x.pbar.iloc[0]:.3f}; share of 1 among the 50: before hedge "
                f"{x.pred_nohedge.mean():.2f}, final {x.pred.mean():.2f}; month start {int(x.month_start.iloc[0])}")

if __name__ == '__main__':
    main()
