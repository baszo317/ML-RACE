r"""
leak_check.py -- 檢查 static 上傳檔有沒有問題：格式、答案段（in-sample、抄到 y）、日期錯位、同一 row（當天資訊）的 leak、
大盤 timing leak。每個檔給 FATAL / WARN / OK。2026-10-26。

用法
  python leak_check.py 檔案.csv [更多.csv ...] [--refs R.csv ...] [--cutoff 2026-08-31 ...] [--out 報告.txt] [--days 每日.csv]
  - 也可以把檔案拖到 leak_check.bat 上；雙擊 leak_check.bat 會檢查 others\ 裡所有的 csv。路徑可以用萬用字元（others\*.csv）。
  - --refs：合規參考檔，預設 v4 static、v9 static w0.25、v9 static（都是 rolling，每天只用前一個交易日以前的資料訓練）。
  - --cutoff：另外要比較前後的日期。預設會加上原始課程資料（data_backup_original）的最後一天（8/31）。
  - --out：報告另存一份文字檔；--days：每個檔每天的準確率、phi、分數存成 csv。
  - 只需要 numpy 和 pandas。label 讀 data\market\*.csv（etl:adj_open、etl:adj_close、y），預測日的 -1 佔位列略過。

結論
  FATAL  確定有問題：格式錯，或分數裡有答案。不能上傳，也不能拿來做組內比較。
  WARN   可疑：某個檢查超出合規檔的範圍。要看程式才能確定是 leak、在 static 上挑設定，還是真的比較好。
  OK     沒看到大問題。不代表合規：分散的小量 leak（見最下面「抓得到多少」）、在 static 上挑設定（in-sample upstream
         selection）、只影響幾天的訓練截止日錯誤，從檔案都看不出來，要看程式。

檢查什麼
 1. 格式：id 和日期 column 的集合、順序和 sample_submission 相同；每格是 "0" 或 "1"（1.0、空白都算錯）；每個日期在資料裡都有答案。
    BOM、換行只列出來（sample 是 UTF-8 BOM + CRLF）。
 2. 分數：AR、timing、xsec（算法同 compare_static.py）、準確率、逐月分數和「每格都猜對」的上限（oracle）比。
 3. 答案段：phi = 當天 50 支的 p 和 y 的相關係數（phi coefficient），1 表示 50 支全部等於答案。比準確率好用：45 支都漲的日子
    全部猜 1，準確率 0.9，但 phi = 0。看單日 phi、連續高 phi 的日子、5 / 10 / 20 天平均、最明顯的切點，以及原始資料最後一天
    前後：用含 6-8 月 label 的原始資料訓練、再預測同一段（in-sample），分數會在那天斷開（phase1_4 就是這樣）。
 4. 日期對齊：p_D 對第 D+k 個交易日的盤中報酬計分（k = -2..+2）。k = +1 或 +2 明顯比 k = 0 高，表示預測往前錯了一格
    （第 D+1 天的預測寫在第 D 天，而它用到第 D 天收盤）。k = -1 只列出來：押昨天反轉或動能的模型本來就和 k = -1 有關。
 5. 當天資訊：第 D row 的開盤跳空、RSI 和 MACD 的變化（含第 D 天收盤）、外資和投信買賣超（收盤後才公布）。
    先把每個變數對開盤前就知道的變數（前一天的值、前一天的報酬、前一晚美股和 VIX 的變化）回歸取殘差 e，
    再看 2p - 1 和 e 的關係：股票、日期兩向去平均，t 按日 cluster。
    為什麼合規檔的 t 在 0 附近：令 F 為第 D 天 9:00 前的資訊，合規的 p_D 由 F 決定。若 e = X - E[X | F]，
      E[(2p - 1) e] = E[(2p - 1) E[e | F]] = E[(2p - 1) * 0] = 0。
    回歸殘差只近似 e（模型真正的預測力沒有被扣掉），所以 RSI、MACD 會有 +2 到 +5 的偏移，門檻是按合規檔訂的。
 6. 大盤 timing：每天猜 1 的比例和當天 Top50 平均盤中報酬、當天 ^TWII 收盤變化的相關。global 檔第 D row 的 ^TWII、^KS11
    是當天收盤，沒有 shift 就會在這裡出現。
 7. 和參考檔決策相同的比例：看是不是複製或小改別人的檔。

門檻怎麼來（2026-10-26 校準，84 天）
  沒有發現 leak 的 8 個檔：v4、v9 static w0.25 / w0.5 / wT0 λ0.5 / max、組員 V3（phase1 (2)）、phase1_1、v12。
  植入假 leak：在 v9 w0.25 和 v4 上，把隨機 x% 的格子換成某個規則，看什麼時候超過門檻（抓得到多少）。
  | 項目                          | 8 個檔                        | WARN            | FATAL          | 植入多少超過 WARN              |
  | AR（bp/日）                   | 15.4-23.9（static 搜尋 32.2） | >= 30           | >= 60          | 抄答案 10%、RSI 10%、MACD 20%  |
  | 單日 phi                      | 最高 0.52-0.69                |                 | >= 0.9         | 抄 1 天答案                    |
  | 連續 3 天以上 phi >= 0.6      | 沒有                          |                 | 有             | 抄 3 天答案                    |
  | 5 / 10 / 20 天平均 phi 最高   | 0.34 / 0.26 / 0.16 以下       | .42 / .33 / .22 | .5 / .4 / .3   | 抄 5 天答案、隨機抄 10%        |
  | 平均 phi（20 天以上）         | 0.055-0.092                   | >= 0.14         | >= 0.25        | 隨機抄 10%                     |
  | 某月 AR / oracle（10 天以上） | 0.16-0.21                     | >= 0.3          | >= 0.5         | 隨機抄 20%、連續抄 10 天       |
  | 當天資訊 t                    | -1.1 到 +4.8                  | < -4 或 > 6     | < -8 或 > 10   | RSI 10%、MACD 20%、gap 20-30%、外資 20% |
  | 大盤 timing 相關              | -0.12 到 +0.15                | >= 0.35 且 t>=3 | >= 0.6 且 t>=4 | 用當天 ^TWII 10-20%            |
  | k = +1、+2 比 k = 0 高        | t -3.6 到 +0.3                | t >= 2.5        | t >= 4         | 整份往前錯一格                 |
  | 最明顯的切點（pooled t）      | |t| 2.0-3.1                   | |t|>=5 且 phi>=.2 | |t|>=6 且 phi>=.3 | 連續抄 5 天、in-sample 30%   |
  | 原始資料最後一天前後（pooled t）| -1.2 到 +0.1                | t>=3 且以前 phi>=.15 | t>=4 且以前 phi>=.3 | in-sample 30%            |
  整段抄答案 3 天以上一定抓得到；在含 label 的資料上訓練、背下 30% 答案的 in-sample 也抓得到。
  分散的小量 leak（上面的比例以下，例如隨機抄 5% 的答案、in-sample 只背下 5%）抓不到。
"""
import os, sys, glob, json, hashlib, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = ['v4/out/submission_v4_static.csv', 'v9/out/submission_v9_static_w025.csv', 'v9/out/submission_v9_static.csv']
VOL = 'price:成交股數'
FLOW_F = 'institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)'
FLOW_T = 'institutional_investors_trading_summary:投信買賣超股數'
# same-day variables: key, label, controls that are known before 09:00 of day D
SAMEDAY = [('gap', 'gap (open)', ['us', 'nas', 'dvix', 'r1', 'cc1', 'gap1']),
           ('drsi', 'dRSI', ['rsi1', 'drsi1', 'us', 'r1', 'cc1']),
           ('dmacd', 'dMACD', ['macd1', 'dmacd1', 'sma1', 'us', 'r1', 'cc1']),
           ('fnet', 'foreign net buy', ['fnet1', 'fnet2', 'us', 'r1', 'cc1']),
           ('tnet', 'trust net buy', ['tnet1', 'tnet2', 'us', 'r1', 'cc1'])]
TH = {'phi_exact': 0.95, 'phi_day': 0.9, 'phi_run': 0.6, 'run_len': 3,
      'roll_warn': {5: 0.42, 10: 0.33, 20: 0.22}, 'roll_fatal': {5: 0.5, 10: 0.4, 20: 0.3},
      'roll_ok': {5: 0.34, 10: 0.26, 20: 0.16},
      'phi_warn': 0.14, 'phi_fatal': 0.25, 'ratio_warn': 0.3, 'ratio_fatal': 0.5, 'ar_warn': 30.0, 'ar_fatal': 60.0,
      't_warn': (-4.0, 6.0), 't_fatal': (-8.0, 10.0), 'tim_warn': (0.35, 3.0), 'tim_fatal': (0.6, 4.0), 'align_t': 2.5, 'align_fatal': 4.0}

try:
    sys.stdout.reconfigure(errors='replace')
except Exception:
    pass


class Out:
    def __init__(self, path=None):
        self.f = open(path, 'w', encoding='utf-8-sig') if path else None

    def __call__(self, s=''):
        print(s)
        if self.f:
            self.f.write(s + '\n')

    def close(self):
        if self.f:
            self.f.close()


# ---------------------------------------------------------------- data
def build_panel(root, ids):
    out = []
    for sid in ids:
        m = pd.read_csv(os.path.join(root, 'data', 'market', f'{sid}.csv'), parse_dates=['date'])
        m = m[(m['etl:adj_open'] > 0) & (m['etl:adj_close'] > 0)].sort_values('date').reset_index(drop=True)
        o, c = m['etl:adj_open'], m['etl:adj_close']
        c1 = c.shift(1); r = (c - o) / o
        d = pd.DataFrame({'id': sid, 'date': m['date'], 'r': r, 'y': m['y'].astype(float), 'yc': (c > o).astype(int)})
        d['gap'] = o / c1 - 1; d['cc'] = c / c1 - 1
        d['r1'] = r.shift(1); d['cc1'] = d['cc'].shift(1); d['gap1'] = d['gap'].shift(1); d['y1'] = d['y'].shift(1)
        d['rsi1'] = m['RSI'].shift(1); d['drsi'] = m['RSI'] - m['RSI'].shift(1); d['drsi1'] = d['drsi'].shift(1)
        d['dmacd'] = (m['MACD'] - m['MACD'].shift(1)) / c1; d['dmacd1'] = d['dmacd'].shift(1)
        d['macd1'] = m['MACD'].shift(1) / c1; d['sma1'] = (c1 - m['SMA'].shift(1)) / c1
        if VOL in m:
            vol = m[VOL].where(m[VOL] > 0)
            for key, col in (('fnet', FLOW_F), ('tnet', FLOW_T)):
                if col in m:
                    d[key] = m[col] / vol; d[key + '1'] = d[key].shift(1); d[key + '2'] = d[key].shift(2)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def load_global(root):
    f = os.path.join(root, 'data', 'global', 'global_data.csv')
    if not os.path.exists(f):
        return None
    g = pd.read_csv(f, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    G = pd.DataFrame({'date': g['date']})
    for col, key in (('^GSPC', 'us'), ('^IXIC', 'nas'), ('^TWII', 'twii')):
        if col in g:
            v = g[col].where(g[col] > 0); G[key] = v / v.shift(1) - 1
    if '^VIX' in g:
        v = g['^VIX'].where(g['^VIX'] > 0); G['dvix'] = v - v.shift(1)
    return G


def last_date(root, sub, sid):
    f = os.path.join(root, sub, 'market', f'{sid}.csv')
    if not os.path.exists(f):
        return None
    m = pd.read_csv(f, usecols=['date', 'etl:adj_open'], parse_dates=['date'])
    m = m[m['etl:adj_open'] > 0]
    return m['date'].max() if len(m) else None


# ---------------------------------------------------------------- reading a submission
def read_sub(path, tpl, data_ids):
    """returns (long frame id/date/p or None, info, format flags)"""
    flags = []
    with open(path, 'rb') as fh:
        raw = fh.read()
    info = {'md5': hashlib.md5(raw).hexdigest(), 'bytes': len(raw), 'bom': raw[:3] == b'\xef\xbb\xbf',
            'crlf': raw.count(b'\r\n'), 'lf': raw.count(b'\n')}
    try:
        s = pd.read_csv(path, dtype=str, encoding='utf-8-sig', keep_default_na=False)
    except Exception as e:
        return None, info, [('FATAL', f'讀不到這個 csv：{e}')]
    s.columns = [str(c).strip() for c in s.columns]
    if 'id' not in s.columns:
        return None, info, [('FATAL', '沒有 id column')]
    s['id'] = s['id'].str.strip()
    cols = [c for c in s.columns if c != 'id']
    info['shape'] = (len(s), len(cols))
    dates, bad_cols = {}, []
    for c in cols:
        try:
            dates[c] = pd.Timestamp(c)
        except Exception:
            bad_cols.append(c)
    if bad_cols:
        flags.append(('FATAL', f'{len(bad_cols)} 個 column 不是日期（例如 {bad_cols[:3]}；重複的 column 會變成 "日期.1"）'))
    if s['id'].duplicated().any():
        flags.append(('FATAL', f'id 重複：{sorted(s.id[s.id.duplicated()].unique())[:5]}'))
    t_ids, t_cols = list(tpl['id']), [c for c in tpl.columns if c != 'id']
    t_dates = [pd.Timestamp(c) for c in t_cols]
    if set(cols) != set(t_cols) and not bad_cols and sorted(dates.values()) == sorted(t_dates) and len(cols) == len(t_cols):
        flags.append(('WARN', f'日期 column 的寫法和 sample_submission 不同（例如 {cols[0]}，sample 是 {t_cols[0]}；常見原因是用 Excel '
                              f'開過再存檔）。日期本身相同，計分不受影響；上傳前改成和 sample 一樣比較保險'))
        cols_cmp = [t_cols[t_dates.index(dates[c])] for c in cols]
    else:
        cols_cmp = cols
    if set(s['id']) != set(t_ids):
        flags.append(('FATAL', f'id 和 sample_submission 不同：多了 {sorted(set(s.id) - set(t_ids))[:5]}，'
                               f'少了 {sorted(set(t_ids) - set(s.id))[:5]}'))
    elif list(s['id']) != t_ids:
        flags.append(('WARN', 'id 集合相同，但順序和 sample_submission 不同'))
    if set(cols_cmp) != set(t_cols):
        flags.append(('FATAL', f'日期 column 和 sample_submission 不同：多了 {sorted(set(cols) - set(t_cols))[:5]}，'
                               f'少了 {sorted(set(t_cols) - set(cols))[:5]}'))
    elif cols_cmp != t_cols:
        flags.append(('WARN', '日期 column 集合相同，但順序和 sample_submission 不同'))
    good = [c for c in cols if c in dates]
    V = s[good]
    ok = V.isin(['0', '1'])
    nbad = int((~ok).values.sum())
    if nbad:
        ex = sorted(set(V.values[~ok.values].ravel().tolist()))[:5]
        flags.append(('FATAL', f'{nbad} 格不是 "0" 或 "1"（例如 {ex}）；數值是 0/1 的仍會計分，其他的略過'))
    unknown = sorted(set(s['id']) - set(data_ids))
    if unknown:
        flags.append(('FATAL', f'data\\market 沒有這些股票：{unknown[:5]}'))
    num = V.apply(pd.to_numeric, errors='coerce')
    W = num.where(num.isin([0, 1]))
    W.insert(0, 'id', s['id'])
    L = W.melt(id_vars='id', var_name='date', value_name='p').dropna(subset=['p'])
    L['date'] = pd.to_datetime(L['date'].map(dates)); L['p'] = L['p'].astype(int)
    L = L[L['id'].isin(data_ids)].drop_duplicates(['id', 'date']).reset_index(drop=True)
    if len(L) == 0:
        return None, info, flags + [('FATAL', '沒有可以計分的格子')]
    return L, info, flags


# ---------------------------------------------------------------- statistics
def phi_day(p, y):
    n11 = (p * y).sum(); n10 = (p * (1 - y)).sum(); n01 = ((1 - p) * y).sum(); n00 = ((1 - p) * (1 - y)).sum()
    den = np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    return float((n11 * n00 - n10 * n01) / den) if den > 0 else 0.0


def nw_se(d, lags=5):
    d = np.asarray(d, float); n = len(d); e = d - d.mean(); s = e @ e / n
    for k in range(1, lags + 1):
        if k < n:
            s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return np.sqrt(max(s, 0.0) / n)


def pooled_t(a, b):
    """two-sample t with a pooled variance (the usual statistic for one change in the mean)"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    diff = a.mean() - b.mean()
    sp2 = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    se = np.sqrt(sp2 * (1 / len(a) + 1 / len(b)))
    if se > 1e-12:
        return diff / se
    return np.inf if diff > 0 else (-np.inf if diff < 0 else 0.0)


def runs(mask):
    out, start, n, prev = [], None, 0, None
    for d, v in mask.items():
        if v:
            if start is None:
                start, n = d, 0
            n += 1; prev = d
        elif start is not None:
            out.append((start, prev, n)); start = None
    if start is not None:
        out.append((start, prev, n))
    return out


def twoway(z, cols):
    for c in cols:
        z[c] = z[c] - z.groupby('id')[c].transform('mean')
        z[c] = z[c] - z.groupby('date')[c].transform('mean')
    return z


def innov_t(x, key, controls):
    """t of (2p - 1) on the part of a same-day variable that is not predictable before 09:00"""
    if key not in x:
        return np.nan
    ctr = [c for c in controls if c in x]
    z = x[['id', 'date', 's', key] + ctr].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(z) < 100:
        return np.nan
    z = twoway(z, ['s', key] + ctr)
    X = z[key].values
    if ctr:
        C = z[ctr].values
        X = X - C @ np.linalg.lstsq(C, X, rcond=None)[0]
        W = np.column_stack([X, C])
    else:
        W = X[:, None]
    s = z['s'].values
    e = s - W @ np.linalg.lstsq(W, s, rcond=None)[0]
    xx = X @ X
    sc = pd.Series(X * e).groupby(z['date'].values).sum().values
    se = np.sqrt((sc ** 2).sum()) / xx
    return float((X @ s) / xx / se) if se > 1e-12 else np.nan


def corr_t(a, b):
    z = pd.concat([a, b], axis=1).dropna(); n = len(z)
    if n < 5 or z.iloc[:, 0].std() == 0 or z.iloc[:, 1].std() == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(z.iloc[:, 0], z.iloc[:, 1])[0, 1])
    return r, r * np.sqrt((n - 2) / max(1 - r * r, 1e-12))


def analyze(L, ctx):
    x = L.merge(ctx['P'], on=['id', 'date'], how='left')
    res = {'missing': int(x['r'].isna().sum() + (x['r'].notna() & x['y'].isna()).sum())}
    x = x.dropna(subset=['r', 'y']).copy()
    x['y'] = x['y'].astype(int)
    res['y_mismatch'] = int((x['y'] != x['yc']).sum())
    x['s'] = 2 * x.p - 1; x['hit'] = (x.p == x.y).astype(int); x['sr'] = x.s * x.r; x['orc'] = (2 * x.y - 1) * x.r
    g = x.groupby('date')
    D = pd.DataFrame({'n': g.size(), 'acc': g['hit'].mean(), 'ybar': g['y'].mean(), 'share1': g['p'].mean(),
                      'AR': g['sr'].mean() * 1e4, 'oracle': g['orc'].mean() * 1e4, 'mkt': g['r'].mean() * 1e4})
    D['phi'] = pd.Series({d: phi_day(v.p.values, v.y.values) for d, v in g})
    D['timing'] = (2 * D.share1 - 1) * D.mkt
    res['D'] = D; res['n_days'] = len(D)
    se_ar = nw_se(D.AR) if len(D) > 2 else 0.0
    res['AR'] = D.AR.mean(); res['t_AR'] = res['AR'] / se_ar if se_ar > 0 else np.nan
    res['timing'] = D.timing.mean(); res['xsec'] = res['AR'] - res['timing']
    res['acc'] = x.hit.mean(); res['share1'] = x.p.mean()
    res['phi'] = D.phi.mean(); res['phi_se'] = D.phi.std() / np.sqrt(len(D)) if len(D) > 1 else np.nan
    M = D.groupby(D.index.to_period('M')).agg(days=('n', 'size'), AR=('AR', 'mean'), oracle=('oracle', 'mean'),
                                              acc=('acc', 'mean'), phi=('phi', 'mean'))
    M['ratio'] = M.AR / M.oracle; res['M'] = M
    # answer-key segments
    res['exact'] = D.index[D.phi >= TH['phi_exact']]
    res['near'] = D.index[(D.phi >= TH['phi_day']) & (D.phi < TH['phi_exact'])]
    res['runs'] = [r for r in runs(D.phi >= TH['phi_run']) if r[2] >= TH['run_len']]
    res['roll'] = {}
    for w in (5, 10, 20):
        if len(D) >= w:
            rm = D.phi.rolling(w).mean()
            res['roll'][w] = (float(rm.max()), rm.idxmax())
    v, best = D.phi.values, None
    for j in range(5, len(v) - 4):
        t = pooled_t(v[:j], v[j:])
        if best is None or abs(t) > abs(best[1]):
            best = (j, t)
    if best is not None:
        j, t = best
        res['split'] = (D.index[j - 1], j, v[:j].mean(), D.acc.values[:j].mean(), len(v) - j, v[j:].mean(),
                        D.acc.values[j:].mean(), t)
    res['cut'] = []
    for c in ctx['cutoffs']:
        b, a = D[D.index <= c], D[D.index > c]
        if len(b) >= 2 and len(a) >= 2:
            res['cut'].append((c, len(b), b.acc.mean(), b.phi.mean(), len(a), a.acc.mean(), a.phi.mean(),
                               pooled_t(b.phi, a.phi)))
    # date alignment: score p_D against the trading day D+k
    cal = ctx['cal']; pos = pd.Series(np.arange(len(cal)), index=pd.DatetimeIndex(cal))
    j0 = L['date'].map(pos)
    A = {}
    for k in (-2, -1, 0, 1, 2):
        j = j0 + k; ok = j.notna() & (j >= 0) & (j < len(cal))
        z = pd.DataFrame({'id': L['id'][ok].values, 'D': L['date'][ok].values,
                          'date': pd.DatetimeIndex(cal)[j[ok].astype(int).values], 'p': L['p'][ok].values})
        z = z.merge(ctx['R'], on=['id', 'date'])
        A[k] = ((2 * z.p - 1) * z.r).groupby(z.D).mean() * 1e4
    res['align'] = {}
    for k, s_k in A.items():
        d = (s_k - A[0]).dropna()
        se = max(nw_se(d), d.std() / np.sqrt(len(d))) if len(d) > 2 else 0.0
        res['align'][k] = (s_k.mean(), d.mean() / se if se > 1e-12 else 0.0)
    # same-day information
    res['same'] = {key: innov_t(x, key, ctr) for key, lab, ctr in SAMEDAY}
    # market timing
    res['tim_mkt'] = corr_t(D.share1, D.mkt)
    G = ctx['G']
    if G is not None and 'twii' in G:
        res['tim_twii'] = corr_t(D.share1, G.set_index('date')['twii'].reindex(D.index))
    else:
        res['tim_twii'] = (np.nan, np.nan)
    res['tim_prev'] = corr_t(D.share1, (x.groupby('date')['r1'].mean() * 1e4).reindex(D.index))
    res['P'] = L.set_index(['id', 'date'])['p']
    return res


def rule_scores(ctx, dates):
    x = ctx['P'][ctx['P']['date'].isin(dates)].copy()
    G = ctx['G']
    if G is not None and 'twii' in G:
        x = x.merge(G[['date', 'twii']], on='date', how='left')
    rules = [('每格都猜對', x['y']), ('RSI_t > RSI_t-1', (x['drsi'] > 0).astype(int)),
             ('MACD_t > MACD_t-1', (x['dmacd'] > 0).astype(int)), ('當天跳空向下猜 1', (x['gap'] < 0).astype(int))]
    if 'fnet' in x:
        rules.append(('當天外資買超猜 1', (x['fnet'] > 0).astype(int)))
    if 'twii' in x:
        rules.append(('當天 ^TWII 漲就全猜 1', (x['twii'] > 0).astype(int)))
    rules.append(('前一天的 y（合規，排行榜 baseline 7.22）', x['y1']))
    out = []
    for name, sig in rules:
        ok = sig.notna() & x['r'].notna()
        z = x[ok]; s = 2 * sig[ok].astype(float) - 1
        out.append((name, float(((s * z['r']).groupby(z['date']).mean() * 1e4).mean())))
    return out


# ---------------------------------------------------------------- verdict
def ref_range(refs_res, get):
    v = [get(r) for r in refs_res.values()]
    v = [u for u in v if u is not None and np.isfinite(u)]
    return f'{min(v):+.2f}..{max(v):+.2f}' if v else '-'


def fmt_t(t):
    if t is None or not np.isfinite(t):
        return '>99' if t == np.inf else ('<-99' if t == -np.inf else '-')
    return f'{t:.1f}'


def judge(res, refs_res, fmt_flags):
    """returns (level, flags, problem days, description of those days); a flag is (level, category, message)"""
    F = [(lv, 'fmt', msg) for lv, msg in fmt_flags]
    D, nd = res['D'], res['n_days']
    if res['missing']:
        F.append(('FATAL', 'fmt', f'{res["missing"]} 格在資料裡沒有答案（日期不是交易日，或還沒收盤）'))
    # ---- answer-key segments
    exact = set(res['exact'])
    ex = runs(D.phi >= TH['phi_exact'])
    if ex:
        txt = '、'.join(f'{a.date()}..{b.date()}（{n} 天）' if n > 1 else f'{a.date()}' for a, b, n in ex[:3])
        more = f' 等 {len(ex)} 段' if len(ex) > 3 else ''
        F.append(('FATAL', 'exact', f'{len(exact)} 天每支都猜對（phi >= {TH["phi_exact"]}）：{txt}{more}。這些日子等於答案'))
    seg, cut_seg = set(exact), set()
    for c, nb, ab, pb, na, aa, pa, t in res['cut']:
        if pb >= 0.3 and t >= 4:
            cut_seg |= set(D.index[D.index <= c])
            F.append(('FATAL', 'cut', f'{c.date()} 以前 {nb} 天 phi {pb:.2f}、以後 {na} 天 {pa:.2f}（t {fmt_t(t)}）：像是用 '
                                      f'{c.date()} 為止的資料（含 label）訓練，再預測同一段（in-sample）'))
        elif pb >= 0.15 and t >= 3:
            F.append(('WARN', 'cut', f'{c.date()} 以前 phi {pb:.2f}、以後 {pa:.2f}（t {fmt_t(t)}）：可能用到 {c.date()} 為止的 label'))
    seg |= cut_seg
    if 'split' in res:
        c, nb, pb, ab, na, pa, aa, t = res['split']
        before = pb > pa
        hi_phi, hi_n = (pb, nb) if before else (pa, na)
        hi_days = set(D.index[:nb] if before else D.index[nb:])
        if abs(t) >= 6 and hi_phi >= 0.3 and hi_n >= 5:
            if not cut_seg:
                seg |= hi_days
            if not cut_seg and not hi_days <= exact:
                F.append(('FATAL', 'split', f'{c.date()} 前後斷開：以前 {nb} 天 phi {pb:.2f}、以後 {na} 天 {pa:.2f}（t {fmt_t(t)}）：'
                                            f'phi 高的那段用到答案（in-sample）或當天資訊'))
        elif abs(t) >= 5 and hi_phi >= 0.2:
            F.append(('WARN', 'split', f'{c.date()} 前後明顯不同：以前 phi {pb:.2f}、以後 {pa:.2f}（t {fmt_t(t)}）'))
    if len(res['near']):
        F.append(('FATAL', 'near', f'{len(res["near"])} 天 phi 在 {TH["phi_day"]}-{TH["phi_exact"]} 之間（第一天 {res["near"][0].date()}）：'
                                   f'當天幾乎每支都對，只用前一天的資訊做不到'))
    rr = [(a, b, n) for a, b, n in res['runs'] if not (D.loc[a:b].phi >= TH['phi_exact']).all()]
    if rr:
        txt = '、'.join(f'{a.date()}..{b.date()}（{n} 天，phi {D.loc[a:b].phi.mean():.2f}）' for a, b, n in rr[:3])
        more = f' 等 {len(rr)} 段' if len(rr) > 3 else ''
        F.append(('FATAL', 'runs', f'連續 {TH["run_len"]} 天以上 phi >= {TH["phi_run"]}：{txt}{more}：這些日子用到答案（in-sample）或當天資訊'))
    for w, (val, at) in res['roll'].items():
        if val >= TH['roll_fatal'][w]:
            F.append(('FATAL', 'roll', f'{w} 天平均 phi 最高 {val:.2f}（到 {at.date()}；FATAL 門檻 {TH["roll_fatal"][w]}，'
                                       f'沒有 leak 的檔最高 {TH["roll_ok"][w]}）'))
        elif val >= TH['roll_warn'][w]:
            F.append(('WARN', 'roll', f'{w} 天平均 phi 最高 {val:.2f}（到 {at.date()}；WARN 門檻 {TH["roll_warn"][w]}，'
                                      f'沒有 leak 的檔最高 {TH["roll_ok"][w]}）'))
    if nd >= 20:
        if res['phi'] >= TH['phi_fatal']:
            F.append(('FATAL', 'mean', f'平均 phi {res["phi"]:.3f}（沒有 leak 的檔 0.055-0.092）：分數裡有大量答案或當天資訊'))
        elif res['phi'] >= TH['phi_warn']:
            F.append(('WARN', 'mean', f'平均 phi {res["phi"]:.3f}（沒有 leak 的檔 0.055-0.092）：可能有一部分答案或當天資訊'))
    # ---- score
    if res['AR'] >= TH['ar_fatal']:
        F.append(('FATAL', 'score', f'AR {res["AR"]:.2f} bp/日：合規的做法到不了（沒有 leak 的檔 15-24；用真實開盤價當 feature 也只有 17.37）'))
    elif res['AR'] >= TH['ar_warn']:
        F.append(('WARN', 'score', f'AR {res["AR"]:.2f} bp/日，高於沒有 leak 的檔（15-24）：可能是 leak，或在 static 上挑過設定'
                                   f'（compare_static.py 的 --tried 可以調整）'))
    if np.isfinite(res['t_AR']) and res['t_AR'] <= -2.5:
        F.append(('WARN', 'score', f'AR {res["AR"]:.2f}（t {res["t_AR"]:.1f}）顯著為負：0/1 可能寫反，或日期錯位'))
    for m, r in res['M'].iterrows():
        if r.days >= 10 and r.ratio >= TH['ratio_fatal']:
            F.append(('FATAL', 'score', f'{m} 的分數是 oracle 的 {r.ratio:.2f}（{r.AR:.2f} / {r.oracle:.2f}；沒有 leak 的檔 0.16-0.21）'))
        elif r.days >= 10 and r.ratio >= TH['ratio_warn']:
            F.append(('WARN', 'score', f'{m} 的分數是 oracle 的 {r.ratio:.2f}（{r.AR:.2f} / {r.oracle:.2f}；沒有 leak 的檔 0.16-0.21）'))
    # ---- date alignment: report the stronger of k = +1, +2
    k_best = max((1, 2), key=lambda k: res['align'][k][1])
    ar_k, t_k = res['align'][k_best]
    if t_k >= TH['align_t']:
        F.append(('FATAL' if t_k >= TH['align_fatal'] else 'WARN', 'align', f'對第 D+{k_best} 天計分（{ar_k:.2f}）比對第 D 天（{res["align"][0][0]:.2f}）高（t {t_k:.1f}）：'
                                   f'預測可能往前錯了 {k_best} 格，第 D 天的 column 放的是用到第 D 天收盤的預測'))
    # ---- same-day information
    for key, lab, _ in SAMEDAY:
        t = res['same'].get(key)
        if t is None or not np.isfinite(t):
            continue
        rng = ref_range(refs_res, lambda r: r['same'].get(key))
        if t <= TH['t_fatal'][0] or t >= TH['t_fatal'][1]:
            F.append(('FATAL', 'same', f'{lab} 的 t {t:+.1f}（參考檔 {rng}）：用到第 D row 的資訊'))
        elif t <= TH['t_warn'][0] or t >= TH['t_warn'][1]:
            F.append(('WARN', 'same', f'{lab} 的 t {t:+.1f}（參考檔 {rng}）：可能用到第 D row 的資訊'))
    # ---- market timing
    for lab, (r, t), one_sided in (('當天 Top50 盤中報酬', res['tim_mkt'], True), ('當天 ^TWII 變化', res['tim_twii'], False)):
        if not np.isfinite(r):
            continue
        v, tv = (r, t) if one_sided else (abs(r), abs(t))
        if v >= TH['tim_fatal'][0] and tv >= TH['tim_fatal'][1]:
            F.append(('FATAL', 'timing', f'每天猜 1 的比例和{lab}相關 {r:+.2f}（t {t:.1f}；沒有 leak 的檔 -0.12 到 +0.15）：'
                                         f'大盤方向用到當天資訊'))
        elif v >= TH['tim_warn'][0] and tv >= TH['tim_warn'][1]:
            F.append(('WARN', 'timing', f'每天猜 1 的比例和{lab}相關 {r:+.2f}（t {t:.1f}；沒有 leak 的檔 -0.12 到 +0.15）：'
                                        f'可能用到當天的大盤資訊（例如沒有 shift 的 ^TWII、^KS11）'))
    for name, agree in res.get('agree', []):
        if agree >= 0.98:
            F.append(('INFO', 'info', f'和 {name} 有 {agree:.1%} 的格子相同'))
    if res['y_mismatch']:
        F.append(('INFO', 'info', f'資料裡有 {res["y_mismatch"]} 格的 y 和 close > open 不同（計分用 y）'))
    level = 'FATAL' if any(f[0] == 'FATAL' for f in F) else ('WARN' if any(f[0] == 'WARN' for f in F) else 'OK')
    seg = sorted(seg)
    if not seg:
        desc = ''
    elif set(seg) <= exact:
        desc = '等於答案的那些日子'
    else:
        desc = f'{seg[0].date()}..{seg[-1].date()} 那段'
    return level, F, seg, desc


PRIMARY = ('exact', 'cut', 'split', 'near', 'runs')
SUPPORT = ('roll', 'mean')
OTHER = ('align', 'same', 'timing', 'score')
CAT_NAMES = {'score': '分數和逐月', 'same': '當天資訊', 'timing': '大盤 timing', 'align': '日期對齊'}
LV = {'FATAL': 0, 'WARN': 1, 'INFO': 2}


def show_flags(out, F, desc, rest_shown, indent='    '):
    """prints the flags in the order that explains the file best; returns the printed (level, message) pairs"""
    pick = lambda cats: sorted([f for f in F if f[1] in cats], key=lambda f: (LV[f[0]], cats.index(f[1])))
    fmt, prim, sup, oth, info = pick(('fmt',)), pick(PRIMARY), pick(SUPPORT), pick(OTHER), pick(('info',))
    lines = list(fmt)
    if desc:                                   # a block of days explains the rest: show it first, fold what it inflates
        lines += prim[:3] + (sup if not prim else [])
        folded = oth
    else:                                      # no block of days: same-row leaks, timing and shifts come first
        lines += oth + prim[:3] + (sup if not prim else [])
        folded = []
    shown = [(lv, msg) for lv, cat, msg in lines]
    for lv, msg in shown:
        out(f'{indent}{lv:5s} {msg}')
    hidden = len(prim[3:]) + (len(sup) if prim else 0)
    if hidden:
        out(f'{indent}      另外 {hidden} 項答案段的檢查也超過門檻（見 [3]）')
    if folded:
        cats = '、'.join(CAT_NAMES[c] for c in dict.fromkeys(f[1] for f in folded))
        sep = ' ' if desc[:1].isdigit() else ''
        out(f'{indent}      另外 {len(folded)} 項檢查（{cats}）也超過門檻，是{sep}{desc}造成的'
            + ('；扣掉之後的結果在 [9]' if rest_shown else ''))
    for lv, cat, msg in info:
        out(f'{indent}{lv:5s} {msg}')
    return shown + [(lv, msg) for lv, cat, msg in info]


# ---------------------------------------------------------------- report
def same_table(out, res, refs_res, indent='    '):
    out(f'{indent}{"variable":18s} {"t":>7s}   參考檔')
    for key, lab, _ in SAMEDAY:
        t = res['same'].get(key)
        tt = '      -' if t is None or not np.isfinite(t) else f'{t:+7.1f}'
        out(f'{indent}{lab:18s} {tt}   {ref_range(refs_res, lambda r: r["same"].get(key))}')


def report(out, name, info, res, refs_res, level, F, desc, fmt_flags, ctx, rest=None):
    out(f'\n==== {name} ====')
    shape = info.get('shape', ('?', '?'))
    out(f'md5 {info["md5"]}  {info["bytes"]} bytes  {"UTF-8 BOM" if info["bom"] else "沒有 BOM"}  '
        f'CRLF {info["crlf"]} / LF {info["lf"]}  {shape[0]} 支 x {shape[1]} 天')
    M = res['M']
    bad = [f for f in fmt_flags if f[0] in ('FATAL', 'WARN')]
    if not bad and not res['missing']:
        out('[1] 格式：OK（id、日期 column 的集合和順序同 sample_submission；每格都是 0 或 1；每格都有答案）')
    else:
        out('[1] 格式：' + ('有問題' if any(f[0] == 'FATAL' for f in bad) or res['missing'] else '要注意'))
        for lv, msg in bad:
            out(f'    {lv:5s} {msg}')
        if res['missing']:
            out(f'    FATAL {res["missing"]} 格在資料裡沒有答案')
    out('[2] 分數（Top50，bp/日；oracle = 每格都猜對）')
    out(f'    AR {res["AR"]:.2f}（t {fmt_t(res["t_AR"])}）  timing {res["timing"]:.2f}  xsec {res["xsec"]:.2f}  '
        f'準確率 {res["acc"]:.3f}  猜 1 比例 {res["share1"]:.3f}  平均 phi {res["phi"]:.3f}（SE {res["phi_se"]:.3f}）')
    out(f'    {"month":8s} {"days":>4s} {"AR":>8s} {"oracle":>8s} {"AR/orc":>7s} {"acc":>6s} {"phi":>6s}')
    for m, r in M.iterrows():
        out(f'    {str(m):8s} {int(r.days):4d} {r.AR:8.2f} {r.oracle:8.2f} {r.ratio:7.2f} {r.acc:6.3f} {r.phi:6.3f}')
    out('[3] 答案段（phi = 當天 50 支 p 和 y 的相關，1 = 全部等於答案；沒有 leak 的檔單日最高 0.52-0.69）')
    out(f'    每支都猜對（phi >= {TH["phi_exact"]}）的日子：{len(res["exact"])} 天'
        + (f'（{res["exact"][0].date()} .. {res["exact"][-1].date()}）' if len(res['exact']) else '')
        + f'；phi 在 {TH["phi_day"]}-{TH["phi_exact"]} 的日子：{len(res["near"])} 天')
    rr = res['runs']
    out(f'    連續 {TH["run_len"]} 天以上 phi >= {TH["phi_run"]}：'
        + ('沒有' if not rr else '；'.join(f'{a.date()}..{b.date()}（{n} 天）' for a, b, n in rr)))
    if res['roll']:
        out('    5 / 10 / 20 天平均 phi 最高：' + ' / '.join(f'{res["roll"][w][0]:.2f}' for w in (5, 10, 20) if w in res['roll'])
            + '（沒有 leak 的檔最高 0.34 / 0.26 / 0.16）')
    if 'split' in res:
        c, nb, pb, ab, na, pa, aa, t = res['split']
        out(f'    最明顯的切點：{c.date()} 以前 {nb} 天 phi {pb:.2f}、準確率 {ab:.3f}；以後 {na} 天 {pa:.2f}、{aa:.3f}'
            f'（t {fmt_t(t)}；沒有 leak 的檔 |t| 最高 3.1）')
    for c, nb, ab, pb, na, aa, pa, t in res['cut']:
        tag = '原始課程資料最後一天' if c == ctx.get('orig_last') else '指定日期'
        out(f'    {tag} {c.date()}：以前 {nb} 天 準確率 {ab:.3f}、phi {pb:.2f}；以後 {na} 天 {aa:.3f}、{pa:.2f}（t {fmt_t(t)}）')
    out('[4] 日期對齊（p_D 對第 D+k 個交易日計分；k = 0 應該最高；k = -1 會受押昨天反轉影響，只列出來）')
    ks = sorted(res['align'])
    out('    k      ' + ''.join(f'{k:>+9d}' for k in ks))
    out('    AR     ' + ''.join(f'{res["align"][k][0]:9.2f}' for k in ks))
    out('    t vs 0 ' + ''.join(('        -' if k == 0 else f'{res["align"][k][1]:9.1f}') for k in ks))
    out('[5] 當天資訊（t = 2p-1 對「扣掉開盤前可知部分」的殘差；兩向去平均、按日 cluster；'
        'WARN < -4 或 > 6，FATAL < -8 或 > 10）')
    same_table(out, res, refs_res)
    out('[6] 大盤 timing（每天猜 1 比例的相關；沒有 leak 的檔 -0.12 到 +0.15）')
    (r1, t1), (r2, t2), (r3, _) = res['tim_mkt'], res['tim_twii'], res['tim_prev']
    out(f'    當天 Top50 盤中報酬 {r1:+.2f}（t {fmt_t(t1)}；參考檔 {ref_range(refs_res, lambda r: r["tim_mkt"][0])}）')
    out(f'    當天 ^TWII 變化     {r2:+.2f}（t {fmt_t(t2)}；參考檔 {ref_range(refs_res, lambda r: r["tim_twii"][0])}）')
    out(f'    前一天 Top50 盤中報酬 {r3:+.2f}（合規資訊，只列出來）')
    if res.get('agree'):
        out('[7] 和參考檔決策相同的比例：' + '、'.join(f'{n} {a:.3f}' for n, a in res['agree']))
    out(f'[8] 結論：{level}')
    shown = show_flags(out, F, desc, rest is not None)
    if level != 'FATAL':
        out('    看不到的：分散的小量 leak（約 10-20% 的格子以下）、在 static 上挑設定、只影響幾天的訓練截止日錯誤，要看程式。')
    if rest is not None:
        res2, refs2, lv2, F2, desc2 = rest
        D2 = res2['D']
        out(f'[9] 扣掉有問題的日子以後（剩 {len(D2)} 天，{D2.index[0].date()}..{D2.index[-1].date()}；用來看其他日子乾不乾淨）')
        out(f'    AR {res2["AR"]:.2f}（t {fmt_t(res2["t_AR"])}）  準確率 {res2["acc"]:.3f}  平均 phi {res2["phi"]:.3f}  '
            f'猜 1 比例 {res2["share1"]:.3f}')
        out('    日期對齊 AR：' + '  '.join(f'k={k:+d} {res2["align"][k][0]:.2f}' for k in sorted(res2['align'])))
        same_table(out, res2, refs2)
        out(f'    大盤 timing：當天 Top50 {res2["tim_mkt"][0]:+.2f}、當天 ^TWII {res2["tim_twii"][0]:+.2f}')
        out(f'    這些日子的結論：{lv2}' + ('（天數少，只能排除大的 leak）' if len(D2) < 40 else ''))
        show_flags(out, F2, desc2, False, indent='      ')
    return shown


def refs_for(refs, L, cache, ctx):
    key = tuple(sorted(pd.DatetimeIndex(L['date'].unique())))
    if key not in cache:
        sub, keyset = {}, set(key)
        for rn, RL in refs.items():
            RL2 = RL[RL['date'].isin(keyset)]
            if RL2['date'].nunique() >= 5:
                sub[rn] = analyze(RL2, ctx)
        cache[key] = sub
    return cache[key]


def expand(paths):
    out = []
    for p in paths:
        if any(ch in p for ch in '*?['):
            out += sorted(glob.glob(p))
        else:
            out.append(p)
    return list(dict.fromkeys(os.path.abspath(p) for p in out))


def main():
    ap = argparse.ArgumentParser(description='static 上傳檔檢查（格式、答案段、日期錯位、當天資訊、大盤 timing）')
    ap.add_argument('files', nargs='+')
    ap.add_argument('--refs', nargs='*', default=None, help='合規參考檔（預設 v4 static、v9 static w0.25、v9 static）')
    ap.add_argument('--cutoff', nargs='*', default=[], help='另外要比較前後的日期，例如 2026-08-31')
    ap.add_argument('--root', default=os.path.dirname(HERE), help='ML RACE 資料夾（預設是這個檔的上一層）')
    ap.add_argument('--out', default=None, help='報告另存一份文字檔')
    ap.add_argument('--days', default=None, help='每個檔每天的準確率、phi、分數存成 csv')
    a = ap.parse_args()
    root = a.root
    files = expand(a.files)
    if not files:
        print('沒有找到要檢查的檔案')
        return 2
    out = Out(a.out)
    with open(os.path.join(root, 'data', 'info.json'), encoding='utf-8') as fh:
        top50 = json.load(fh)['Top50@Y']
    tpl = pd.read_csv(os.path.join(root, 'sample_submission.csv'), dtype=str, encoding='utf-8-sig', keep_default_na=False)
    tpl.columns = [str(c).strip() for c in tpl.columns]; tpl['id'] = tpl['id'].str.strip()
    ids = list(dict.fromkeys(list(tpl['id']) + list(top50)))
    data_ids = [i for i in ids if os.path.exists(os.path.join(root, 'data', 'market', f'{i}.csv'))]
    P = build_panel(root, data_ids); G = load_global(root)
    if G is not None:
        P = P.merge(G[[c for c in G.columns if c in ('date', 'us', 'nas', 'dvix')]], on='date', how='left')
    cal = np.array(sorted(P['date'].unique()))
    ctx = {'P': P, 'R': P[['id', 'date', 'r']], 'cal': cal, 'G': G, 'cutoffs': []}
    orig = last_date(root, 'data_backup_original', data_ids[0])
    if orig is not None:
        ctx['orig_last'] = orig; ctx['cutoffs'].append(orig)
    for c in a.cutoff:
        c = pd.Timestamp(c)
        if c not in ctx['cutoffs']:
            ctx['cutoffs'].append(c)
    ref_paths = a.refs if a.refs is not None else [os.path.join(root, p) for p in REFS]
    refs = {}
    for p in expand(ref_paths):
        if os.path.exists(p):
            L, info, fl = read_sub(p, tpl, data_ids)
            if L is not None:
                refs[os.path.basename(p)] = L
    tdates = [pd.Timestamp(c) for c in tpl.columns if c != 'id']
    out(f'leak_check.py  資料 {os.path.join(root, "data")}（最後交易日 {pd.Timestamp(cal[-1]).date()}；'
        f'原始課程資料最後一天 {orig.date() if orig is not None else "-"}）')
    out('參考檔：' + ('、'.join(refs) if refs else '沒有（只用固定門檻）'))
    out(f'static 期間（{tdates[0].date()}..{tdates[-1].date()}，{len(tdates)} 天）用當天資訊的規則分數，對照用（bp/日）：')
    out('    ' + '、'.join(f'{n} {v:.2f}' for n, v in rule_scores(ctx, tdates)))
    ref_cache, summary, day_tables = {}, [], []
    for f in files:
        name = os.path.basename(f)
        if not os.path.exists(f):
            out(f'\n==== {name} ====\n    FATAL 找不到檔案')
            summary.append((name, None, 'FATAL', '找不到檔案'))
            continue
        L, info, fmt_flags = read_sub(f, tpl, data_ids)
        if L is None:
            out(f'\n==== {name} ====')
            for lv, msg in fmt_flags:
                out(f'    {lv:5s} {msg}')
            summary.append((name, None, 'FATAL', fmt_flags[-1][1]))
            continue
        refs_res = refs_for(refs, L, ref_cache, ctx)
        res = analyze(L, ctx)
        res['agree'] = []
        for rn, rr in refs_res.items():
            j = res['P'].index.intersection(rr['P'].index)
            if len(j):
                res['agree'].append((rn, float((res['P'][j] == rr['P'][j]).mean())))
        level, F, seg, desc = judge(res, refs_res, fmt_flags)
        rest = None
        if seg:
            L2 = L[~L['date'].isin(set(seg))]
            if L2['date'].nunique() >= 5:
                refs2 = refs_for(refs, L2, ref_cache, ctx)
                res2 = analyze(L2, ctx)
                lv2, F2, _, desc2 = judge(res2, refs2, [])
                rest = (res2, refs2, lv2, F2, desc2)
        shown = report(out, name, info, res, refs_res, level, F, desc, fmt_flags, ctx, rest)
        top = next((m for lv, m in shown if lv == level), '')
        summary.append((name, res, level, top))
        if a.days:
            day_tables.append(res['D'].assign(file=name).rename_axis('date').reset_index())
    if len(files) > 1:
        out('\n==== 總表 ====')
        out(f'{"file":40s} {"AR":>8s} {"acc":>6s} {"phi":>6s}  結論')
        for name, res, level, top in summary:
            nums = f'{res["AR"]:8.2f} {res["acc"]:6.3f} {res["phi"]:6.3f}' if res else f'{"-":>8s} {"-":>6s} {"-":>6s}'
            out(f'{name[:40]:40s} {nums}  {level}' + (f'：{top}' if level != 'OK' else ''))
    if a.days and day_tables:
        pd.concat(day_tables, ignore_index=True).to_csv(a.days, index=False, encoding='utf-8-sig')
        out(f'\n每日表已存到 {a.days}')
    out.close()
    return 1 if any(s[2] == 'FATAL' for s in summary) else 0


if __name__ == '__main__':
    sys.exit(main())
