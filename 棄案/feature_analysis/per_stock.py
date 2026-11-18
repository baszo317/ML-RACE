import pandas as pd, numpy as np, os, json
HERE=os.path.dirname(os.path.abspath(__file__))
A=pd.read_pickle(os.path.join(HERE,'features.pkl')); IC=pd.read_csv(os.path.join(HERE,'per_stock_feature_IC.csv'),index_col=0,dtype={0:str}); IC.index=IC.index.astype(str)
names={'2330':'台積電','2317':'鴻海','2454':'聯發科','2308':'台達電','2881':'富邦金','2882':'國泰金','2382':'廣達','2412':'中華電','2891':'中信金','3711':'日月光投控','2886':'兆豐金','2303':'聯電','6669':'緯穎','5871':'中租-KY','2884':'玉山金','2357':'華碩','2345':'智邦','1216':'統一','2885':'元大金','2603':'長榮','3045':'台灣大','2892':'第一金','2880':'華南金','6505':'台塑化','5880':'合庫金','3231':'緯創','3017':'奇鋐','2383':'台光電','2207':'和泰車','2327':'國巨','4904':'遠傳','2002':'中鋼','3008':'大立光','2890':'永豐金','2883':'凱基金','1303':'南亞','2301':'光寶科','2395':'研華','3034':'聯詠','2379':'瑞昱','8069':'元太','2887':'台新金','2912':'統一超','3661':'世芯-KY','2059':'川湖','3653':'健策','1301':'台塑','2615':'萬海','3293':'鈊象','2609':'陽明','2801':'彰銀','2618':'長榮航','4938':'和碩','6223':'旺矽','5876':'上海商銀','1101':'台泥','2408':'南亞科','3037':'欣興','2360':'致茂','6446':'藥華藥','8046':'南電','3443':'創意','2376':'技嘉','3529':'力旺','1590':'亞德客-KY','6488':'環球晶','1519':'華城','5347':'世界','3665':'貿聯-KY','5274':'信驊','2368':'金像電','2449':'京元電子','1326':'台化','1402':'遠東新','3533':'嘉澤','2356':'英業達','2633':'台灣高鐵','3036':'文曄','1504':'東元','1102':'亞泥','2834':'臺企銀','2324':'仁寶','2105':'正新','8299':'群聯','2474':'可成','2610':'華航','3044':'健鼎','2404':'漢唐','6415':'矽力*-KY','5269':'祥碩','2344':'華邦電','4958':'臻鼎-KY','9910':'豐泰','8464':'億豐','1476':'儒鴻','2377':'微星','2812':'台中銀','6409':'旭隼','3702':'大聯大','6274':'台燿'}
semi={'2330','2454','2303','3711','2379','3034','3661','5274','3443','2408','2344','6415','3529','5347','8299','6488','2449','6223','5269'}
fin={s for s in names if s.startswith('28') and s!='2834'}|{'2834','5871','5876','5880'}
trad={'1216','6505','2207','2002','1303','2912','1301','2615','2609','2603','2618','1101','1326','1402','2633','1504','1102','2105','2610','9910','8464','1476','1590','1519','2412','3045','4904','6446','3293'}
def grp(s): return '金融' if s in fin else '半導體' if s in semi else '傳產/電信/航運/其他' if s in trad else '電子/AI硬體'
key=['txoi_chg','us_ixic_ret1_lag1','us_vix_chg_lag1','us_gspc_ret1','ret5','intraday_ret','trust_net','foreign_net','dealer_self_net','clv','range','RSI','BB_pctb']
rows=[]
for sid,d in A.groupby('sid'):
    ic=IC.loc[sid]; top=ic.abs().sort_values(ascending=False).index[:3]
    te=d[d.date>='2025-01-01']
    rows.append(dict(sid=sid,name=names.get(sid,''),group=grp(sid),top50=sid in set(A[A.top50].sid),n=len(d),y_rate=d.y.mean(),
        tie_rate=(d.target_r==0).mean(),mean_intra_bp=d.target_r.mean()*1e4,all0_AR_2025_26_bp=-te.target_r.mean()*1e4,
        top3_features=', '.join(f"{f}({ic[f]:+.3f})" for f in top),**{f'IC_{k}':ic[k] for k in key}))
P=pd.DataFrame(rows).sort_values(['top50','sid'],ascending=[False,True])
P.round(4).to_csv(os.path.join(HERE,'per_stock_summary.csv'),index=False,encoding='utf-8-sig')
pd.set_option('display.width',300); pd.set_option('display.max_rows',200); pd.set_option('display.max_colwidth',70)
print(P[['sid','name','group','top50','y_rate','tie_rate','mean_intra_bp','all0_AR_2025_26_bp','top3_features']].round(3).to_string(index=False))
print(P.groupby('group')[['y_rate','tie_rate','mean_intra_bp']+[f'IC_{k}' for k in key]].mean().T.round(4))
print(P.groupby('group').size())
# how many per-stock |IC|>0.05 overall vs expected by chance
print('frac |IC|>0.05:',(IC.abs()>0.05).mean().mean(), ' expected under null approx', 2*(1-0.977))
