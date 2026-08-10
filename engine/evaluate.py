"""回测：训练集拟合，测试集(最近赛季)样本外评估。
指标：Brier score + log loss（胜平负三分类），对比 Elo / 泊松 / 市场。
诚实预期：市场基准通常很强，模型能接近就已不错。"""
import sys, numpy as np, pandas as pd
from config import SEASONS
from data import load_matches
from model import Poisson, elo_run, elo_1x2, implied_1x2
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

OUT = ["H", "D", "A"]


def onehot(ftr):
    return {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}[ftr]


def brier(p, y):  # 多分类 Brier
    return sum((p[o] - y[i]) ** 2 for i, o in enumerate(OUT))


def logloss(p, y):
    return -sum(y[i] * np.log(max(p[o], 1e-12)) for i, o in enumerate(OUT))


def run():
    m = load_matches()
    # 用最后一个赛季当测试集：以最晚 6 个月为界（近似一个赛季）
    cut = m["Date"].max() - pd.Timedelta(days=300)
    train, test = m[m.Date < cut], m[m.Date >= cut]
    print(f"训练 {len(train)} 场 / 测试 {len(test)} 场（{test.Date.min().date()}→{test.Date.max().date()}）\n")

    # 多个基线在 train 拟合
    pois = Poisson().fit(train)
    poisxg = Poisson(use_xg=True).fit(train)
    poisdc = Poisson().fit(train).fit_rho(train)   # D · Dixon-Coles 修正
    print(f"（Dixon-Coles rho = {poisdc.rho:.3f}）")
    # Elo：train 跑出评分后，继续在 test 上 walk-forward
    r, _ = elo_run(train)

    acc = {"elo": [], "poisson(进球)": [], "poisson-xG(射正)": [], "poisson-DC": [], "market": []}
    for m_ in test.itertuples():
        y = onehot(m_.FTR)
        rh, ra = r.get(m_.HomeTeam, 1500), r.get(m_.AwayTeam, 1500)
        pe = elo_1x2(rh, ra)
        pp = pois.predict(m_.HomeTeam, m_.AwayTeam)
        px = poisxg.predict(m_.HomeTeam, m_.AwayTeam)
        pdc = poisdc.predict(m_.HomeTeam, m_.AwayTeam)
        pm = implied_1x2(getattr(m_, "B365H", None), getattr(m_, "B365D", None), getattr(m_, "B365A", None))
        acc["elo"].append((brier(pe, y), logloss(pe, y)))
        acc["poisson(进球)"].append((brier(pp, y), logloss(pp, y)))
        acc["poisson-xG(射正)"].append((brier(px, y), logloss(px, y)))
        acc["poisson-DC"].append((brier(pdc, y), logloss(pdc, y)))
        if pm:
            acc["market"].append((brier(pm, y), logloss(pm, y)))
        # 更新 Elo
        eh = 1 / (1 + 10 ** ((ra - rh - 60) / 400))
        sh = 1.0 if m_.FTHG > m_.FTAG else (0.5 if m_.FTHG == m_.FTAG else 0.0)
        r[m_.HomeTeam] = rh + 20 * (sh - eh); r[m_.AwayTeam] = ra + 20 * ((1 - sh) - (1 - eh))

    print(f"{'模型':<20}{'Brier↓':>10}{'LogLoss↓':>12}{'样本':>8}")
    print("-" * 50)
    for k in ("elo", "poisson(进球)", "poisson-xG(射正)", "poisson-DC", "market"):
        a = np.array(acc[k]) if acc[k] else None
        if a is None: continue
        print(f"{k:<20}{a[:,0].mean():>10.4f}{a[:,1].mean():>12.4f}{len(a):>8}")
    print("\n（越低越好；市场=Bet365 去水隐含概率，通常是最强基准）")
    return acc


if __name__ == "__main__":
    run()
