"""实验 ④：赔率轨迹（开盘 → 收盘 = 资金流向）

⚠️ 方法论要点（必须诚实对待）：
  收盘赔率在「开赛时」才确定，是公认最锐利的市场共识。
  因此把它当特征用，必须同时用「收盘市场」当基准来对比——
  只跟开盘市场比会得到虚高的结论（因为我们偷看了更晚的信息）。

变体：
  base       基础特征 + 0.6×开盘市场          （赛前早期即可产出）
  +traj      基础+轨迹特征 + 0.6×开盘市场
  +traj@close 基础+轨迹 + 0.6×收盘市场         （临场预测才可用）
基准：
  开盘市场 / 收盘市场
"""
import sys
import numpy as np, pandas as pd
import lightgbm as lgb
from features import FEAT_PATH, FEATURE_BASE, FEATURE_ODDS_TRAJ, build
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

Y = {"H": 0, "D": 1, "A": 2}


def brier(P, y):
    T = np.zeros_like(P); T[np.arange(len(y)), y] = 1
    return float(((P - T) ** 2).sum(axis=1).mean())


def logloss(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())


def norm(P):
    P = np.clip(P, 1e-6, 1); return P / P.sum(axis=1, keepdims=True)


def mk():
    return lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=120,
        learning_rate=0.03, num_leaves=7, min_child_samples=120, subsample=0.7,
        subsample_freq=1, colsample_bytree=0.5, reg_alpha=3., reg_lambda=6.,
        verbose=-1, random_state=42)


def main(folds=4, test_frac=0.15):
    if not FEAT_PATH.exists(): build()
    df = pd.read_csv(FEAT_PATH, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"]); df = df.sort_values("Date").reset_index(drop=True)
    y = df["FTR"].map(Y).values
    ALL = FEATURE_BASE + FEATURE_ODDS_TRAJ
    Xb = df[FEATURE_BASE].values.astype(float)
    Xa = df[ALL].values.astype(float)
    OPEN = df[["mkt_h", "mkt_d", "mkt_a"]].values.astype(float)
    CLOSE = df[["mkc_h", "mkc_d", "mkc_a"]].values.astype(float)

    n = len(df); tn = int(n * test_frac)
    starts = [n - tn * (folds - i) for i in range(folds)]
    keys = ["开盘市场", "收盘市场", "base", "+traj", "+traj@close"]
    res = {k: [] for k in keys}; ll = {k: [] for k in keys}
    imp = np.zeros(len(ALL))

    for s in starts:
        tr, te = slice(0, s), slice(s, s + tn)
        ytr, yte = y[tr], y[te]
        if len(yte) < 50: continue
        op, cl = OPEN[te], CLOSE[te]

        mB = mk().fit(Xb[tr], ytr); PB = mB.predict_proba(Xb[te])
        mA = mk().fit(Xa[tr], ytr); PA = mA.predict_proba(Xa[te]); imp += mA.feature_importances_

        P_base = norm(np.where(np.isnan(op), PB, .6 * op + .4 * PB))
        P_traj = norm(np.where(np.isnan(op), PA, .6 * op + .4 * PA))
        P_trjc = norm(np.where(np.isnan(cl), PA, .6 * cl + .4 * PA))

        for k, P in [("开盘市场", norm(op)), ("收盘市场", norm(cl)),
                     ("base", P_base), ("+traj", P_traj), ("+traj@close", P_trjc)]:
            ok = ~np.isnan(P).any(axis=1)
            res[k].append(brier(P[ok], yte[ok])); ll[k].append(logloss(P[ok], yte[ok]))

    print(f"{'变体':<14}{'Brier↓':>10}{'LogLoss↓':>11}{'vs开盘':>10}{'vs收盘':>10}")
    print("-" * 58)
    mo = float(np.mean(res["开盘市场"])); mc = float(np.mean(res["收盘市场"]))
    for k in sorted(keys, key=lambda z: np.mean(res[z])):
        b = float(np.mean(res[k])); l = float(np.mean(ll[k]))
        print(f"{k:<14}{b:>10.4f}{l:>11.4f}{b-mo:>+10.4f}{b-mc:>+10.4f}")
    print("-" * 58)
    print(f"\n收盘 vs 开盘：{mc-mo:+.4f}（负=收盘更准，符合「收盘线更锐利」的常识）")

    print("\n赔率轨迹特征的重要性排名：")
    order = np.argsort(-imp)
    rank = {ALL[i]: r + 1 for r, i in enumerate(order)}
    for f in FEATURE_ODDS_TRAJ:
        print(f"  #{rank[f]:<3} {f:<14} {imp[ALL.index(f)]:.0f}")
    print("\n全部特征 TOP8：")
    for i in order[:8]:
        star = " ★轨迹" if ALL[i] in FEATURE_ODDS_TRAJ else ""
        print(f"  {ALL[i]:<14} {imp[i]:.0f}{star}")
    return res


if __name__ == "__main__":
    main()
