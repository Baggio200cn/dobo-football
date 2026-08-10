"""实验 ⑤：主力缺阵指数 是否提升预测？

变体：
  base      当前最好（基础+赔率轨迹）
  +inj      再加 主客队缺阵人数/权重/差值
仅英超子集单独评估（因为缺阵数据只覆盖英超）。
"""
import sys
import numpy as np, pandas as pd
import lightgbm as lgb
from config import BASE
from features import FEAT_PATH, FEATURE_BASE, FEATURE_ODDS_TRAJ
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

Y = {"H": 0, "D": 1, "A": 2}
INJ = ["abs_n_h", "abs_n_a", "abs_n_diff", "abs_w_h", "abs_w_a", "abs_w_diff"]


def brier(P, y):
    T = np.zeros_like(P); T[np.arange(len(y)), y] = 1
    return float(((P - T) ** 2).sum(axis=1).mean())


def norm(P):
    P = np.clip(P, 1e-6, 1); return P / P.sum(axis=1, keepdims=True)


def mk():
    return lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=120,
        learning_rate=0.03, num_leaves=7, min_child_samples=120, subsample=0.7,
        subsample_freq=1, colsample_bytree=0.5, reg_alpha=3., reg_lambda=6.,
        verbose=-1, random_state=42)


def load():
    df = pd.read_csv(FEAT_PATH, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"]); df = df.sort_values("Date").reset_index(drop=True)
    A = pd.read_csv(BASE / "availability.csv", encoding="utf-8-sig")
    A["date"] = pd.to_datetime(A["date"])
    # 用 (日期, 球队) 关联；日期允许 ±1 天容差 → 先按精确日，再按队+最近日期兜底
    key = {}
    for r in A.itertuples():
        key[(r.date.date(), r.team)] = (r.absent_n, r.absent_w)
    def look(d, t):
        dd = d.date()
        for off in (0, -1, 1, -2, 2):
            k = (dd + pd.Timedelta(days=off).to_pytimedelta(), t)
            if k in key: return key[k]
        return (np.nan, np.nan)
    hn, hw, an, aw = [], [], [], []
    for r in df.itertuples():
        a1, w1 = look(r.Date, r.HomeTeam); a2, w2 = look(r.Date, r.AwayTeam)
        hn.append(a1); hw.append(w1); an.append(a2); aw.append(w2)
    df["abs_n_h"], df["abs_w_h"] = hn, hw
    df["abs_n_a"], df["abs_w_a"] = an, aw
    df["abs_n_diff"] = df.abs_n_h - df.abs_n_a
    df["abs_w_diff"] = df.abs_w_h - df.abs_w_a
    return df


def main(folds=4, test_frac=0.15):
    df = load()
    epl = df.Div == "E0"
    cov = df.loc[epl, "abs_n_h"].notna().mean()
    print(f"关联覆盖率 · 英超 {cov:.1%}（{int(df.loc[epl,'abs_n_h'].notna().sum())}/{int(epl.sum())} 场）")
    print(f"全库覆盖率 {df.abs_n_h.notna().mean():.1%}（其余联赛无 FPL 数据）\n")
    if cov < 0.5:
        print("⚠ 覆盖率过低，关联可能失败"); return

    y = df["FTR"].map(Y).values
    BASEF = FEATURE_BASE + FEATURE_ODDS_TRAJ
    Xb = df[BASEF].values.astype(float)
    Xi = df[BASEF + INJ].values.astype(float)
    OPEN = df[["mkt_h", "mkt_d", "mkt_a"]].values.astype(float)

    n = len(df); tn = int(n * test_frac)
    starts = [n - tn * (folds - i) for i in range(folds)]
    keys = ["市场", "base", "+inj"]
    allr = {k: [] for k in keys}; eplr = {k: [] for k in keys}
    imp = np.zeros(len(BASEF + INJ))

    for s in starts:
        tr, te = slice(0, s), slice(s, s + tn)
        ytr, yte = y[tr], y[te]
        if len(yte) < 50: continue
        op = OPEN[te]
        is_epl = epl.values[te]

        mb = mk().fit(Xb[tr], ytr); Pb = mb.predict_proba(Xb[te])
        mi = mk().fit(Xi[tr], ytr); Pi = mi.predict_proba(Xi[te]); imp += mi.feature_importances_
        P_base = norm(np.where(np.isnan(op), Pb, .6 * op + .4 * Pb))
        P_inj = norm(np.where(np.isnan(op), Pi, .6 * op + .4 * Pi))

        for k, P in [("市场", norm(op)), ("base", P_base), ("+inj", P_inj)]:
            ok = ~np.isnan(P).any(axis=1)
            allr[k].append(brier(P[ok], yte[ok]))
            m2 = ok & is_epl
            if m2.sum() > 30: eplr[k].append(brier(P[m2], yte[m2]))

    print(f"{'变体':<8}{'全库Brier':>11}{'英超Brier':>11}{'英超vs base':>13}")
    print("-" * 46)
    b_all = {k: float(np.mean(v)) for k, v in allr.items()}
    b_epl = {k: float(np.mean(v)) for k, v in eplr.items() if v}
    for k in keys:
        d = b_epl[k] - b_epl["base"] if k in b_epl else np.nan
        print(f"{k:<8}{b_all[k]:>11.4f}{b_epl.get(k,np.nan):>11.4f}{d:>+13.4f}")
    print("-" * 46)
    gain_all = b_all["base"] - b_all["+inj"]
    gain_epl = b_epl["base"] - b_epl["+inj"]
    print(f"\n增益（正=有效）：全库 {gain_all:+.4f} · 英超 {gain_epl:+.4f}")

    ALL = BASEF + INJ
    order = np.argsort(-imp)
    rank = {ALL[i]: r + 1 for r, i in enumerate(order)}
    print(f"\n缺阵特征重要性排名（共 {len(ALL)} 个特征）：")
    for f in INJ:
        print(f"  #{rank[f]:<3} {f:<12} {imp[ALL.index(f)]:.0f}")
    return b_all, b_epl


if __name__ == "__main__":
    main()
