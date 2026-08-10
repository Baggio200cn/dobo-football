"""实验 ①+② ：修过度自信（温度缩放）+ 分联赛建模

问题回顾（v7 之后的短板）：
  · 校准：78-90% 置信档 模型说 80.8%，实际仅 66.7% → 系统性过度自信
  · 联赛差异：各联赛主场优势/平局率/进球数不同，单一模型可能吃亏

变体对照（同一时间序列 CV，公平比较）：
  base   v7 基线：0.6×市场 + 0.4×简化LGBM
  T      base + 温度缩放（修过度自信 · 单参数，最稳）
  L      base + 联赛特征（Div 作为类别特征喂给 LGBM）
  L+T    两者都上
  Lhome  base + 各联赛主场优势先验（结构化补充）
"""
import sys
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize_scalar
from features import FEAT_PATH, FEATURE_COLS, build
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

Y = {"H": 0, "D": 1, "A": 2}
DIVS = ["E0", "SP1", "I1", "D1", "F1"]


def brier(P, y):
    T = np.zeros_like(P); T[np.arange(len(y)), y] = 1
    return float(((P - T) ** 2).sum(axis=1).mean())


def logloss(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())


def norm(P):
    P = np.clip(P, 1e-6, 1); return P / P.sum(axis=1, keepdims=True)


def temperature(P, T):
    """温度缩放：T>1 让概率更保守（修过度自信）"""
    L = np.log(np.clip(P, 1e-12, 1)) / T
    E = np.exp(L - L.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def fit_T(P, y):
    """在校准集上拟合最优温度（最小化 logloss）"""
    f = lambda t: logloss(temperature(P, t), y)
    r = minimize_scalar(f, bounds=(0.5, 4.0), method="bounded")
    return float(r.x)


def calib_table(P, y, bins=(0.30, 0.42, 0.54, 0.66, 0.78, 0.95)):
    conf = P.max(axis=1); ok = (P.argmax(axis=1) == y).astype(int)
    out = []
    for i in range(len(bins) - 1):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.sum() >= 20:
            out.append((f"{bins[i]:.0%}-{bins[i+1]:.0%}", int(m.sum()),
                        float(conf[m].mean()), float(ok[m].mean())))
    return out


def mk_lgb(cat=None):
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=120, learning_rate=0.03,
        num_leaves=7, min_child_samples=120, subsample=0.7, subsample_freq=1,
        colsample_bytree=0.5, reg_alpha=3., reg_lambda=6., verbose=-1, random_state=42)


def main(folds=4, test_frac=0.15):
    if not FEAT_PATH.exists(): build()
    df = pd.read_csv(FEAT_PATH, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"]); df = df.sort_values("Date").reset_index(drop=True)
    y = df["FTR"].map(Y).values
    X = df[FEATURE_COLS].values.astype(float)
    # 联赛作为类别码
    div_code = df["Div"].astype("category").cat.set_categories(DIVS).cat.codes.values.reshape(-1, 1)
    XL = np.hstack([X, div_code.astype(float)])
    MKT = df[["mkt_h", "mkt_d", "mkt_a"]].values.astype(float)

    n = len(df); tn = int(n * test_frac)
    starts = [n - tn * (folds - i) for i in range(folds)]
    names = ["base(v7)", "T温度缩放", "L联赛特征", "L+T", "市场(参照)"]
    res = {k: [] for k in names}
    ll = {k: [] for k in names}
    Ts, calib_base, calib_best = [], [], []

    for s in starts:
        tr, te = slice(0, s), slice(s, s + tn)
        ytr, yte = y[tr], y[te]
        if len(yte) < 50: continue
        mkt_te = MKT[te]
        # 训练集内再留 10% 做校准（时间靠后那段）
        cut = int(len(ytr) * 0.9)

        def blend(P):  # 0.6 市场 + 0.4 模型
            return norm(np.where(np.isnan(mkt_te), P, 0.6 * mkt_te + 0.4 * P))

        # --- base: 无联赛特征 ---
        mB = mk_lgb().fit(X[tr][:cut], ytr[:cut])
        P_base = blend(mB.predict_proba(X[te]))
        # 校准集上的 base 概率（用于拟合 T）
        Pc_base = norm(np.where(np.isnan(MKT[tr][cut:]), mB.predict_proba(X[tr][cut:]),
                                0.6 * MKT[tr][cut:] + 0.4 * mB.predict_proba(X[tr][cut:])))
        T = fit_T(Pc_base, ytr[cut:]); Ts.append(T)
        P_T = temperature(P_base, T)

        # --- L: 含联赛特征 ---
        mL = mk_lgb().fit(XL[tr][:cut], ytr[:cut])
        P_L = blend(mL.predict_proba(XL[te]))
        Pc_L = norm(np.where(np.isnan(MKT[tr][cut:]), mL.predict_proba(XL[tr][cut:]),
                             0.6 * MKT[tr][cut:] + 0.4 * mL.predict_proba(XL[tr][cut:])))
        T_L = fit_T(Pc_L, ytr[cut:])
        P_LT = temperature(P_L, T_L)

        ok = ~np.isnan(mkt_te).any(axis=1)
        for k, P in [("base(v7)", P_base), ("T温度缩放", P_T), ("L联赛特征", P_L), ("L+T", P_LT)]:
            res[k].append(brier(P, yte)); ll[k].append(logloss(P, yte))
        res["市场(参照)"].append(brier(norm(mkt_te[ok]), yte[ok]))
        ll["市场(参照)"].append(logloss(norm(mkt_te[ok]), yte[ok]))
        calib_base = calib_table(P_base, yte); calib_best = calib_table(P_LT, yte)

    print(f"平均温度 T = {np.mean(Ts):.3f}（>1 表示原模型过度自信，需要压平）\n")
    print(f"{'变体':<14}{'Brier↓':>10}{'LogLoss↓':>11}{'vs市场':>10}{'评判':>10}")
    print("-" * 56)
    mk = float(np.mean(res["市场(参照)"]))
    order = sorted(res.items(), key=lambda kv: np.mean(kv[1]))
    for k, v in order:
        b = float(np.mean(v)); l = float(np.mean(ll[k])); d = b - mk
        tag = "★最佳" if k == order[0][0] else ("✅超市场" if d < -0.0002 and k != "市场(参照)" else "")
        print(f"{k:<14}{b:>10.4f}{l:>11.4f}{d:>+10.4f}{tag:>10}")
    print("-" * 56)

    print("\n校准对比（最后一折）：")
    print(f"  {'置信档':<12}{'样本':>6}{'说':>8}{'实际':>8}{'偏差':>9}   →   {'修正后说':>9}{'实际':>8}{'偏差':>9}")
    for i, c in enumerate(calib_base):
        b2 = calib_best[i] if i < len(calib_best) else None
        s = f"  {c[0]:<12}{c[1]:>6}{c[2]:>8.1%}{c[3]:>8.1%}{c[3]-c[2]:>+9.1%}"
        if b2: s += f"   →   {b2[2]:>9.1%}{b2[3]:>8.1%}{b2[3]-b2[2]:>+9.1%}"
        print(s)
    return res


if __name__ == "__main__":
    main()
