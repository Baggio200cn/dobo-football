"""诊断实验：LightGBM 为何输给 Elo？

假设：模型复杂度过高 → 过拟合（5056 样本 × 26 特征，400 棵树太多）
对照变体：
  A 原版 LGBM（400树/leaves15）        —— 基线
  B 极简 LGBM（120树/leaves7/强正则）   —— 测试"简化是否有效"
  C 市场锚定：LGBM 只学"市场的修正量"    —— 站在市场肩上
  D 融合：0.5×市场 + 0.5×Elo           —— 最朴素的融合
  E 融合：0.6×市场 + 0.4×LGBM(简)

用同一套时间序列 CV，公平对比。
"""
import sys
import numpy as np, pandas as pd
import lightgbm as lgb
from features import FEAT_PATH, FEATURE_COLS, build
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

Y = {"H": 0, "D": 1, "A": 2}


def brier(P, y):
    T = np.zeros_like(P); T[np.arange(len(y)), y] = 1
    return float(((P - T) ** 2).sum(axis=1).mean())


def norm(P):
    P = np.clip(P, 1e-6, 1); return P / P.sum(axis=1, keepdims=True)


def main(folds=4, test_frac=0.15):
    if not FEAT_PATH.exists(): build()
    df = pd.read_csv(FEAT_PATH, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"]); df = df.sort_values("Date").reset_index(drop=True)
    y = df["FTR"].map(Y).values
    X = df[FEATURE_COLS].values.astype(float)
    MKT = df[["mkt_h", "mkt_d", "mkt_a"]].values.astype(float)
    ELO = df[["elo_p_h", "elo_p_d", "elo_p_a"]].values.astype(float)

    n = len(df); tn = int(n * test_frac)
    starts = [n - tn * (folds - i) for i in range(folds)]
    res = {k: [] for k in ["市场", "Elo", "A原版LGBM", "B极简LGBM", "C市场锚定", "D市场+Elo", "E市场+简LGBM"]}

    for s in starts:
        tr, te = slice(0, s), slice(s, s + tn)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        if len(yte) < 50: continue
        mkt_te, elo_te = MKT[te], ELO[te]

        # A 原版
        a = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=400,
              learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.7, reg_alpha=.5, reg_lambda=1.,
              verbose=-1, random_state=42).fit(Xtr, ytr)
        PA = a.predict_proba(Xte)

        # B 极简（强正则）
        b = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=120,
              learning_rate=0.03, num_leaves=7, min_child_samples=120, subsample=0.7,
              subsample_freq=1, colsample_bytree=0.5, reg_alpha=3., reg_lambda=6.,
              verbose=-1, random_state=42).fit(Xtr, ytr)
        PB = b.predict_proba(Xte)

        # C 市场锚定：目标改为「实际结果 - 市场概率」的残差方向
        #   做法：仍分类，但样本权重降低市场已确定的场次；简化实现=只喂非市场特征
        nonmkt = [i for i, f in enumerate(FEATURE_COLS) if not f.startswith("mkt")]
        c = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=120,
              learning_rate=0.03, num_leaves=7, min_child_samples=120, subsample=0.7,
              subsample_freq=1, colsample_bytree=0.6, reg_alpha=3., reg_lambda=6.,
              verbose=-1, random_state=42).fit(Xtr[:, nonmkt], ytr)
        PCraw = c.predict_proba(Xte[:, nonmkt])
        PC = norm(np.where(np.isnan(mkt_te), PCraw, 0.75 * mkt_te + 0.25 * PCraw))

        # D 市场+Elo
        PD = norm(np.where(np.isnan(mkt_te), elo_te, 0.5 * mkt_te + 0.5 * elo_te))
        # E 市场+简LGBM
        PE = norm(np.where(np.isnan(mkt_te), PB, 0.6 * mkt_te + 0.4 * PB))

        ok = ~np.isnan(mkt_te).any(axis=1)
        res["市场"].append(brier(norm(mkt_te[ok]), yte[ok]))
        res["Elo"].append(brier(elo_te, yte))
        res["A原版LGBM"].append(brier(PA, yte))
        res["B极简LGBM"].append(brier(PB, yte))
        res["C市场锚定"].append(brier(PC, yte))
        res["D市场+Elo"].append(brier(PD, yte))
        res["E市场+简LGBM"].append(brier(PE, yte))

    print(f"{'变体':<16}{'Brier均值':>10}{'vs市场':>10}{'评判':>12}")
    print("-" * 50)
    mk = float(np.mean(res["市场"]))
    order = sorted(res.items(), key=lambda kv: np.mean(kv[1]))
    for k, v in order:
        mval = float(np.mean(v)); d = mval - mk
        tag = "★ 最佳" if k == order[0][0] else ("✅ 优于Elo" if mval < np.mean(res["Elo"]) and k != "市场" else "")
        print(f"{k:<16}{mval:>10.4f}{d:>+10.4f}{tag:>12}")
    print("-" * 50)
    print("（Brier 越低越好 · vs市场 为正=落后市场）")
    return res


if __name__ == "__main__":
    main()
