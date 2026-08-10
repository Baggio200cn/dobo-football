"""时间序列交叉验证 + LightGBM 两层堆叠

铁律：绝不用随机 CV。按时间滚动切分——用过去训练，预测未来。
    折1: [训练 ....][测试 ..]
    折2: [训练 ......][测试 ..]
    折3: [训练 ........][测试 ..]

对比：市场基准 / Elo 结构模型 / LightGBM（第二层）
指标：Brier · LogLoss（越低越好）+ 概率校准

用法：
  python timecv.py            # 跑 4 折时间序列 CV
  python timecv.py --folds 6
"""
import sys, json
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from config import BASE
from features import FEAT_PATH, FEATURE_COLS, build
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

CLS = ["H", "D", "A"]
Y = {"H": 0, "D": 1, "A": 2}


def load_feat():
    if not FEAT_PATH.exists():
        print("特征表不存在，先构建…"); build()
    df = pd.read_csv(FEAT_PATH, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def brier(P, y):
    """P: (n,3) 概率; y: (n,) 0/1/2"""
    T = np.zeros_like(P); T[np.arange(len(y)), y] = 1
    return float(((P - T) ** 2).sum(axis=1).mean())


def logloss(P, y):
    p = np.clip(P[np.arange(len(y)), y], 1e-12, 1)
    return float(-np.log(p).mean())


def calib_report(P, y, bins=5):
    """校准检查：模型说 X%，实际发生率是多少"""
    out = []
    conf = P.max(axis=1); pred = P.argmax(axis=1)
    correct = (pred == y).astype(int)
    edges = np.linspace(0.3, 0.9, bins + 1)
    for i in range(bins):
        msk = (conf >= edges[i]) & (conf < edges[i + 1])
        if msk.sum() >= 20:
            out.append({"band": f"{edges[i]:.0%}-{edges[i+1]:.0%}",
                        "n": int(msk.sum()),
                        "said": float(conf[msk].mean()),
                        "actual": float(correct[msk].mean())})
    return out


def run(folds=4, test_frac=0.15):
    df = load_feat()
    y_all = df["FTR"].map(Y).values
    X_all = df[FEATURE_COLS].values.astype(float)
    mkt_all = df[["mkt_h", "mkt_d", "mkt_a"]].values.astype(float)
    elo_all = df[["elo_p_h", "elo_p_d", "elo_p_a"]].values.astype(float)

    n = len(df)
    test_n = int(n * test_frac)
    # 滚动起点：每折测试窗口向后推进
    starts = [n - test_n * (folds - i) for i in range(folds)]

    agg = {"市场基准": [], "Elo 结构": [], "LightGBM": [], "LGB+校准": []}
    imp_sum = np.zeros(len(FEATURE_COLS))
    calib_last = []

    print(f"样本 {n} 场 · {folds} 折时间序列 CV · 每折测试 {test_n} 场\n")
    print(f"{'折':<4}{'训练':>7}{'测试':>7}{'市场':>9}{'Elo':>9}{'LGBM':>9}{'LGB+校准':>10}")
    print("-" * 56)

    for k, s in enumerate(starts, 1):
        tr = slice(0, s); te = slice(s, s + test_n)
        Xtr, ytr = X_all[tr], y_all[tr]
        Xte, yte = X_all[te], y_all[te]
        if len(yte) < 50: continue

        # 内部再切一小段做校准集（仍是时间在前）
        cut = int(len(ytr) * 0.9)
        Xf, yf = Xtr[:cut], ytr[:cut]
        Xc, yc = Xtr[cut:], ytr[cut:]

        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=400,
            learning_rate=0.03, num_leaves=15, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=1.0, verbose=-1, random_state=42)
        m.fit(Xf, yf)
        imp_sum += m.feature_importances_

        P_lgb = m.predict_proba(Xte)
        # 校准（对最大类概率做 isotonic，再归一）
        Pc_raw = m.predict_proba(Xc)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(Pc_raw.max(axis=1), (Pc_raw.argmax(axis=1) == yc).astype(float))
        adj = iso.predict(P_lgb.max(axis=1))
        P_cal = P_lgb.copy()
        idx = P_lgb.argmax(axis=1)
        for i in range(len(P_cal)):
            old = P_cal[i, idx[i]]; new = np.clip(adj[i], 0.05, 0.95)
            if old > 0:
                rest = 1 - new
                others = np.delete(np.arange(3), idx[i])
                tot = P_cal[i, others].sum()
                P_cal[i, idx[i]] = new
                if tot > 0: P_cal[i, others] *= rest / tot
        P_cal /= P_cal.sum(axis=1, keepdims=True)

        P_mkt = mkt_all[te]; P_elo = elo_all[te]
        ok = ~np.isnan(P_mkt).any(axis=1)

        r = {
            "市场基准": brier(P_mkt[ok], yte[ok]),
            "Elo 结构": brier(P_elo, yte),
            "LightGBM": brier(P_lgb, yte),
            "LGB+校准": brier(P_cal, yte),
        }
        for kk, v in r.items(): agg[kk].append(v)
        calib_last = calib_report(P_cal, yte)

        print(f"{k:<4}{s:>7}{len(yte):>7}{r['市场基准']:>9.4f}{r['Elo 结构']:>9.4f}"
              f"{r['LightGBM']:>9.4f}{r['LGB+校准']:>10.4f}")

    print("-" * 56)
    means = {k: float(np.mean(v)) for k, v in agg.items() if v}
    print(f"{'均值':<18}{means['市场基准']:>9.4f}{means['Elo 结构']:>9.4f}"
          f"{means['LightGBM']:>9.4f}{means['LGB+校准']:>10.4f}")

    best_ours = min(means["Elo 结构"], means["LightGBM"], means["LGB+校准"])
    gap = best_ours - means["市场基准"]
    print(f"\n我们最好的模型与市场差距：{gap:+.4f} Brier"
          f"（{'已超越市场！需谨慎复核' if gap < 0 else '仍落后，持续逼近中'}）")

    print("\n特征重要性 TOP10：")
    order = np.argsort(-imp_sum)
    for i in order[:10]:
        print(f"  {FEATURE_COLS[i]:<16} {imp_sum[i]:.0f}")

    if calib_last:
        print("\n概率校准检查（最后一折）：")
        print(f"  {'置信区间':<12}{'样本':>6}{'模型说':>9}{'实际':>9}{'偏差':>9}")
        for c in calib_last:
            d = c["actual"] - c["said"]
            print(f"  {c['band']:<12}{c['n']:>6}{c['said']:>9.1%}{c['actual']:>9.1%}{d:>+9.1%}")

    out = {"means": means, "gap_vs_market": gap,
           "top_features": [{"f": FEATURE_COLS[i], "imp": float(imp_sum[i])} for i in order[:10]],
           "calibration": calib_last, "folds": folds}
    (BASE / "cv_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 结果已存 cv_result.json")
    return out


if __name__ == "__main__":
    f = 4
    if "--folds" in sys.argv:
        try: f = int(sys.argv[sys.argv.index("--folds") + 1])
        except Exception: pass
    run(f)
