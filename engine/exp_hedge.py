"""v16-B：加保（复式双选）到底该不该带平局？

单选永远不会选平（v16-A 已证明：10 档里没有一档平局是第一）。
但复式加保是**两个**选项 —— 平局完全可以从这里进来。

现行逻辑：按模型概率取前二。
待验证：这个前二，在真实数据上是不是覆盖率最高的组合？
        特别是在「平局相」的比赛上，是不是该选 胜+平 而不是 胜+负？
"""
import sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from data import load_matches

PAIRS = {"胜+平": ("H", "D"), "胜+负": ("H", "A"), "平+负": ("D", "A")}


def vig_mult(o):
    r = 1 / np.asarray(o, float)
    return r / r.sum(axis=-1, keepdims=True)


def main():
    m = load_matches()
    t = m[(m.B365H > 1) & (m.B365D > 1) & (m.B365A > 1)].copy()
    O = t[["B365H", "B365D", "B365A"]].values
    P = vig_mult(O)                       # 市场概率 H,D,A
    ftr = t.FTR.values
    n = len(t)
    print(f"样本 {n:,} 场\n")

    # 现行逻辑：模型前二
    order = np.argsort(-P, axis=1)
    top2 = order[:, :2]
    IDX = {"H": 0, "D": 1, "A": 2}
    yi = np.array([IDX[v] for v in ftr])
    hit_top2 = ((top2[:, 0] == yi) | (top2[:, 1] == yi))
    # 前二是否包含平局
    has_d = (top2 == 1).any(1)

    print("=" * 74)
    print("整体：现行「模型前二」的实际覆盖率")
    print("=" * 74)
    print(f"  模型前二命中率        {hit_top2.mean():>7.2%}")
    print(f"  前二中包含平局的场次   {has_d.mean():>7.2%}  （{has_d.sum():,} 场）")
    for nm, (a, b) in PAIRS.items():
        cov = ((ftr == a) | (ftr == b)).mean()
        print(f"  若全部固定选 {nm}     {cov:>7.2%}")

    # 按平局概率分档，看每档最优组合
    print("\n" + "=" * 74)
    print("按市场平局概率分 10 档：每档哪种加保组合覆盖率最高？")
    print("=" * 74)
    d = pd.DataFrame({"pd": P[:, 1], "FTR": ftr, "top2hit": hit_top2, "hasD": has_d})
    d["档"] = pd.qcut(d.pd, 10, labels=[f"D{i+1}" for i in range(10)])
    print(f"  {'档':<5}{'场次':>7}{'平局%':>8}{'胜+平':>8}{'胜+负':>8}{'平+负':>8}"
          f"{'最优':>7}{'现行前二':>10}{'前二含平':>9}{'损失':>8}")
    print("  " + "-" * 76)
    tot_loss = 0
    for k, g in d.groupby("档", observed=True):
        covs = {nm: float(((g.FTR == a) | (g.FTR == b)).mean()) for nm, (a, b) in PAIRS.items()}
        best = max(covs, key=covs.get)
        cur = float(g.top2hit.mean())
        loss = covs[best] - cur
        tot_loss += loss * len(g)
        print(f"  {str(k):<5}{len(g):>7,}{g.pd.mean():>8.1%}"
              f"{covs['胜+平']:>8.1%}{covs['胜+负']:>8.1%}{covs['平+负']:>8.1%}"
              f"{best:>7}{cur:>10.1%}{g.hasD.mean():>9.0%}{loss:>+8.2%}")
    print("  " + "-" * 76)
    print(f"  加权平均损失：{tot_loss/n:+.3%}   （现行前二 vs 每档事后最优）")

    # 关键子集：模型前二 = 胜+负（排除平局）的比赛，实际打平多少
    excl = ~has_d
    print("\n" + "=" * 74)
    print("被加保排除掉平局的比赛，实际打平多少？")
    print("=" * 74)
    print(f"  前二 = 胜+负（平局被排除）：{excl.sum():,} 场")
    sub = ftr[excl]
    print(f"    实际主胜 {(sub=='H').mean():.1%} · 平局 {(sub=='D').mean():.1%} · 客胜 {(sub=='A').mean():.1%}")
    print(f"    → 覆盖 {((sub=='H')|(sub=='A')).mean():.1%}，漏掉 {(sub=='D').mean():.1%}")
    alt = max(((sub == "H") | (sub == "D")).mean(), ((sub == "D") | (sub == "A")).mean())
    print(f"    → 若改为带平局的组合，最高只能覆盖 {alt:.1%}")
    print(f"    → {'应该改' if alt > ((sub=='H')|(sub=='A')).mean() else '不该改：胜+负 确实更优'}")

    # 三选（全包）的价值
    print("\n" + "=" * 74)
    print("参考：三选全包的覆盖率 = 100%，但注数 ×3")
    print("  16 元 = 8 注：3 个双选(2³=8) vs 1 个三选+1 个双选(3×2=6，浪费 2 注)")
    print("=" * 74)


if __name__ == "__main__":
    main()
