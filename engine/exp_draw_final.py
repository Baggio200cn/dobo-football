"""v16：平局问题的终审

前四轮已查明并修复：
  · 旧版固定 draw=0.26        → 改为随实力差变化
  · draw_prob 天花板 28.96%   → v15 二维模型提到 34.0%
但用户仍在问「为什么始终没有平局」。还剩两个我没查过的地方：

  A 我们历史上到底放弃了多少分？（所有已锁定期次的实际平局场数）
  B 抽水剔除方式是否系统性压低平局？
    现行 implied_1x2 用「乘法归一化」(p_i = (1/o_i)/Σ(1/o_j))。
    博彩学界公认该法对冷门有偏。若它压低平局，则我们的融合概率
    从源头就偏低 —— 这才是「始终没有平局」的真正上游原因。
  C 决定性检验：在模型给出平局概率最高的那一档里，
    实际的 胜/平/负 三分怎么分？平局到底有没有当过第一？
"""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
from data import load_matches

ZH = {"3": "胜", "1": "平", "0": "负"}


# ---------- 三种抽水剔除法 ----------
def vig_mult(o):
    """乘法归一化（现行）"""
    r = 1 / np.asarray(o, float)
    return r / r.sum(axis=-1, keepdims=True)


def vig_add(o):
    """加法：把超额部分等量减去"""
    r = 1 / np.asarray(o, float)
    excess = (r.sum(-1, keepdims=True) - 1) / r.shape[-1]
    return np.clip(r - excess, 1e-6, 1)


def vig_shin(o, iters=60):
    """Shin (1993)：假设存在知情交易者比例 z，对冷门抽水更重"""
    r = 1 / np.asarray(o, float)
    s = r.sum(-1, keepdims=True)
    z = np.zeros_like(s)
    for _ in range(iters):
        num = np.sqrt(z ** 2 + 4 * (1 - z) * r ** 2 / s)
        p = (num - z) / (2 * (1 - z))
        z = z + 0.5 * (p.sum(-1, keepdims=True) - 1)
        z = np.clip(z, 0, 0.35)
    num = np.sqrt(z ** 2 + 4 * (1 - z) * r ** 2 / s)
    p = (num - z) / (2 * (1 - z))
    return p / p.sum(-1, keepdims=True)


def calib(p, y, name, nb=8):
    """分箱校准：模型说 x%，实际发生多少 %"""
    q = pd.qcut(p, nb, duplicates="drop")
    t = pd.DataFrame({"p": p, "y": y, "q": q})
    g = t.groupby("q", observed=True).agg(n=("y", "size"), 模型=("p", "mean"), 实际=("y", "mean"))
    g["偏差"] = g.实际 - g.模型
    print(f"\n  【{name}】平均绝对偏差 MAE = {g.偏差.abs().mean():.4f}   "
          f"总体：模型 {p.mean():.2%} vs 实际 {y.mean():.2%}  ({y.mean()-p.mean():+.2%})")
    print(f"    {'模型给的':>10}{'实际发生':>10}{'偏差':>9}{'场次':>9}")
    for _, v in g.iterrows():
        flag = "  ←低估" if v.偏差 > 0.012 else ("  ←高估" if v.偏差 < -0.012 else "")
        print(f"    {v.模型:>10.1%}{v.实际:>10.1%}{v.偏差:>+9.1%}{int(v.n):>9,}{flag}")
    return float(g.偏差.abs().mean())


def main():
    # ============ A 历史战绩：我们放弃了多少分 ============
    print("=" * 76)
    print("A  我们历史上放弃了多少分？")
    print("=" * 76)
    tot_m = tot_d = tot_pick_d = 0
    print(f"  {'期号':<8}{'开奖串':<18}{'实际平局':>9}{'我们选平':>9}{'因平局失分':>11}")
    print("  " + "-" * 58)
    for f in sorted(glob.glob(str(BASE / "batches" / "sfc_*_LOCK.json"))):
        L = json.loads(Path(f).read_text(encoding="utf-8"))
        if not L.get("实际开奖"): continue
        res = L["实际开奖"]; s = L["单式"]["串"]
        nd = res.count("1"); npd = s.count("1")
        lost = sum(1 for i in range(14) if res[i] == "1" and s[i] != "1")
        tot_m += 14; tot_d += nd; tot_pick_d += npd
        print(f"  {L['期号']:<8}{res:<18}{nd:>9}{npd:>9}{lost:>11}")
    print("  " + "-" * 58)
    if tot_m:
        print(f"  合计 {tot_m} 场：实际平局 {tot_d} 场（{tot_d/tot_m:.1%}）· "
              f"我们选平 {tot_pick_d} 次 · 全部失分")
        print(f"  → 我们把 {tot_d/tot_m:.0%} 的场次**无条件让给对手**。这是用户质疑的实质。")

    # ============ B 抽水剔除方式是否压低平局 ============
    print("\n" + "=" * 76)
    print("B  抽水剔除方式：乘法归一化是不是系统性压低了平局？")
    print("=" * 76)
    m = load_matches()
    t = m[(m.B365H > 1) & (m.B365D > 1) & (m.B365A > 1)].copy()
    O = t[["B365H", "B365D", "B365A"]].values
    y = (t.FTR == "D").astype(int).values
    print(f"  样本 {len(t):,} 场 · 平均抽水 {(1/O).sum(1).mean()-1:.2%}")
    print(f"  实际平局率 {y.mean():.2%}")
    maes = {}
    for nm, fn in [("乘法归一化（现行）", vig_mult), ("加法剔除", vig_add), ("Shin 法", vig_shin)]:
        P = fn(O)
        maes[nm] = calib(P[:, 1], y, nm)
    best = min(maes, key=maes.get)
    print(f"\n  → 校准最好的是：{best}（MAE {maes[best]:.4f}）")
    if best != "乘法归一化（现行）":
        print(f"  → 现行方法 MAE {maes['乘法归一化（现行）']:.4f}，可改善 "
              f"{maes['乘法归一化（现行）']-maes[best]:.4f}")

    # ============ C 决定性检验 ============
    print("\n" + "=" * 76)
    print("C  决定性检验：平局概率最高的那一档，实际三分怎么分？")
    print("=" * 76)
    P = vig_mult(O)
    t2 = pd.DataFrame({"pd": P[:, 1], "FTR": t.FTR.values})
    t2["档"] = pd.qcut(t2.pd, 10, labels=[f"D{i+1}" for i in range(10)])
    g = t2.groupby("档", observed=True).apply(
        lambda x: pd.Series({"场次": len(x), "市场平局概率": x.pd.mean(),
                             "实际主胜": (x.FTR == "H").mean(),
                             "实际平局": (x.FTR == "D").mean(),
                             "实际客胜": (x.FTR == "A").mean()}), include_groups=False)
    print(f"  {'档位':<6}{'场次':>8}{'市场平局%':>11}{'实主胜':>9}{'实平局':>9}{'实客胜':>9}   平局是第一?")
    print("  " + "-" * 68)
    for k, v in g.iterrows():
        win = max(v.实际主胜, v.实际平局, v.实际客胜)
        isd = "★ 是！" if v.实际平局 >= win - 1e-9 else "否"
        print(f"  {str(k):<6}{int(v.场次):>8,}{v.市场平局概率:>11.1%}"
              f"{v.实际主胜:>9.1%}{v.实际平局:>9.1%}{v.实际客胜:>9.1%}   {isd}")
    print("  " + "-" * 68)
    top = g.iloc[-1]
    print(f"\n  最高档（{int(top.场次):,} 场，市场给平局 {top.市场平局概率:.1%}）：")
    print(f"    实际 主胜 {top.实际主胜:.1%} · 平局 {top.实际平局:.1%} · 客胜 {top.实际客胜:.1%}")
    if top.实际平局 >= max(top.实际主胜, top.实际客胜):
        print("    → 平局确实是最高概率结果。argmax 不选平是错的，必须修。")
    else:
        d = max(top.实际主胜, top.实际客胜) - top.实际平局
        print(f"    → 即使在平局最可能的 10% 场次里，平局仍比第一名低 {d:.1%}。")
        print("    → argmax 不选平，是数据的结论，不是模型的缺陷。")


if __name__ == "__main__":
    main()
