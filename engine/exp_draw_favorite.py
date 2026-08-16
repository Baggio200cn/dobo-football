"""v15 实验：模型有没有能力认出「平局相」的比赛？

用户质疑：14 场一个平都不选，逻辑上说不通。
这个质疑要用三个检验回答，不能用「argmax 数学最优」搪塞：

  Q1 按我们自己的概率，这 14 场「一个平局都不出」的概率是多少？
     —— 如果极低，说明我们押的是一个几乎不可能的世界。
  Q2 市场（博彩公司）会不会把平局定为单场最高概率？会多少次？
     —— 市场会而我们不会，就是模型缺能力，不是策略选择。
  Q3 平局率到底只由「实力差」决定，还是也由「进球总量」决定？
     —— 现行 draw_prob 只吃 Elo 差。若进球总量也显著，模型就是瞎的。
"""
import sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
from data import load_matches
from model import elo_1x2, implied_1x2, draw_prob
import json


def main():
    m = load_matches().sort_values("Date").reset_index(drop=True)

    # ---------- Q1 ----------
    print("=" * 74)
    print("Q1  按我们自己的概率：26106 这 14 场「零平局」的概率")
    print("=" * 74)
    P = json.loads((BASE / "batches" / "sfc_26106.json").read_text(encoding="utf-8"))["picks"]
    dp = np.array([p["平"] for p in P])
    p0 = float(np.prod(1 - dp))
    print(f"  逐场平局概率：{' '.join(f'{x:.0%}' for x in dp)}")
    print(f"  平均 {dp.mean():.1%} · 期望平局数 {dp.sum():.2f} 场")
    print(f"\n  P(14 场一个平局都没有) = {p0:.2%}")
    print(f"  P(至少 1 场平局)       = {1-p0:.2%}")
    # 平局数分布
    from sfc_lock import hit_distribution
    dist = hit_distribution(dp)
    print(f"\n  平局场数分布：", "  ".join(f"{i}场={dist[i]:.1%}" for i in range(0, 8)))
    print(f"  最可能出 {int(dist.argmax())} 场平局")
    print(f"\n  ⚠ 我们的单式串一个 1 都没有 → 只有在概率 {p0:.1%} 的世界里才可能全中。")

    # ---------- Q2 ----------
    print("\n" + "=" * 74)
    print("Q2  市场会不会把平局定为最高概率？（我们从来不会）")
    print("=" * 74)
    t = m[(m.B365H > 1) & (m.B365D > 1) & (m.B365A > 1)].copy()
    Mk = np.array([[implied_1x2(h, d, a)[k] for k in "HDA"]
                   for h, d, a in zip(t.B365H, t.B365D, t.B365A)])
    fav = Mk.argmax(1)
    n = len(t)
    print(f"  样本 {n:,} 场（有 B365 赔率）")
    for i, k in enumerate(["主胜", "平局", "客胜"]):
        print(f"    市场把「{k}」定为最高概率：{(fav==i).sum():>7,} 场  {(fav==i).mean():>6.2%}")
    dfav = fav == 1
    if dfav.sum():
        sub = t[dfav]
        print(f"\n  市场认定平局最可能的 {dfav.sum():,} 场里，实际结果：")
        vc = sub.FTR.value_counts(normalize=True)
        for k, zh in [("H", "主胜"), ("D", "平局"), ("A", "客胜")]:
            print(f"    {zh} {vc.get(k,0):.1%}")
        print(f"  → 市场在这些场次上的判断{'成立' if vc.get('D',0)>max(vc.get('H',0),vc.get('A',0)) else '不成立'}")
    print(f"\n  我们的 Elo 模型：draw_prob 上限 = {draw_prob(0):.2%}（Elo差=0 时）")
    print(f"  此时 主胜=客胜={(1-draw_prob(0))/2:.2%} > 平局 → 平局永远不可能是最高概率")
    print(f"  ⚠ 这不是「策略选择不选平」，是模型**结构上没有能力**选平。")

    # ---------- Q3 ----------
    print("\n" + "=" * 74)
    print("Q3  平局率只由实力差决定吗？还是也由「进球总量」决定？")
    print("=" * 74)
    # 滚动统计每队近 20 场场均总进球，作为「进球倾向」代理（无前视）
    from collections import defaultdict, deque
    gf = defaultdict(lambda: deque(maxlen=20))
    rows = []
    R = {}
    K, HFA = 20.0, 60.0
    for x in m.itertuples():
        rh = R.get(x.HomeTeam, 1500.0); ra = R.get(x.AwayTeam, 1500.0)
        hh, aa = gf[x.HomeTeam], gf[x.AwayTeam]
        tot = (np.mean(hh) + np.mean(aa)) if len(hh) >= 10 and len(aa) >= 10 else np.nan
        rows.append((rh + HFA - ra, tot, x.FTR == "D"))
        eh = 1 / (1 + 10 ** (-(rh + HFA - ra) / 400))
        s = {"H": 1.0, "D": 0.5, "A": 0.0}[x.FTR]
        R[x.HomeTeam] = rh + K * (s - eh)
        R[x.AwayTeam] = ra + K * ((1 - s) - (1 - eh))
        hh.append(x.FTHG + x.FTAG); aa.append(x.FTHG + x.FTAG)
    d = pd.DataFrame(rows, columns=["gap", "tot", "isD"]).dropna()
    d = d.iloc[20000:]
    print(f"  样本 {len(d):,} 场（两队各已有 ≥10 场历史）")

    # 只看势均力敌（|gap|<60），按进球总量分层
    even = d[d.gap.abs() < 60]
    q = pd.qcut(even.tot, 5, labels=["最低20%", "较低", "中等", "较高", "最高20%"])
    g = even.groupby(q, observed=True).agg(场次=("isD", "size"), 平局率=("isD", "mean"),
                                           场均总进球=("tot", "mean"))
    print(f"\n  【势均力敌场次 |Elo差|<60，共 {len(even):,} 场】按进球总量五等分：")
    print(f"  {'进球倾向':<10}{'场次':>9}{'场均总进球':>12}{'实际平局率':>12}{'模型给的':>10}")
    print("  " + "-" * 56)
    mdl = even.gap.map(draw_prob).mean()
    for k, v in g.iterrows():
        flag = "  ← 平局最可能?" if v.平局率 > (1 - v.平局率) / 2 else ""
        print(f"  {str(k):<10}{int(v.场次):>9,}{v.场均总进球:>12.2f}{v.平局率:>12.1%}{mdl:>10.1%}{flag}")
    print("  " + "-" * 56)
    lo = g.平局率.iloc[0]; hi = g.平局率.iloc[-1]
    print(f"  最低进球组 {lo:.1%}  vs  最高进球组 {hi:.1%}   差 {lo-hi:+.1%}")
    print(f"  现行模型对这 5 组给的是**同一个** {mdl:.1%} —— 完全看不见这个维度。")

    # 极端：低进球 + 极均势
    ext = d[(d.gap.abs() < 30) & (d.tot < d.tot.quantile(0.15))]
    if len(ext) > 200:
        pr = ext.isD.mean()
        print(f"\n  【极端组合】|Elo差|<30 且 进球倾向最低 15%：{len(ext):,} 场")
        print(f"    实际平局率 {pr:.1%}   主胜+客胜合计 {1-pr:.1%}（各约 {(1-pr)/2:.1%}）")
        print(f"    → 平局{'确实是单场最高概率结果' if pr > (1-pr)/2 else '仍不是最高概率结果'}")


if __name__ == "__main__":
    main()
