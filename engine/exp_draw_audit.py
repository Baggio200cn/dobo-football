"""v13 实验：平局审计

要回答三个问题（两期实战后，「平局是不是我们的系统性短板」）：
  Q1 平局概率模型 DRAW_FIT 在 10.7 万场全量上校准得准不准？
  Q2 「从不选平」这个 argmax 策略，理论上让我们损失了多少命中率？
  Q3 有没有一条规则（在某些场次改选平）能提升命中率？—— 回测检验
"""
import sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from data import load_matches
from model import elo_run, elo_1x2, draw_prob, DRAW_FIT


def main():
    m = load_matches().sort_values("Date").reset_index(drop=True)
    print(f"全量：{len(m):,} 场 / {m.Div.nunique()} 联赛 / {m.Date.min():%Y-%m} ~ {m.Date.max():%Y-%m}")

    # 滚动 Elo：预测时只用该场之前的信息（无前视）
    r, hist = elo_run(m, return_pre=True) if "return_pre" in elo_run.__code__.co_varnames \
              else (None, None)
    if hist is None:
        # 兼容：手动重跑一遍滚动 Elo，记录每场赛前评分
        R, rows = {}, []
        K, HFA = 20.0, 60.0
        for x in m.itertuples():
            rh = R.get(x.HomeTeam, 1500.0); ra = R.get(x.AwayTeam, 1500.0)
            rows.append((rh, ra))
            eh = 1 / (1 + 10 ** (-(rh + HFA - ra) / 400))
            s = {"H": 1.0, "D": 0.5, "A": 0.0}[x.FTR]
            R[x.HomeTeam] = rh + K * (s - eh)
            R[x.AwayTeam] = ra + K * ((1 - s) - (1 - eh))
        hist = pd.DataFrame(rows, columns=["EloH", "EloA"])

    d = pd.concat([m[["Date", "Div", "FTR"]].reset_index(drop=True), hist], axis=1)
    d["gap"] = d.EloH + 60 - d.EloA
    d["pD_model"] = d.gap.map(draw_prob)
    d["isD"] = (d.FTR == "D").astype(int)

    # 只用有足够历史的样本（去掉每队最初若干场，Elo 还没稳定）
    d = d.iloc[20000:].reset_index(drop=True)
    print(f"评估样本（去掉前 2 万场热身）：{len(d):,} 场，实际平局率 {d.isD.mean():.2%}\n")

    # ---------- Q1 校准 ----------
    print("=" * 72)
    print("Q1  平局概率校准：模型说的 vs 实际发生的")
    print("=" * 72)
    bins = [-9999, -300, -200, -120, -60, 0, 60, 120, 200, 300, 9999]
    d["bkt"] = pd.cut(d.gap, bins)
    g = d.groupby("bkt", observed=True).agg(场次=("isD", "size"), 实际平局率=("isD", "mean"),
                                            模型平局率=("pD_model", "mean"))
    g["偏差"] = g.实际平局率 - g.模型平局率
    print(f"{'Elo差区间':<18}{'场次':>8}{'模型':>9}{'实际':>9}{'偏差':>9}")
    print("-" * 72)
    for k, v in g.iterrows():
        flag = "  ⚠" if abs(v.偏差) > 0.015 else ""
        print(f"{str(k):<18}{int(v.场次):>8,}{v.模型平局率:>9.1%}{v.实际平局率:>9.1%}{v.偏差:>+9.2%}{flag}")
    print("-" * 72)
    print(f"{'合计':<18}{len(d):>8,}{d.pD_model.mean():>9.1%}{d.isD.mean():>9.1%}"
          f"{d.isD.mean()-d.pD_model.mean():>+9.2%}")
    print(f"\nDRAW_FIT 参数 a={DRAW_FIT[0]:.3e} b={DRAW_FIT[1]:.3e} c={DRAW_FIT[2]:.4f}")

    # ---------- Q2 从不选平的代价 ----------
    print("\n" + "=" * 72)
    print("Q2  「从不选平」的代价：argmax 策略 vs 各种替代策略")
    print("=" * 72)
    P = np.array([[p["H"], p["D"], p["A"]] for p in
                  (elo_1x2(rh, ra) for rh, ra in zip(d.EloH, d.EloA))])
    y = d.FTR.map({"H": 0, "D": 1, "A": 2}).values

    def score(pred, name):
        hit = (pred == y).mean()
        n_d = (pred == 1).mean()
        print(f"  {name:<34}命中率 {hit:>7.2%}   选平比例 {n_d:>6.1%}")
        return hit

    base = score(P.argmax(1), "A 纯 argmax（现行）")
    # B: 势均力敌时强制选平
    for th in (0.03, 0.05, 0.08):
        pred = P.argmax(1).copy()
        even = np.abs(P[:, 0] - P[:, 2]) < th
        pred[even] = 1
        score(pred, f"B 主客概率差<{th:.0%} 时改选平")
    # C: 平局概率超过阈值就选平
    for th in (0.27, 0.28, 0.29):
        pred = P.argmax(1).copy()
        pred[P[:, 1] > th] = 1
        score(pred, f"C 平局概率>{th:.0%} 时改选平")
    # D: 上界——总是选实际最高概率的那个（=A），和「全选主」对照
    score(np.zeros(len(y), int), "D 全部选主胜（对照）")
    print(f"\n  基准（纯 argmax）= {base:.2%}")

    # ---------- Q3 期望 vs 实际的失手结构 ----------
    print("\n" + "=" * 72)
    print("Q3  失手结构：错的时候，有多少是被平局打掉的？")
    print("=" * 72)
    pred = P.argmax(1)
    miss = pred != y
    miss_d = miss & (y == 1)
    print(f"  总失手率            {miss.mean():.2%}")
    print(f"  其中实际是平局      {miss_d.sum()/miss.sum():.1%}  （占全部失手）")
    print(f"  平局占全部比赛      {(y==1).mean():.1%}")
    print(f"\n  → 理论基线：14 场里平均 {14*(y==1).mean():.1f} 场平局，"
          f"因从不选平，这些全是失手")
    print(f"  → 即：每期约 {14*(y==1).mean():.1f} 场的失手是「结构性」的，不是模型判断错误")

    # ---------- 实战对照 ----------
    print("\n" + "=" * 72)
    print("实战对照：两期真实开奖")
    print("=" * 72)
    real = {"26101": "03003133010300", "26104": "10130013130333"}
    tot_d = tot_n = 0
    for k, v in real.items():
        nd = v.count("1"); tot_d += nd; tot_n += 14
        print(f"  {k}  平局 {nd}/14 = {nd/14:.1%}")
    exp_d = (y == 1).mean()
    print(f"  合计  平局 {tot_d}/{tot_n} = {tot_d/tot_n:.1%}   长期基线 {exp_d:.1%}")
    z = (tot_d/tot_n - exp_d) / np.sqrt(exp_d*(1-exp_d)/tot_n)
    print(f"  z = {z:+.2f}  →  {'显著偏多' if z>1.96 else '显著偏少' if z<-1.96 else '与基线无显著差异'}")


if __name__ == "__main__":
    main()
