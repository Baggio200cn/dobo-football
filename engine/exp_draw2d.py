"""v15：二维平局模型 —— 平局概率 = f(实力差, 进球倾向)

现行 draw_prob 只吃 Elo 差，看不见「这场是不是低进球对局」。
v14 审计证明该维度真实存在（势均力敌场次内相差 3.0pp）。

本脚本：
  1. 在训练期拟合二维模型（无前视：进球倾向只用该场之前的历史）
  2. 在测试期比较 一维 vs 二维 的 Brier / LogLoss
  3. 检查二维模型能否让「平局」成为某些场次的最高概率
"""
import sys
from collections import defaultdict, deque
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from data import load_matches
from model import draw_prob

OUT = ["H", "D", "A"]
ONEHOT = {"H": np.array([1., 0, 0]), "D": np.array([0, 1., 0]), "A": np.array([0, 0, 1.])}


def build():
    """滚动生成 (gap, tot, FTR)。tot = 两队近 20 场场均总进球（各自视角）。"""
    m = load_matches().sort_values("Date").reset_index(drop=True)
    gf = defaultdict(lambda: deque(maxlen=20))
    R, rows = {}, []
    K, HFA = 20.0, 60.0
    for x in m.itertuples():
        rh = R.get(x.HomeTeam, 1500.0); ra = R.get(x.AwayTeam, 1500.0)
        hh, aa = gf[x.HomeTeam], gf[x.AwayTeam]
        tot = (np.mean(hh) + np.mean(aa)) / 2 if len(hh) >= 10 and len(aa) >= 10 else np.nan
        rows.append((x.Date, rh + HFA - ra, tot, x.FTR))
        eh = 1 / (1 + 10 ** (-(rh + HFA - ra) / 400))
        s = {"H": 1.0, "D": 0.5, "A": 0.0}[x.FTR]
        R[x.HomeTeam] = rh + K * (s - eh)
        R[x.AwayTeam] = ra + K * ((1 - s) - (1 - eh))
        g = x.FTHG + x.FTAG
        hh.append(g); aa.append(g)
    return pd.DataFrame(rows, columns=["Date", "gap", "tot", "FTR"]).dropna()


def fit2d(tr):
    """logit(P平) ~ |gap| + gap² + tot   —— 用逻辑回归，参数少不易过拟合"""
    from sklearn.linear_model import LogisticRegression
    X = np.c_[np.abs(tr.gap), tr.gap ** 2, tr.tot]
    y = (tr.FTR == "D").astype(int).values
    lr = LogisticRegression(max_iter=1000).fit(X, y)
    return lr


def pd2d(lr, gap, tot):
    X = np.c_[np.abs(gap), np.asarray(gap) ** 2, tot]
    return lr.predict_proba(X)[:, 1]


def probs(gap, dprob):
    """给定 Elo 差与平局概率，拆出 H/D/A"""
    eh = 1 / (1 + 10 ** (-np.asarray(gap) / 400))
    return np.c_[eh * (1 - dprob), dprob, (1 - eh) * (1 - dprob)]


def main():
    d = build()
    d = d.iloc[20000:].reset_index(drop=True)
    cut = d.Date.max() - pd.Timedelta(days=365)
    tr, te = d[d.Date < cut], d[d.Date >= cut]
    print(f"训练 {len(tr):,} 场 · 测试 {len(te):,} 场（{te.Date.min():%Y-%m-%d}~{te.Date.max():%Y-%m-%d}）\n")

    lr = fit2d(tr)
    b = lr.coef_[0]
    print("二维平局模型（逻辑回归）系数：")
    print(f"  |Elo差|   {b[0]:+.5f}   （实力差越大，平局越少）")
    print(f"  Elo差²    {b[1]:+.3e}")
    print(f"  进球倾向  {b[2]:+.5f}   （场均总进球越高，平局越少）")
    print(f"  截距      {lr.intercept_[0]:+.4f}")

    Y = np.array([ONEHOT[v] for v in te.FTR])
    d1 = te.gap.map(draw_prob).values
    d2 = pd2d(lr, te.gap.values, te.tot.values)
    P1, P2 = probs(te.gap.values, d1), probs(te.gap.values, d2)

    def brier(P): return float(((P - Y) ** 2).sum(1).mean())
    def ll(P): return float(-np.log(np.clip((P * Y).sum(1), 1e-12, 1)).mean())
    def hit(P): return float((P.argmax(1) == Y.argmax(1)).mean())

    print("\n" + "=" * 68)
    print("一维（现行） vs 二维（加进球倾向）")
    print("=" * 68)
    print(f"{'模型':<26}{'Brier↓':>10}{'LogLoss↓':>11}{'命中率':>10}{'选平次数':>10}")
    print("-" * 68)
    print(f"{'一维 draw_prob(gap)':<26}{brier(P1):>10.4f}{ll(P1):>11.4f}"
          f"{hit(P1):>10.2%}{(P1.argmax(1)==1).sum():>10,}")
    print(f"{'二维 f(gap, 进球倾向)':<26}{brier(P2):>10.4f}{ll(P2):>11.4f}"
          f"{hit(P2):>10.2%}{(P2.argmax(1)==1).sum():>10,}")
    print("-" * 68)
    print(f"{'改善':<26}{brier(P1)-brier(P2):>+10.4f}{ll(P1)-ll(P2):>+11.4f}"
          f"{hit(P2)-hit(P1):>+10.2%}")

    print(f"\n平局概率范围：一维 {d1.min():.1%}~{d1.max():.1%}   二维 {d2.min():.1%}~{d2.max():.1%}")
    nfav = (P2.argmax(1) == 1).sum()
    print(f"二维模型把平局定为最高概率：{nfav:,} 场 / {len(te):,} = {nfav/len(te):.2%}")
    if nfav:
        acc = (te.FTR.values[P2.argmax(1) == 1] == "D").mean()
        print(f"  这些场次实际开出平局：{acc:.1%}")
    else:
        print("  → 即便加了进球维度，平局仍然一次都当不上最高概率。")
        print(f"  → 二维模型的平局上限 {d2.max():.2%}，仍低于均势时主/客各 {(1-d2.max())/2:.2%}")

    # 校准检查
    print("\n" + "=" * 68)
    print("平局概率校准（测试期）")
    print("=" * 68)
    for nm, dd in [("一维", d1), ("二维", d2)]:
        q = pd.qcut(dd, 6, duplicates="drop")
        t2 = pd.DataFrame({"p": dd, "y": (te.FTR == "D").astype(int).values, "q": q})
        g = t2.groupby("q", observed=True).agg(n=("y", "size"), 模型=("p", "mean"), 实际=("y", "mean"))
        mae = float((g.模型 - g.实际).abs().mean())
        print(f"  {nm}模型 分组平均绝对偏差 MAE = {mae:.4f}")


if __name__ == "__main__":
    main()
