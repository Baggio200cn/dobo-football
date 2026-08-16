"""v14 实验：实盘融合权重体检

实盘脚本（sfc26xxx.py）用的是 `W_MKT×市场 + (1-W_MKT)×Elo`，W_MKT 固定 0.55。
这个权重是早期在五大联赛上定的。现在训练库扩到 36 联赛、10.7 万场，
必须重新检验：0.55 还是最优吗？纯市场是不是更好？
"""
import sys
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from data import load_matches
from model import elo_1x2, implied_1x2

OUT = ["H", "D", "A"]
ONEHOT = {"H": np.array([1., 0, 0]), "D": np.array([0, 1., 0]), "A": np.array([0, 0, 1.])}


def main():
    m = load_matches().sort_values("Date").reset_index(drop=True)
    # 滚动 Elo，记录每场赛前评分（无前视）
    R, pre = {}, []
    K, HFA = 20.0, 60.0
    for x in m.itertuples():
        rh = R.get(x.HomeTeam, 1500.0); ra = R.get(x.AwayTeam, 1500.0)
        pre.append((rh, ra))
        eh = 1 / (1 + 10 ** (-(rh + HFA - ra) / 400))
        s = {"H": 1.0, "D": 0.5, "A": 0.0}[x.FTR]
        R[x.HomeTeam] = rh + K * (s - eh)
        R[x.AwayTeam] = ra + K * ((1 - s) - (1 - eh))
    m[["EloH", "EloA"]] = pre

    # 测试窗口：最近一年，且必须有赔率
    cut = m.Date.max() - pd.Timedelta(days=365)
    t = m[(m.Date >= cut) & m.B365H.notna() & m.B365D.notna() & m.B365A.notna()].copy()
    print(f"测试窗口 {t.Date.min():%Y-%m-%d} ~ {t.Date.max():%Y-%m-%d}   {len(t):,} 场"
          f" / {t.Div.nunique()} 联赛\n")

    E = np.array([[elo_1x2(a, b)[k] for k in OUT] for a, b in zip(t.EloH, t.EloA)])
    Mk = np.array([[implied_1x2(h, d, a)[k] for k in OUT]
                   for h, d, a in zip(t.B365H, t.B365D, t.B365A)])
    Y = np.array([ONEHOT[v] for v in t.FTR])

    def brier(P): return float(((P - Y) ** 2).sum(1).mean())
    def hit(P):   return float((P.argmax(1) == Y.argmax(1)).mean())

    print("=" * 62)
    print("融合权重扫描   P = w×市场 + (1-w)×Elo")
    print("=" * 62)
    print(f"{'w(市场)':>8}{'Brier↓':>10}{'命中率':>10}{'对比现行':>12}")
    print("-" * 62)
    rows = []
    for w in np.round(np.arange(0.0, 1.01, 0.05), 2):
        P = w * Mk + (1 - w) * E
        P = P / P.sum(1, keepdims=True)
        rows.append((w, brier(P), hit(P)))
    cur = [r for r in rows if abs(r[0] - 0.55) < 1e-9][0]
    for w, b, h in rows:
        tag = "  ← 现行" if abs(w - 0.55) < 1e-9 else ""
        star = " ★最优" if b == min(r[1] for r in rows) else ""
        print(f"{w:>8.2f}{b:>10.4f}{h:>10.2%}{b-cur[1]:>+12.4f}{tag}{star}")
    print("-" * 62)
    best = min(rows, key=lambda r: r[1])
    print(f"\n现行 w=0.55  Brier {cur[1]:.4f}  命中 {cur[2]:.2%}")
    print(f"最优 w={best[0]:.2f}  Brier {best[1]:.4f}  命中 {best[2]:.2%}"
          f"   可改善 {cur[1]-best[1]:.4f}")
    print(f"纯市场 w=1.00 Brier {rows[-1][1]:.4f}  命中 {rows[-1][2]:.2%}")
    print(f"纯 Elo w=0.00 Brier {rows[0][1]:.4f}  命中 {rows[0][2]:.2%}")

    # 按联赛档次拆分：胜负彩常用的中小联赛 vs 五大联赛
    print("\n" + "=" * 62)
    print("分层检验：五大联赛 vs 其他（胜负彩实际取材更偏后者）")
    print("=" * 62)
    BIG = {"E0", "SP1", "I1", "D1", "F1"}
    for name, mask in [("五大联赛", t.Div.isin(BIG).values),
                       ("其他 31 个联赛", (~t.Div.isin(BIG)).values)]:
        if mask.sum() < 200: continue
        def b_of(w):
            P = w * Mk + (1 - w) * E
            P = P / P.sum(1, keepdims=True)
            return float((((P - Y) ** 2).sum(1))[mask].mean())
        sub = [(w, b_of(w)) for w in np.round(np.arange(0, 1.01, 0.05), 2)]
        b55 = [x for x in sub if abs(x[0]-0.55) < 1e-9][0][1]
        bb = min(sub, key=lambda x: x[1])
        print(f"  {name:<16}{mask.sum():>7,} 场   w=0.55 → {b55:.4f}   "
              f"最优 w={bb[0]:.2f} → {bb[1]:.4f}   差 {b55-bb[1]:.4f}")


if __name__ == "__main__":
    main()
