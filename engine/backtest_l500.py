"""扩展联赛库的验收回测

这是「扩建联赛数据库」项目的**唯一验收标准**。

为什么不能用开奖结果验收：
  一个真实水平 50% 的数据源，有 21% 的概率中 9/14。
  要区分 55% 与 50% 需 1,563 场 ≈ 112 期 ≈ 45 周。
  42 场彩票样本没有任何鉴别力。

回测设计（与主库同一套铁律）
  · 按时间切分，绝不随机划分（防前视泄漏）
  · 指标 Brier / LogLoss，越低越好；衡量概率准不准，不是猜对与否
  · 基准是市场（500 平均欧赔去抽水），这是极强的对手

三个对照
  A 市场基准          纯赔率隐含概率
  B Elo500 单独       只用新建的扩展 Elo 池
  C 融合              0.55 市场 + 0.45 Elo500

验收线：C 必须**不差于** A。
  若 C 比 A 差，说明新数据在这些联赛上贡献的是噪声，
  应维持「双方无实测则模型弃权」的现行策略，不要强行让它投票。
"""
import sys, json
from collections import defaultdict, deque
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
from model import draw_prob2, implied_1x2

K, HFA = 20.0, 60.0
W_MKT = 0.55
OUT = ["H", "D", "A"]
Y = {"H": np.array([1., 0, 0]), "D": np.array([0, 1., 0]), "A": np.array([0, 0, 1.])}


def rolling(rows):
    """滚动生成每场的赛前 Elo 与进球倾向（无前视）"""
    R, gf = {}, defaultdict(lambda: deque(maxlen=20))
    cnt = defaultdict(int)
    out = []
    for m in sorted(rows, key=lambda x: (x["日期"] or "", x["fid"])):
        if m["主进"] is None: continue
        h, a = str(m["主ID"]), str(m["客ID"])
        rh, ra = R.get(h, 1500.0), R.get(a, 1500.0)
        gt = None
        if len(gf[h]) >= 5 and len(gf[a]) >= 5:
            gt = (float(np.mean(gf[h])) + float(np.mean(gf[a]))) / 2
        ftr = "H" if m["主进"] > m["客进"] else ("D" if m["主进"] == m["客进"] else "A")
        out.append({"日期": m["日期"], "联赛": m["联赛"], "gap": rh + HFA - ra, "gt": gt,
                    "ftr": ftr, "赔": (m["胜赔"], m["平赔"], m["负赔"]),
                    "n_h": cnt[h], "n_a": cnt[a]})
        eh = 1 / (1 + 10 ** (-(rh + HFA - ra) / 400))
        s = {"H": 1.0, "D": 0.5, "A": 0.0}[ftr]
        R[h] = rh + K * (s - eh); R[a] = ra + K * ((1 - s) - (1 - eh))
        cnt[h] += 1; cnt[a] += 1
        g = m["主进"] + m["客进"]; gf[h].append(g); gf[a].append(g)
    return out


def elo_p(gap, gt):
    d = draw_prob2(gap, gt)
    eh = 1 / (1 + 10 ** (-gap / 400))
    return np.array([eh * (1 - d), d, (1 - eh) * (1 - d)])


def score(P, Yv, name):
    P = np.clip(np.array(P), 1e-9, 1); P = P / P.sum(1, keepdims=True)
    br = float(((P - Yv) ** 2).sum(1).mean())
    ll = float(-np.log(np.clip((P * Yv).sum(1), 1e-12, 1)).mean())
    hit = float((P.argmax(1) == Yv.argmax(1)).mean())
    return br, ll, hit


def main():
    fp = BASE / "batches" / "L500_MATCHES.json"
    rows = json.loads(fp.read_text(encoding="utf-8"))
    ev = rolling(rows)
    # 只评估「双方都已积累 ≥10 场」的比赛 —— Elo 未收敛的场次没有评估意义
    ev = [e for e in ev if e["n_h"] >= 10 and e["n_a"] >= 10
          and all(x for x in e["赔"]) ]
    print(f"总库 {len(rows):,} 场 → 可评估 {len(ev):,} 场（双方均已积累 ≥10 场且有完整赔率）")
    if len(ev) < 100:
        print("\n⛔ 样本不足 100 场，回测无意义。等抓取完成后再跑。")
        by = defaultdict(int)
        for r in rows: by[r["联赛"]] += 1
        print("   当前各联赛场次：", dict(sorted(by.items(), key=lambda x: -x[1])))
        return

    ev.sort(key=lambda x: x["日期"])
    cut = int(len(ev) * 0.6)
    te = ev[cut:]
    print(f"按时间切分：训练前 {cut} 场（仅用于 Elo 预热）· 测试后 {len(te)} 场"
          f"（{te[0]['日期']} ~ {te[-1]['日期']}）\n")

    Yv = np.array([Y[e["ftr"]] for e in te])
    A = [list(implied_1x2(*e["赔"]).values()) for e in te]
    A = [[p[0], p[1], p[2]] for p in [[d["H"], d["D"], d["A"]] for d in
         [implied_1x2(*e["赔"]) for e in te]]]
    B = [elo_p(e["gap"], e["gt"]) for e in te]
    C = [W_MKT * np.array(a) + (1 - W_MKT) * np.array(b) for a, b in zip(A, B)]

    print("=" * 74)
    print("  扩展联赛库验收回测")
    print("=" * 74)
    print(f"  {'方案':<22}{'Brier↓':>10}{'LogLoss↓':>11}{'命中率':>10}")
    print("  " + "-" * 52)
    res = {}
    for nm, P in [("A 市场基准", A), ("B Elo500 单独", B), ("C 融合 0.55市场", C)]:
        br, ll, hit = score(P, Yv, nm)
        res[nm] = br
        print(f"  {nm:<22}{br:>10.4f}{ll:>11.4f}{hit:>10.2%}")
    print("  " + "-" * 52)

    # ⚠ 显著性检验：小样本下 Brier 差异极易被噪声主导。
    #   首次跑出 -0.0061 时我差点判「验收不通过」，做完检验才发现 SE=0.0071
    #   比差异本身还大，p=0.396。没有这一步就会得出错误结论。
    from scipy import stats
    Pa = np.clip(np.array(A), 1e-9, 1); Pa = Pa / Pa.sum(1, keepdims=True)
    Pc = np.clip(np.array(C), 1e-9, 1); Pc = Pc / Pc.sum(1, keepdims=True)
    ba = ((Pa - Yv) ** 2).sum(1); bc = ((Pc - Yv) ** 2).sum(1)
    dd = ba - bc                       # 正 = 融合更好
    n = len(te)
    se = dd.std(ddof=1) / np.sqrt(n)
    t, p = stats.ttest_rel(ba, bc)
    lo, hi = dd.mean() - 1.96 * se, dd.mean() + 1.96 * se
    need = (dd.std(ddof=1) / 0.005) ** 2

    print(f"\n  融合 vs 市场：Brier {dd.mean():+.4f}（正 = 融合更好）")
    print(f"    标准误 {se:.4f} · 95%CI [{lo:+.4f}, {hi:+.4f}] · t={t:.2f} p={p:.3f}")
    print(f"    要检出 0.005 的差异需约 {need:,.0f} 场，现有 {n} 场")

    if p > 0.05:
        print(f"\n  ⏸ 结论：**样本不足，无法判定**（p={p:.3f}）。")
        print(f"     这个 {abs(dd.mean()):.4f} 的差异在噪声范围内，不能据此判通过或不通过。")
        print(f"     继续扩充数据到 {need:,.0f} 场以上再评。")
    elif dd.mean() > 0:
        print(f"\n  ✅ 验收通过：融合显著优于市场基准，可接入预测。")
    else:
        print(f"\n  ❌ 验收不通过：新库显著贡献噪声。")
        print(f"     维持现行策略（双方无实测则模型弃权），不要强行让它投票。")

    # 分联赛看
    print(f"\n  {'联赛':<12}{'场次':>6}{'市场Brier':>11}{'融合Brier':>11}{'差值':>9}")
    byl = defaultdict(list)
    for i, e in enumerate(te): byl[e["联赛"]].append(i)
    for l, ix in sorted(byl.items(), key=lambda x: -len(x[1])):
        if len(ix) < 30: continue
        yv = Yv[ix]
        ba = score([A[i] for i in ix], yv, l)[0]
        bc = score([C[i] for i in ix], yv, l)[0]
        print(f"  {l:<12}{len(ix):>6}{ba:>11.4f}{bc:>11.4f}{ba-bc:>+9.4f}")


if __name__ == "__main__":
    main()
