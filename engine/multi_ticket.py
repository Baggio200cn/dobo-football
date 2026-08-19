"""多张单式 vs 单张复式：预算怎么花更好？

用户提案
    不买 1 张 16 元复式，改买 4 张 2 元单式（共 8 元）：
      第 1 张 = 模型正常输出
      第 2–4 张 = 在「大众最看好」的场次上反向

理论依据（本项目实测）
    置信 55–65% 区间：我们说 59.3%，实际只中 36.4%（偏差 −22.9%，n=11）
    → 最有把握的那一档恰恰最不准。用户直觉正指向此处。
    注意：该偏差 p=0.109 未达显著，样本仅 11 场，是线索不是定论。

两种反向方向（正好相反，都测）
    U 用户式：反「置信最高」的 k 场   —— 赌高置信档过度自信
    M 我先前：反「p2/p1 最高」的 k 场 —— 赌翻转代价最小

结构差异
    复式 8 注：覆盖 3 场加保的全部 2³ 组合，8 个方案彼此「相邻」
    4 张单式：4 个方案可以彼此拉开，覆盖更分散的情形
    每元覆盖的方案数相同（0.5 个/元），差别在**分布形状**。

⚠ 74 期样本下，9/9 中奖仅 0–3 次，任何 EV 比较都无统计效力。
  因此同时报告「最佳票命中数」这个高信息量指标。
"""
import sys
import numpy as np
from scipy import stats
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import contrarian_r9 as C

CODE = C.CODE


def tickets(P, idx9, mode, n_flip_list):
    """生成多张单式。mode: 'U' 反置信最高 / 'M' 反 p2/p1 最高"""
    pr = {i: sorted(P[i].items(), key=lambda x: -x[1]) for i in idx9}
    if mode == "U":
        order = sorted(idx9, key=lambda i: -pr[i][0][1])          # 置信最高优先
    else:
        order = sorted(idx9, key=lambda i: -(pr[i][1][1] / pr[i][0][1]))
    out = []
    for k in n_flip_list:
        flip = set(order[:k])
        out.append({i: CODE[(pr[i][1][0] if i in flip else pr[i][0][0])] for i in idx9})
    return out


def eval_plan(D, build, cost):
    """build(P, idx9) → [票1, 票2, ...]；返回统计"""
    wins, prizes, best_hits = 0, [], []
    for d in D:
        idx9 = C.pick9(d["P"])
        ts = build(d["P"], idx9)
        hs = [sum(1 for i in idx9 if t[i] == d["res"][i]) for t in ts]
        best_hits.append(max(hs))
        if max(hs) == 9:
            wins += 1; prizes.append(d["任九单注"])
    pw = wins / len(D)
    avg = float(np.mean(prizes)) if prizes else 0.0
    return {"中奖期": wins, "P": pw, "均奖金": avg, "EV": pw * avg,
            "成本": cost, "回报": pw * avg / cost,
            "最佳命中均值": float(np.mean(best_hits)),
            "最佳命中分布": np.bincount(best_hits, minlength=10)[:10]}


def complex_plan(P, idx9, n_hedge):
    """复式：把加保 n 场展开成 2^n 张单式来评估（等价）"""
    pr = {i: sorted(P[i].items(), key=lambda x: -x[1]) for i in idx9}
    order = sorted(idx9, key=lambda i: -(pr[i][1][1] / pr[i][0][1]))
    hed = order[:n_hedge]
    base = {i: CODE[pr[i][0][0]] for i in idx9}
    out = []
    import itertools
    for combo in itertools.product(*[[0, 1]] * n_hedge):
        t = dict(base)
        for j, c in enumerate(combo):
            if c: t[hed[j]] = CODE[pr[hed[j]][1][0]]
        out.append(t)
    return out


def main():
    D = C.load()
    print("=" * 88)
    print(f"  多张单式 vs 单张复式 · 实测 {len(D)} 期")
    print("=" * 88)

    plans = [
        ("① 单式 1 张（纯模型）", lambda P, i: tickets(P, i, "U", [0]), 2),
        ("② 用户式 4 张：反置信最高 0/1/2/3", lambda P, i: tickets(P, i, "U", [0, 1, 2, 3]), 8),
        ("③ 我先前式 4 张：反 p2/p1 最高", lambda P, i: tickets(P, i, "M", [0, 1, 2, 3]), 8),
        ("④ 复式 8 注（加保 3 场）", lambda P, i: complex_plan(P, i, 3), 16),
        ("⑤ 复式 4 注（加保 2 场）", lambda P, i: complex_plan(P, i, 2), 8),
        ("⑥ 用户式 8 张（反 0–7）", lambda P, i: tickets(P, i, "U", list(range(8))), 16),
    ]
    print(f"\n  {'方案':<32}{'成本':>6}{'中奖期':>8}{'实测P':>9}{'均奖金':>10}"
          f"{'期望赔付':>10}{'回报率':>9}{'最佳命中':>9}")
    print("  " + "-" * 86)
    R = {}
    for name, f, cost in plans:
        r = eval_plan(D, f, cost); R[name] = r
        print(f"  {name:<32}{cost:>6}{r['中奖期']:>8}{r['P']:>9.2%}{r['均奖金']:>10,.0f}"
              f"{r['EV']:>10.2f}{r['回报']:>9.2f}×{r['最佳命中均值']:>9.2f}")
    print("  " + "-" * 86)
    print(f"  官方返奖率参考 0.65×")

    print("\n" + "=" * 88)
    print("  高信息量指标：最佳票命中数分布（比 9/9 有效样本多得多）")
    print("=" * 88)
    print(f"  {'方案':<32}" + "".join(f"{h}场".rjust(7) for h in range(4, 10)))
    print("  " + "-" * 76)
    for name, _, _ in plans:
        d = R[name]["最佳命中分布"]
        print(f"  {name:<32}" + "".join(f"{d[h]:>7}" for h in range(4, 10)))
    print("  " + "-" * 76)

    # 用户式 vs 复式 的配对检验（用最佳命中数，样本足够）
    print("\n" + "=" * 88)
    print("  统计检验：用户式 4 张（8元） vs 复式 8 注（16元）")
    print("=" * 88)
    a, b = [], []
    for d in D:
        i9 = C.pick9(d["P"])
        ta = tickets(d["P"], i9, "U", [0, 1, 2, 3])
        tb = complex_plan(d["P"], i9, 3)
        a.append(max(sum(1 for i in i9 if t[i] == d["res"][i]) for t in ta))
        b.append(max(sum(1 for i in i9 if t[i] == d["res"][i]) for t in tb))
    a, b = np.array(a, float), np.array(b, float)
    t, p = stats.ttest_rel(a, b)
    print(f"  最佳命中数：用户式 {a.mean():.3f}  vs  复式 {b.mean():.3f}   差 {a.mean()-b.mean():+.3f}")
    print(f"  配对 t 检验 t={t:.2f}  p={p:.3f}   n={len(D)}")
    print(f"  成本：8 元 vs 16 元（用户式**省一半**）")
    if p > 0.05:
        print(f"  → 命中能力**无显著差异**，但用户式成本只有一半。")
    elif a.mean() > b.mean():
        print(f"  → 用户式显著更优，且更便宜。")
    else:
        print(f"  → 复式显著更优，但贵一倍。")

    print("\n" + "=" * 88)
    print("  ⚠ 9/9 中奖在 74 期中仅出现 0–3 次，EV 与回报率**无统计效力**，仅供参考。")
    print("     返奖率 65%，长期期望为负。本测算为方法论研究，不构成投注建议。")
    print("=" * 88)
    return R


if __name__ == "__main__":
    main()
