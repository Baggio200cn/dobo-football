"""期次锁定：把本期预测完整封存，供赛后复盘

产出 batches/sfc_<期>_LOCK.json，含：
  · 14 场逐场概率与单选
  · 预测胜率 + 命中数分布
  · 16 元复式标准样板（8 注）
  · 时间戳（证明赛前生成）

用法：python sfc_lock.py
"""
import sys, json, itertools, datetime as dt
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE

SRC = BASE / "batches" / "sfc_26101.json"
BUDGET_YUAN = 16
CODE = {"H": "3", "D": "1", "A": "0"}


def hit_distribution(probs):
    """泊松二项：命中 k 场的概率分布"""
    d = np.zeros(len(probs) + 1); d[0] = 1.0
    for p in probs:
        nd = np.zeros_like(d)
        for k in range(len(d)):
            if d[k] == 0: continue
            nd[k] += d[k] * (1 - p)
            if k + 1 < len(d): nd[k + 1] += d[k] * p
        d = nd
    return d


def best_plan(M, max_bets):
    """穷举 ∏k ≤ max_bets 的最优复式"""
    n = len(M)
    def prob(ks): return float(np.prod([sum(M[i]["pr"][j][0] for j in range(ks[i])) for i in range(n)]))
    cands = []
    for nd in range(0, 5):
        if 2 ** nd > max_bets: break
        for c in itertools.combinations(range(n), nd):
            ks = [1] * n
            for i in c: ks[i] = 2
            cands.append((ks, 2 ** nd))
    for t in range(n):
        for m in range(0, 3):
            if 3 * 2 ** m > max_bets: break
            for c in itertools.combinations([x for x in range(n) if x != t], m):
                ks = [1] * n; ks[t] = 3
                for i in c: ks[i] = 2
                cands.append((ks, 3 * 2 ** m))
    best = max(((prob(k), b, k) for k, b in cands if b <= max_bets), key=lambda x: x[0])
    return best


def main():
    s = json.loads(SRC.read_text(encoding="utf-8"))
    P = s["picks"]
    M = []
    for p in P:
        pr = sorted([(p["胜"], "H"), (p["平"], "D"), (p["负"], "A")], reverse=True)
        M.append({"no": p["场次"], "name": f"{p['主队']}vs{p['客队']}", "pr": pr})

    conf = np.array([p["置信"] for p in P])
    dist = hit_distribution(conf)
    cum = dist[::-1].cumsum()[::-1]

    # 16 元复式
    max_bets = BUDGET_YUAN // 2
    pr_all, bets, ks = best_plan(M, max_bets)
    fu_line, fu_detail = [], []
    for i in range(14):
        k = ks[i]
        sel = "".join(sorted(CODE[M[i]["pr"][j][1]] for j in range(k)))
        cov = sum(M[i]["pr"][j][0] for j in range(k))
        fu_line.append(sel)
        fu_detail.append({"场次": M[i]["no"], "对阵": M[i]["name"], "选项": sel,
                          "选项数": k, "覆盖概率": round(cov, 4)})
    # 复式的期望命中（按覆盖概率）
    fu_conf = np.array([d["覆盖概率"] for d in fu_detail])
    fu_dist = hit_distribution(fu_conf)
    fu_cum = fu_dist[::-1].cumsum()[::-1]

    home = np.array([p["胜"] for p in P])
    lock = {
        "期号": s["期号"], "截止": s["截止"],
        "锁定时间": str(dt.datetime.now())[:19],
        "训练库": {"场次": s["训练库场次"], "联赛": s["训练库联赛"]},
        "单式": {
            "串": s["单式"],
            "期望命中": round(float(conf.sum()), 2),
            "预测胜率": round(float(conf.mean()), 4),
            "分布": [round(float(x), 5) for x in dist],
            "累计": [round(float(x), 5) for x in cum],
            "最可能命中": int(dist.argmax()),
            "中10场以上": round(float(cum[10]), 4),
        },
        "复式16元": {
            "注数": bets, "金额": bets * 2,
            "串": fu_line, "明细": fu_detail,
            "全中概率": pr_all,
            "期望命中": round(float(fu_conf.sum()), 2),
            "覆盖胜率": round(float(fu_conf.mean()), 4),
            "分布": [round(float(x), 5) for x in fu_dist],
            "累计": [round(float(x), 5) for x in fu_cum],
            "加保场": [d["场次"] for d in fu_detail if d["选项数"] > 1],
        },
        "基准": {
            "全买主胜": round(float(home.mean()), 4),
            "历史主胜率": 0.439,
            "随机": round(1 / 3, 4),
        },
        "逐场": [{"场次": p["场次"], "赛事": p["赛事"], "对阵": f"{p['主队']}vs{p['客队']}",
                 "胜": p["胜"], "平": p["平"], "负": p["负"],
                 "单选": p["推荐码"], "置信": p["置信"],
                 "复式选": fu_line[i], "Elo差": p["Elo差"], "来源": p["来源"],
                 "实际": None, "单式命中": None, "复式命中": None} for i, p in enumerate(P)],
        "已复盘": False,
    }
    fp = BASE / "batches" / f"sfc_{s['期号']}_LOCK.json"
    fp.write_text(json.dumps(lock, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=" * 62)
    print(f"  第 {s['期号']} 期 · 预测已锁定  {lock['锁定时间']}")
    print("=" * 62)
    print(f"\n【单式】{lock['单式']['串']}")
    print(f"  预测胜率 {lock['单式']['预测胜率']:.1%} · 期望命中 {lock['单式']['期望命中']}/14"
          f" · 最可能中 {lock['单式']['最可能命中']} 场")
    print(f"\n【16 元复式】{bets} 注 · {bets*2} 元 · 加保第 {lock['复式16元']['加保场']} 场")
    print(f"  " + "  ".join(f"{i+1}:{x}" for i, x in enumerate(fu_line)))
    print(f"  覆盖胜率 {lock['复式16元']['覆盖胜率']:.1%} · 期望命中 {lock['复式16元']['期望命中']}/14"
          f" · 全中概率 {pr_all:.3e}")
    print(f"\n【对照基准】全买主胜 {lock['基准']['全买主胜']:.1%} · 历史主胜 43.9% · 随机 33.3%")
    print(f"\n✅ 已锁定 {fp.name} —— 赛后运行 sfc_review.py 复盘")
    return lock


if __name__ == "__main__":
    main()
