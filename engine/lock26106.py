"""锁定第 26106 期：胜负彩 14 场全竞猜 + 16 元复式（8 注上限）

16 元 = 8 注（2 元/注）。在 14 场上，倍数向量各档乘积须 ≤ 8：
    · 0 个三选 + 0~3 个双选     （2^3 = 8）
    · 1 个三选 + 0~1 个双选     （3×2 = 6）
穷举全部方案，取「全中概率」最大者。
同时给出任选九对照（同为 16 元）。
"""
import sys, json, itertools, datetime as dt
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
import sfc_lock as SL

CODE = {"H": "3", "D": "1", "A": "0"}
ZH = {"H": "胜", "D": "平", "A": "负"}
PERIOD = "26106"
BUDGET = 16
MAXBETS = BUDGET // 2       # 8 注


def best14(M, maxbets=MAXBETS):
    """穷举 14 场复式方案，返回 (全中概率, 注数, 每场选项数)"""
    n = len(M)
    best = None

    def cover(ks):
        return float(np.prod([sum(M[i]["pr"][t][0] for t in range(ks[i])) for i in range(n)]))

    # case A: 只用双选
    for nd in range(0, 4):
        if 2 ** nd > maxbets: break
        for c in itertools.combinations(range(n), nd):
            ks = [1] * n
            for i in c: ks[i] = 2
            pr = cover(ks)
            if best is None or pr > best[0]: best = (pr, 2 ** nd, list(ks))
    # case B: 1 个三选 + 0~1 个双选
    for i3 in range(n):
        for nd in (0, 1):
            if 3 * (2 ** nd) > maxbets: break
            for c in itertools.combinations([x for x in range(n) if x != i3], nd):
                ks = [1] * n; ks[i3] = 3
                for i in c: ks[i] = 2
                pr = cover(ks)
                if best is None or pr > best[0]: best = (pr, 3 * (2 ** nd), list(ks))
    return best


def build_detail(M, ks, idx):
    detail, line = [], []
    for j in idx:
        k = ks[j]
        sel = "".join(sorted(CODE[M[j]["pr"][t][1]] for t in range(k)))
        cov = sum(M[j]["pr"][t][0] for t in range(k))
        zh = "+".join(ZH[M[j]["pr"][t][1]] for t in range(k))
        line.append(sel)
        detail.append({"场次": M[j]["no"], "对阵": M[j]["nm"], "选项": sel, "中文": zh,
                       "选项数": k, "覆盖概率": round(cov, 4),
                       "单选置信": round(M[j]["c"], 4)})
    return detail, line


def main():
    s = json.loads((BASE / "batches" / f"sfc_{PERIOD}.json").read_text(encoding="utf-8"))
    P = s["picks"]
    M = []
    for p in P:
        pr = sorted([(p["胜"], "H"), (p["平"], "D"), (p["负"], "A")], reverse=True)
        M.append({"no": p["场次"], "nm": f"{p['主队']}vs{p['客队']}", "pr": pr, "c": p["置信"]})
    conf = np.array([p["置信"] for p in P])

    # ===== 14 场单式 =====
    single14 = "".join(p["推荐码"] for p in P)
    dist14 = SL.hit_distribution(conf)
    p14 = float(np.prod(conf))

    # ===== 14 场 16 元复式 =====
    pr_fu, bets, ks = best14(M)
    detail, line = build_detail(M, ks, range(14))
    cov14 = np.array([d["覆盖概率"] for d in detail])
    distf = SL.hit_distribution(cov14)

    # ===== 任选九对照（同 16 元）=====
    order = sorted(range(14), key=lambda i: -conf[i])
    top9 = sorted(order[:9])
    c9 = np.array([conf[i] for i in top9])
    p9 = float(np.prod(c9))
    best9 = None
    for nd in range(0, 4):
        if 2 ** nd > MAXBETS: break
        for c in itertools.combinations(range(9), nd):
            k9 = [1] * 9
            for i in c: k9[i] = 2
            pr = float(np.prod([sum(M[top9[j]]["pr"][t][0] for t in range(k9[j])) for j in range(9)]))
            if best9 is None or pr > best9[0]: best9 = (pr, 2 ** nd, list(k9))
    pr9f, bets9, ks9 = best9
    ks9full = [1] * 14
    for j, i in enumerate(top9): ks9full[i] = ks9[j]
    d9, l9 = build_detail(M, ks9full, top9)
    cov9 = np.array([d["覆盖概率"] for d in d9])

    lock = {
        "期号": PERIOD, "截止": s["截止"], "锁定时间": str(dt.datetime.now())[:19],
        "玩法建议": "胜负彩14场（本期主推）",
        "训练库": {"场次": s["训练库场次"], "联赛": s["训练库联赛"]},
        "单式": {"串": single14, "期望命中": round(float(conf.sum()), 2),
                "预测胜率": round(float(conf.mean()), 4),
                "分布": [round(float(x), 5) for x in dist14],
                "累计": [round(float(x), 5) for x in dist14[::-1].cumsum()[::-1]],
                "最可能命中": int(dist14.argmax()),
                "全中概率": p14},
        "复式16元": {"注数": bets, "金额": bets * 2, "串": line, "明细": detail,
                   "全中概率": pr_fu, "覆盖胜率": round(float(cov14.mean()), 4),
                   "期望命中": round(float(cov14.sum()), 2),
                   "分布": [round(float(x), 5) for x in distf],
                   "累计": [round(float(x), 5) for x in distf[::-1].cumsum()[::-1]],
                   "加保场": [d["场次"] for d in detail if d["选项数"] > 1],
                   "较单式提升": round(pr_fu / p14, 2)},
        "任选九对照": {"场次": [M[i]["no"] for i in top9],
                   "放弃": [M[i]["no"] for i in sorted(order[9:])],
                   "单式串": "".join(CODE[M[i]["pr"][0][1]] for i in top9),
                   "预测胜率": round(float(c9.mean()), 4),
                   "期望命中": round(float(c9.sum()), 2), "全中概率": p9,
                   "复式": {"注数": bets9, "金额": bets9 * 2, "串": l9, "明细": d9,
                           "全中概率": pr9f, "覆盖胜率": round(float(cov9.mean()), 4),
                           "期望命中": round(float(cov9.sum()), 2),
                           "加保场": [d["场次"] for d in d9 if d["选项数"] > 1]}},
        "逐场": [{"场次": p["场次"], "赛事": p["赛事"], "对阵": f"{p['主队']}vs{p['客队']}",
                "胜": p["胜"], "平": p["平"], "负": p["负"], "单选": p["推荐码"],
                "置信": p["置信"], "Elo差": p["Elo差"], "来源": p["来源"],
                "入选九场": p["场次"] in [M[i]["no"] for i in top9],
                "实际": None, "单式命中": None} for p in P],
        "已复盘": False,
    }
    fp = BASE / "batches" / f"sfc_{PERIOD}_LOCK.json"
    fp.write_text(json.dumps(lock, ensure_ascii=False, indent=1), encoding="utf-8")

    F = lock["复式16元"]; R = lock["任选九对照"]
    print("=" * 78)
    print(f"  第 {PERIOD} 期已锁定  {lock['锁定时间']}  （截止 {s['截止']}）")
    print("=" * 78)
    print(f"\n【主推：胜负彩 14 场 · 16 元复式 {F['注数']} 注】加保第 {F['加保场']} 场")
    print(f"  {'场':>3} {'对阵':<26}{'单选':>5}{'复式选择':>10}{'覆盖':>8}")
    print("  " + "-" * 54)
    for d in F["明细"]:
        star = " ★" if d["选项数"] > 1 else ""
        one = ZH[max([("H", P[d['场次']-1]['胜']), ("D", P[d['场次']-1]['平']),
                      ("A", P[d['场次']-1]['负'])], key=lambda x: x[1])[0]]
        print(f"  {d['场次']:>3} {d['对阵']:<26}{one:>5}{d['中文']:>10}{d['覆盖概率']:>8.0%}{star}")
    print("  " + "-" * 54)
    print(f"\n  单式串   {lock['单式']['串']}")
    print(f"  复式串   {'-'.join(F['串'])}")
    print(f"  单式：预测胜率 {lock['单式']['预测胜率']:.1%} · 期望命中 {lock['单式']['期望命中']}/14"
          f" · 全中 1/{1/p14:,.0f}")
    print(f"  复式：覆盖胜率 {F['覆盖胜率']:.1%} · 期望命中 {F['期望命中']}/14"
          f" · 全中 1/{1/pr_fu:,.0f}（提升 {F['较单式提升']}×）")
    print(f"  最可能命中 {lock['单式']['最可能命中']} 场；命中 ≥12 场概率 {lock['单式']['累计'][12]:.2%}")
    print(f"\n【对照：任选九 · 同 16 元】入选 {R['场次']}")
    print(f"  复式覆盖胜率 {R['复式']['覆盖胜率']:.1%} · 期望命中 {R['复式']['期望命中']}/9"
          f" · 全中 1/{1/R['复式']['全中概率']:,.0f}")
    print(f"  → 任九全中难度是 14 场的 1/{pr9f/pr_fu:,.0f}")
    print(f"\n✅ {fp.name}")
    return lock


if __name__ == "__main__":
    main()
