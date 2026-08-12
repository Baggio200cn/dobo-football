"""锁定第 26104 期：胜负彩 14 场 + 任选九场（含 16 元复式方案）"""
import sys, json, itertools, datetime as dt
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
import sfc_lock as SL

CODE = {"H": "3", "D": "1", "A": "0"}
ZH = {"H": "胜", "D": "平", "A": "负"}
BUDGET = 16          # 元
PERIOD = "26104"


def main():
    s = json.loads((BASE / "batches" / f"sfc_{PERIOD}.json").read_text(encoding="utf-8"))
    P = s["picks"]
    M = []
    for p in P:
        pr = sorted([(p["胜"], "H"), (p["平"], "D"), (p["负"], "A")], reverse=True)
        M.append({"no": p["场次"], "nm": f"{p['主队']}vs{p['客队']}", "pr": pr, "c": p["置信"]})
    conf = np.array([p["置信"] for p in P])

    # ===== 胜负彩 14 场 =====
    dist14 = SL.hit_distribution(conf)
    single14 = "".join(p["推荐码"] for p in P)
    p14 = float(np.prod(conf))

    # ===== 任选九：按置信取前 9 =====
    order = sorted(range(14), key=lambda i: -conf[i])
    top9 = sorted(order[:9])            # 按场次排序便于填单
    drop = sorted(order[9:])
    c9 = np.array([conf[i] for i in top9])
    p9 = float(np.prod(c9))
    dist9 = SL.hit_distribution(c9)

    # ===== 任选九 16 元复式（≤8 注）=====
    maxbets = BUDGET // 2
    best = None
    for nd in range(0, 4):
        if 2 ** nd > maxbets: break
        for c in itertools.combinations(range(9), nd):
            ks = [1] * 9
            for i in c: ks[i] = 2
            pr = float(np.prod([sum(M[top9[j]]["pr"][t][0] for t in range(ks[j])) for j in range(9)]))
            if best is None or pr > best[0]: best = (pr, 2 ** nd, ks)
    pr_fu, bets, ks = best

    detail, line = [], []
    for j, i in enumerate(top9):
        k = ks[j]
        sel = "".join(sorted(CODE[M[i]["pr"][t][1]] for t in range(k)))
        cov = sum(M[i]["pr"][t][0] for t in range(k))
        zh = "+".join(ZH[M[i]["pr"][t][1]] for t in range(k))
        line.append(sel)
        detail.append({"场次": M[i]["no"], "对阵": M[i]["nm"], "选项": sel, "中文": zh,
                       "选项数": k, "覆盖概率": round(cov, 4), "单选置信": round(conf[i], 4)})
    cov9 = np.array([d["覆盖概率"] for d in detail])

    lock = {
        "期号": PERIOD, "截止": s["截止"], "锁定时间": str(dt.datetime.now())[:19],
        "玩法建议": "任选九场",
        "训练库": {"场次": s["训练库场次"], "联赛": s["训练库联赛"]},
        "单式": {"串": single14, "期望命中": round(float(conf.sum()), 2),
                "预测胜率": round(float(conf.mean()), 4),
                "分布": [round(float(x), 5) for x in dist14],
                "累计": [round(float(x), 5) for x in dist14[::-1].cumsum()[::-1]],
                "最可能命中": int(dist14.argmax()),
                "全中概率": p14},
        "任选九": {
            "场次": [M[i]["no"] for i in top9],
            "放弃": [M[i]["no"] for i in drop],
            "单式串": "".join(CODE[M[i]["pr"][0][1]] for i in top9),
            "预测胜率": round(float(c9.mean()), 4),
            "期望命中": round(float(c9.sum()), 2),
            "全中概率": p9,
            "较14场容易": round(p9 / p14, 1),
            "分布": [round(float(x), 5) for x in dist9],
            "复式": {"注数": bets, "金额": bets * 2, "串": line, "明细": detail,
                    "全中概率": pr_fu, "覆盖胜率": round(float(cov9.mean()), 4),
                    "期望命中": round(float(cov9.sum()), 2),
                    "加保场": [d["场次"] for d in detail if d["选项数"] > 1],
                    "较单式提升": round(pr_fu / p9, 2)},
        },
        "逐场": [{"场次": p["场次"], "赛事": p["赛事"], "对阵": f"{p['主队']}vs{p['客队']}",
                "胜": p["胜"], "平": p["平"], "负": p["负"], "单选": p["推荐码"],
                "置信": p["置信"], "Elo差": p["Elo差"], "来源": p["来源"],
                "入选九场": p["场次"] in [M[i]["no"] for i in top9],
                "实际": None, "单式命中": None} for p in P],
        "已复盘": False,
    }
    fp = BASE / "batches" / f"sfc_{PERIOD}_LOCK.json"
    fp.write_text(json.dumps(lock, ensure_ascii=False, indent=1), encoding="utf-8")

    R = lock["任选九"]; F = R["复式"]
    print("=" * 68)
    print(f"  第 {PERIOD} 期已锁定  {lock['锁定时间']}  （截止 {s['截止']}）")
    print("=" * 68)
    print(f"\n【玩法】任选九场 —— 比胜负彩 14 场容易 {R['较14场容易']:,.0f} 倍")
    print(f"  入选场次：{R['场次']}")
    print(f"  放弃场次：{R['放弃']}（把握最低的 5 场）")
    print(f"  预测胜率 {R['预测胜率']:.1%} · 期望命中 {R['期望命中']}/9 · 全中概率 1/{1/R['全中概率']:,.0f}")
    print(f"\n【16 元复式】{F['注数']} 注 · {F['金额']} 元 · 加保第 {F['加保场']} 场")
    print(f"  {'场':>3} {'对阵':<26}{'选择':>8}{'覆盖':>8}")
    for d in F["明细"]:
        star = " ★加保" if d["选项数"] > 1 else ""
        print(f"  {d['场次']:>3} {d['对阵']:<26}{d['中文']:>8}{d['覆盖概率']:>8.0%}{star}")
    print(f"\n  覆盖胜率 {F['覆盖胜率']:.1%} · 期望命中 {F['期望命中']}/9")
    print(f"  全中概率 1/{1/F['全中概率']:,.0f}（比单式提升 {F['较单式提升']}×）")
    print(f"\n✅ {fp.name}")
    return lock


if __name__ == "__main__":
    main()
