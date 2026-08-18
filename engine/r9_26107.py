"""第 26107 期 · 任选九场 16 元复式 —— 正式出票并写入锁定档

为什么主推任选九而非胜负彩 14 场：
    同为 16 元 8 注，14 场全中概率 1/1,273，任选九 1/33，容易 39 倍。
    胜负彩不发「中 8 场」的奖，14 场那张票期望命中虽高但基本不可能中奖。

本期数据等级 C（纯市场）：14 场里 13 场双方球队不在我们数据库内，
    模型全程弃权、由市场独立定价。v18 回测已证明这是这些联赛上的最优策略
    （权重扫描最优点 w_市场 = 1.00），因此「纯市场」不是妥协而是正解。
    但该期仍单独记账，不并入主台账。
"""
import sys, json, itertools, datetime as dt
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
import sfc_lock as SL

PERIOD = "26107"
BUDGET = 16
CODE = {"胜": "3", "平": "1", "负": "0"}
ZH = {"3": "胜", "1": "平", "0": "负"}


def main():
    S = json.loads((BASE / "batches" / f"sfc_{PERIOD}.json").read_text(encoding="utf-8"))
    P = S["picks"]
    L = json.loads((BASE / "batches" / f"sfc_{PERIOD}_LOCK.json").read_text(encoding="utf-8"))
    conf = np.array([p["置信"] for p in P])

    M = []
    for p in P:
        pr = sorted([(p["胜"], "3"), (p["平"], "1"), (p["负"], "0")], reverse=True)
        M.append({"no": p["场次"], "赛事": p["赛事"],
                  "nm": f"{p['主队']} vs {p['客队']}", "pr": pr, "c": p["置信"],
                  "p": {"3": p["胜"], "1": p["平"], "0": p["负"]}})

    # 选 9 场：按置信度取前 9（等价于最大化 ∏p，也天然避开平局风险最高的场次）
    order = sorted(range(14), key=lambda i: -conf[i])
    top9 = sorted(order[:9]); drop = sorted(order[9:])
    c9 = np.array([conf[i] for i in top9])
    p9_single = float(np.prod(c9))

    # 16 元 = 8 注：穷举倍数向量，乘积 ≤ 8，最大化 ∏覆盖概率
    maxbets = BUDGET // 2
    best = None
    for nd in range(0, 4):
        if 2 ** nd > maxbets: break
        for c in itertools.combinations(range(9), nd):
            ks = [1] * 9
            for i in c: ks[i] = 2
            pr = float(np.prod([sum(M[top9[j]]["pr"][t][0] for t in range(ks[j]))
                                for j in range(9)]))
            if best is None or pr > best[0]: best = (pr, 2 ** nd, list(ks))
    pr_fu, bets, ks = best

    detail, line = [], []
    for j, i in enumerate(top9):
        k = ks[j]
        sel = "".join(sorted((M[i]["pr"][t][1] for t in range(k)), reverse=True))
        cov = sum(M[i]["pr"][t][0] for t in range(k))
        zh = "+".join(ZH[c] for c in sel)
        line.append(sel)
        detail.append({"场次": M[i]["no"], "赛事": M[i]["赛事"], "对阵": M[i]["nm"],
                       "选项": sel, "中文": zh, "选项数": k,
                       "覆盖概率": round(cov, 4), "单选置信": round(M[i]["c"], 4),
                       "胜": M[i]["p"]["3"], "平": M[i]["p"]["1"], "负": M[i]["p"]["0"]})
    cov9 = np.array([d["覆盖概率"] for d in detail])
    dist = SL.hit_distribution(cov9)

    R9 = {"玩法": "任选九场", "预算": BUDGET, "注数": bets, "金额": bets * 2,
          "场次": [M[i]["no"] for i in top9],
          "放弃": [M[i]["no"] for i in drop],
          "单式串": "".join(M[i]["pr"][0][1] for i in top9),
          "复式串": line, "明细": detail,
          "加保场": [d["场次"] for d in detail if d["选项数"] > 1],
          "单式全中概率": p9_single,
          "复式全中概率": pr_fu,
          "较单式提升": round(pr_fu / p9_single, 2),
          "覆盖胜率": round(float(cov9.mean()), 4),
          "期望命中": round(float(cov9.sum()), 2),
          "分布": [round(float(x), 5) for x in dist],
          "数据等级": L.get("数据等级", "C"), "单独记账": True,
          "出票时间": str(dt.datetime.now())[:19]}
    L["任选九_主推"] = R9
    L["玩法建议"] = "任选九场（主推）· 胜负彩14场（对照）"
    (BASE / "batches" / f"sfc_{PERIOD}_LOCK.json").write_text(
        json.dumps(L, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 打印 ----
    print("=" * 84)
    print(f"  胜负彩第 {PERIOD} 期 · 任选九场 · 16 元复式（{bets} 注）")
    print(f"  投注截止 {L['截止']}   出票 {R9['出票时间']}   数据等级 {R9['数据等级']}（单独记账）")
    print("=" * 84)
    print(f"\n  {'场':>3} {'赛事':<5}{'对阵':<30}{'胜':>6}{'平':>6}{'负':>6}{'填涂':>9}{'覆盖':>7}")
    print("  " + "-" * 76)
    for d in detail:
        star = " ★加保" if d["选项数"] > 1 else ""
        print(f"  {d['场次']:>3} {d['赛事']:<5}{d['对阵']:<30}"
              f"{d['胜']:>6.2f}{d['平']:>6.2f}{d['负']:>6.2f}"
              f"{d['中文']:>9}{d['覆盖概率']:>7.0%}{star}")
    print("  " + "-" * 76)
    print(f"\n  放弃 {R9['放弃']} 场（把握最低的 5 场）")

    print(f"\n  【填单速查】按场次顺序，只填这 9 场：")
    print(f"    " + "  ".join(f"{d['场次']}:{d['选项']}" for d in detail))
    print(f"    单式串（不加保时）{R9['单式串']}")

    print(f"\n  【{bets} 注全展开】")
    opts = [list(d["选项"]) for d in detail]
    for n, combo in enumerate(itertools.product(*opts), 1):
        print(f"    第{n} 注  " + " ".join(f"{d['场次']}={c}" for d, c in zip(detail, combo)))

    print(f"\n  【指标】")
    print(f"    覆盖胜率   {R9['覆盖胜率']:.1%}")
    print(f"    期望命中   {R9['期望命中']}/9")
    print(f"    全中概率   1/{1/pr_fu:,.0f}（单式 1/{1/p9_single:,.0f}，提升 {R9['较单式提升']}×）")
    print(f"    命中分布   " + " ".join(f"{i}场={dist[i]:.1%}" for i in range(5, 10)))
    print(f"    成本       {bets} 注 × 2 元 = {bets*2} 元")
    print(f"\n  ⚠ 任选九只设一等奖，需 9 场全中；中 8 场无奖。")
    print(f"  ⚠ 本站不提供购彩功能，以上为模型输出记录，不构成任何投注建议。")
    print(f"\n  ✅ 已写入 sfc_{PERIOD}_LOCK.json（字段 任选九_主推）")
    return R9


if __name__ == "__main__":
    main()
