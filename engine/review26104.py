"""第 26104 期复盘：胜负彩 14 场 + 任选九 + 16 元复式"""
import sys, json
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE

ZH = {"3": "胜", "1": "平", "0": "负"}
PERIOD = "26104"
RESULT = "10130013130333"       # 2026-08-15 开奖
PRIZE = {"一等奖": (11, 460367), "二等奖": (239, 9080), "任九": (2851, 3251)}


def main():
    fp = BASE / "batches" / f"sfc_{PERIOD}_LOCK.json"
    L = json.loads(fp.read_text(encoding="utf-8"))
    R = L["任选九"]; F = R["复式"]
    r9_set = set(R["场次"])
    fu = {d["场次"]: d for d in F["明细"]}

    rows, s_hit = [], 0
    for i, g in enumerate(L["逐场"]):
        act = RESULT[i]
        ok = (g["单选"] == act)
        s_hit += ok
        g["实际"] = act; g["单式命中"] = bool(ok)
        rows.append({"场次": g["场次"], "对阵": g["对阵"], "预测": ZH[g["单选"]],
                     "置信": g["置信"], "实际": ZH[act], "命中": ok,
                     "入选九": g["场次"] in r9_set})

    # 任选九
    r9 = [r for r in rows if r["入选九"]]
    r9_hit = sum(r["命中"] for r in r9)
    # 复式（覆盖即算中）
    fu_hit = 0
    for r in r9:
        sel = fu[r["场次"]]["选项"]
        act = RESULT[r["场次"] - 1]
        if act in sel: fu_hit += 1

    S = L["单式"]
    print("=" * 74)
    print(f"  第 {PERIOD} 期复盘 · 锁定于 {L['锁定时间']}（赛前，截止 {L['截止']}）")
    print("=" * 74)
    print(f"\n开奖：{RESULT}")
    print(f"预测：{S['串']}\n")
    print(f"{'场':>3} {'对阵':<24}{'预测':>5}{'置信':>7}{'实际':>5}{'单式':>5}  {'任九':>5}{'复式选':>7}{'复式':>5}")
    print("-" * 74)
    for r in rows:
        no = r["场次"]
        if r["入选九"]:
            d = fu[no]; act = RESULT[no - 1]
            fok = act in d["选项"]
            tail = f"{'✓':>5}{d['选项']:>7}{'✓' if fok else '✗':>5}"
        else:
            tail = f"{'—':>5}{'':>7}{'':>5}"
        mark = "  ← 高置信翻车" if (not r["命中"] and r["置信"] >= 0.55) else ""
        print(f"{no:>3} {r['对阵']:<24}{r['预测']:>5}{r['置信']:>7.0%}{r['实际']:>5}"
              f"{'✓' if r['命中'] else '✗':>5}  {tail}{mark}")
    print("-" * 74)

    exp14, exp9, expf = S["期望命中"], R["期望命中"], F["期望命中"]
    print(f"\n【核心结果】")
    print(f"  胜负彩14场  实际 {s_hit}/14 = {s_hit/14:.1%}   预测 {S['预测胜率']:.1%}（期望 {exp14}）  偏差 {s_hit-exp14:+.2f}")
    print(f"  任选九单式  实际 {r9_hit}/9 = {r9_hit/9:.1%}    预测 {R['预测胜率']:.1%}（期望 {exp9}）  偏差 {r9_hit-exp9:+.2f}")
    print(f"  任选九复式  实际 {fu_hit}/9 = {fu_hit/9:.1%}    覆盖 {F['覆盖胜率']:.1%}（期望 {expf}）  偏差 {fu_hit-expf:+.2f}")
    print(f"  加保收益：复式比单式多中 {fu_hit-r9_hit} 场")

    # 是否中奖
    print(f"\n【是否中奖】任选九需 9 场全中")
    print(f"  单式 {r9_hit}/9 → {'🎉 中奖' if r9_hit==9 else f'未中（差 {9-r9_hit} 场）'}")
    print(f"  复式 {fu_hit}/9 → {'🎉 中奖' if fu_hit==9 else f'未中（差 {9-fu_hit} 场）'}")
    n, p = PRIZE["任九"]
    print(f"  本期任九开出 {n:,} 注，单注 {p:,} 元（若中，16 元成本 → 税后约 {p:,} 元）")

    # 归因
    miss = [r for r in rows if not r["命中"]]
    draws = [r for r in miss if r["实际"] == "平"]
    print(f"\n【失手归因】14 场错 {len(miss)} 场")
    print(f"  被平局打掉 {len(draws)} 场：" + "、".join(f"第{r['场次']}场" for r in draws))
    print(f"  本期实际平局数：{RESULT.count('1')} 场（14 场中占 {RESULT.count('1')/14:.0%}）")
    hi = [r for r in miss if r["置信"] >= 0.55]
    if hi:
        print(f"  高置信(≥55%)翻车 {len(hi)} 场：")
        for r in hi: print(f"    第{r['场次']}场 {r['对阵']} 判{r['预测']}({r['置信']:.0%}) → 实际{r['实际']}")

    L["实际开奖"] = RESULT
    L["复盘"] = {"单式命中": s_hit, "任九命中": r9_hit, "任九复式命中": fu_hit,
                "单式胜率": round(s_hit/14, 4), "任九胜率": round(r9_hit/9, 4),
                "期望14": exp14, "期望9": exp9, "期望复式": expf,
                "偏差14": round(s_hit-exp14, 2), "偏差9": round(r9_hit-exp9, 2),
                "加保收益": fu_hit-r9_hit, "中奖": bool(fu_hit == 9),
                "实际平局数": RESULT.count("1"),
                "失手": len(miss), "被平局打掉": len(draws), "逐场": rows,
                "奖金": {k: {"注数": v[0], "单注": v[1]} for k, v in PRIZE.items()}}
    L["已复盘"] = True
    fp.write_text(json.dumps(L, ensure_ascii=False, indent=1), encoding="utf-8")

    # 台账
    lp = BASE / "batches" / "SFC_LEDGER.json"
    led = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else []
    led = [x for x in led if x["期号"] != PERIOD]
    led.append({"期号": PERIOD, "单式命中": s_hit, "任九命中": r9_hit, "复式命中": fu_hit,
                "预测胜率": S["预测胜率"], "实际胜率": round(s_hit/14, 4),
                "期望": exp14, "偏差": round(s_hit-exp14, 2)})
    led.sort(key=lambda x: x["期号"])
    lp.write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding="utf-8")

    tot_h = sum(x["单式命中"] for x in led); tot_n = len(led)*14
    print(f"\n【累积台账】{len(led)} 期 · {tot_n} 场 · 累计命中 {tot_h} = {tot_h/tot_n:.1%}")
    for x in led:
        print(f"   {x['期号']}  预测 {x['预测胜率']:.1%} → 实际 {x['实际胜率']:.1%}  偏差 {x['偏差']:+.2f}")
    print(f"\n✅ 已写入 {fp.name} 与 SFC_LEDGER.json")
    return L


if __name__ == "__main__":
    main()
