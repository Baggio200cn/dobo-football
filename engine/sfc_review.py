"""胜负彩复盘：把锁定的预测与真实开奖对照

用法：python sfc_review.py 26101 03003133010300
     （第二个参数为 14 位开奖结果，胜=3 平=1 负=0）
"""
import sys, json
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE

ZH = {"3": "胜", "1": "平", "0": "负"}


def main(period, result):
    fp = BASE / "batches" / f"sfc_{period}_LOCK.json"
    L = json.loads(fp.read_text(encoding="utf-8"))
    assert len(result) == 14, "开奖结果必须 14 位"

    rows, s_hit, f_hit = [], 0, 0
    for i, g in enumerate(L["逐场"]):
        act = result[i]
        s_ok = (g["单选"] == act)
        f_sel = L["复式16元"]["串"][i]
        f_ok = (act in f_sel)
        s_hit += s_ok; f_hit += f_ok
        g["实际"] = act; g["单式命中"] = bool(s_ok); g["复式命中"] = bool(f_ok)
        rows.append({
            "场次": g["场次"], "对阵": g["对阵"],
            "预测": ZH[g["单选"]], "置信": g["置信"],
            "实际": ZH[act], "单式": "✓" if s_ok else "✗",
            "复式选": f_sel, "复式": "✓" if f_ok else "✗",
        })

    S, F = L["单式"], L["复式16元"]
    exp_s, exp_f = S["期望命中"], F["期望命中"]
    p_s = S["分布"][s_hit] if s_hit < len(S["分布"]) else 0
    cum_lo = sum(S["分布"][6:11])

    print("=" * 68)
    print(f"  第 {period} 期复盘 · 锁定于 {L['锁定时间']}（赛前）")
    print("=" * 68)
    print(f"\n开奖：{result}")
    print(f"预测：{S['串']}\n")
    print(f"{'场':>3} {'对阵':<22}{'预测':>5}{'置信':>7}{'实际':>5}{'单式':>5}{'复式选':>7}{'复式':>5}")
    print("-" * 68)
    for r in rows:
        mark = "  ← 高置信翻车" if (r["单式"] == "✗" and r["置信"] >= 0.6) else ""
        print(f"{r['场次']:>3} {r['对阵']:<22}{r['预测']:>5}{r['置信']:>7.0%}{r['实际']:>5}"
              f"{r['单式']:>5}{r['复式选']:>7}{r['复式']:>5}{mark}")
    print("-" * 68)

    print(f"\n【核心结果】")
    print(f"  单式  实际命中 {s_hit}/14 = {s_hit/14:.1%}   预测胜率 {S['预测胜率']:.1%}"
          f"（期望 {exp_s}）  偏差 {s_hit-exp_s:+.2f} 场")
    print(f"  复式  实际命中 {f_hit}/14 = {f_hit/14:.1%}   覆盖胜率 {F['覆盖胜率']:.1%}"
          f"（期望 {exp_f}）  偏差 {f_hit-exp_f:+.2f} 场")
    print(f"  加保收益：复式比单式多中 {f_hit-s_hit} 场")
    print(f"\n  命中 {s_hit} 场的先验概率 {p_s:.1%}"
          f" · 落在 6-10 场区间的先验概率 {cum_lo:.1%} → {'✅ 在预期区间内' if 6<=s_hit<=10 else '⚠️ 超出预期区间'}")

    # 归因
    fails = [r for r in rows if r["单式"] == "✗"]
    draws = [r for r in fails if r["实际"] == "平"]
    print(f"\n【失手归因】共 {len(fails)} 场")
    print(f"  其中被平局打掉 {len(draws)} 场：" + "、".join(f"第{r['场次']}场({r['对阵']})" for r in draws))
    hi = [r for r in fails if r["置信"] >= 0.6]
    if hi:
        print(f"  高置信(≥60%)翻车 {len(hi)} 场：")
        for r in hi:
            print(f"    第{r['场次']}场 {r['对阵']} 判{r['预测']}({r['置信']:.0%}) → 实际{r['实际']}")

    L["实际开奖"] = result
    L["复盘"] = {"单式命中": s_hit, "复式命中": f_hit,
                "单式胜率": round(s_hit/14, 4), "复式胜率": round(f_hit/14, 4),
                "期望单式": exp_s, "期望复式": exp_f,
                "偏差单式": round(s_hit-exp_s, 2), "偏差复式": round(f_hit-exp_f, 2),
                "加保收益": f_hit-s_hit, "落在预期区间": bool(6 <= s_hit <= 10),
                "失手": len(fails), "被平局打掉": len(draws), "逐场": rows}
    L["已复盘"] = True
    fp.write_text(json.dumps(L, ensure_ascii=False, indent=1), encoding="utf-8")

    # 累积台账
    lp = BASE / "batches" / "SFC_LEDGER.json"
    led = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else []
    led = [x for x in led if x["期号"] != period]
    led.append({"期号": period, "单式命中": s_hit, "复式命中": f_hit,
                "预测胜率": S["预测胜率"], "实际胜率": round(s_hit/14, 4),
                "期望": exp_s, "偏差": round(s_hit-exp_s, 2)})
    led.sort(key=lambda x: x["期号"])
    lp.write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding="utf-8")

    tot_h = sum(x["单式命中"] for x in led); tot_n = len(led)*14
    print(f"\n【累积台账】{len(led)} 期 · {tot_n} 场 · 累计命中 {tot_h} 场 = {tot_h/tot_n:.1%}")
    print(f"✅ 已写入 {fp.name} 与 SFC_LEDGER.json")
    return L


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
