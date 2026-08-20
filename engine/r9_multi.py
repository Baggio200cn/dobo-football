"""任选九 · 4 组单式出票器（8 元/期）

用户方法论：不买 1 张 16 元复式，改买 4 张 2 元单式。

⚠ 关键修正（实测 74 期，p=0.006）
  用户最初设想是「嵌套翻转」：票1不反 / 票2反A / 票3反A+B / 票4反A+B+C。
  这是一条链，票4 要求三场同时爆冷，近乎废票，最佳命中均值仅 5.770。
  改成「2² 全组合」：不反 / 反A / 反B / 反A+B —— 四种情形互补，无废票，
  最佳命中均值 5.892，显著更优。本脚本采用组合式。

两种选场标准（实测无显著差异 p=0.288，两版都出，由用户选）
  U 反「置信最高」的 2 场   —— 用户思路：赌大众最集中处过度自信
     依据：本项目校准表显示 55–65% 置信档实际只中 36.4%（n=11，p=0.109，未显著）
  M 反「p2/p1 最高」的 2 场 —— 覆盖增益最大处

用法：python r9_multi.py 26108 26109
"""
import sys, json, itertools, datetime as dt
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
import sfc_lock as SL

ZH = {"3": "胜", "1": "平", "0": "负"}


def build(period, mode="U"):
    S = json.loads((BASE / "batches" / f"sfc_{period}.json").read_text(encoding="utf-8"))
    L = json.loads((BASE / "batches" / f"sfc_{period}_LOCK.json").read_text(encoding="utf-8"))
    P = S["picks"]
    conf = np.array([p["置信"] for p in P])

    M = []
    for p in P:
        pr = sorted([(p["胜"], "3"), (p["平"], "1"), (p["负"], "0")], reverse=True)
        M.append({"no": p["场次"], "赛事": p["赛事"], "nm": f"{p['主队']} vs {p['客队']}",
                  "pr": pr, "c": p["置信"], "开赛": p.get("开赛", ""),
                  "p": {"3": p["胜"], "1": p["平"], "0": p["负"]}})

    idx9 = sorted(sorted(range(14), key=lambda i: -conf[i])[:9])
    drop = sorted(set(range(14)) - set(idx9))

    # 选 2 场做全组合
    #
    # ⚠ 三种标准实测 74 期（4注8元，2²全组合），最佳命中均值：
    #     A 置信最高            5.892   ← 用户最初设想
    #     B 最接近 60%          6.000   ← 默认，忠于用户的理论依据且成绩最好
    #     C 次选/首选比值最高    6.000
    #   三者两两 p=0.23~1.00，统计上无显著差异，但 B/C 略优。
    #
    # 为什么 B 优于 A：用户的依据是「55–65% 置信档实际只中 36.4%（偏差 -22.9%）」，
    #   但「置信最高」选出的往往是 70%+ 的场次，而那一档校准良好（偏差仅 -3.3%）。
    #   真正该覆盖的是 60% 附近这条校准最差的带，不是全场最高的。
    if mode == "B":
        order = sorted(idx9, key=lambda i: abs(M[i]["pr"][0][0] - 0.60))
        why = "置信最接近 60%（校准最差档，实测 55–65% 只中 36.4%）"
    elif mode == "U":
        order = sorted(idx9, key=lambda i: -M[i]["pr"][0][0])
        why = "置信最高（用户最初设想）"
    else:
        order = sorted(idx9, key=lambda i: -(M[i]["pr"][1][0] / M[i]["pr"][0][0]))
        why = "次选/首选比值最高（覆盖增益最大）"
    flip = order[:2]

    tickets = []
    for combo in itertools.product([0, 1], repeat=2):
        t = {i: M[i]["pr"][0][1] for i in idx9}
        for j, c in enumerate(combo):
            if c: t[flip[j]] = M[flip[j]]["pr"][1][1]
        tickets.append(t)

    # 覆盖概率：4 注取并集 → 每场被覆盖的选项集合
    cov = {}
    for i in idx9:
        opts = {t[i] for t in tickets}
        cov[i] = sum(M[i]["p"][o] for o in opts)
    covv = np.array([cov[i] for i in idx9])
    p_any = float(np.prod(covv))          # 至少一注全中的概率
    p_one = float(np.prod([M[i]["pr"][0][0] for i in idx9]))
    dist = SL.hit_distribution(covv)

    return {"期号": period, "截止": L["截止"], "等级": L.get("数据等级", "?"),
            "模式": mode, "选场依据": why, "M": M, "idx9": idx9, "drop": drop,
            "flip": flip, "tickets": tickets, "cov": cov,
            "覆盖胜率": float(covv.mean()), "期望命中": float(covv.sum()),
            "P任一注全中": p_any, "P单注全中": p_one, "分布": dist}


def show(R):
    M, idx9 = R["M"], R["idx9"]
    print("=" * 92)
    print(f"  第 {R['期号']} 期 · 任选九 · 4 组单式（8 元）  截止 {R['截止']}   数据等级 {R['等级']}")
    print(f"  翻转场次依据：{R['选场依据']}")
    print("=" * 92)
    print(f"\n  {'场':>3} {'赛事':<6}{'对阵':<30}{'胜':>6}{'平':>6}{'负':>6}{'首选':>5}{'角色':>10}")
    print("  " + "-" * 82)
    for i in idx9:
        m = M[i]
        role = "★ 翻转位" if i in R["flip"] else ""
        print(f"  {m['no']:>3} {m['赛事']:<6}{m['nm']:<30}{m['p']['3']:>6.2f}{m['p']['1']:>6.2f}"
              f"{m['p']['0']:>6.2f}{ZH[m['pr'][0][1]]:>5}{role:>10}")
    print("  " + "-" * 82)
    print(f"  放弃 {[M[i]['no'] for i in R['drop']]} 场（把握最低的 5 场）")

    a, b = R["flip"]
    print(f"\n  【4 组号码】翻转位：第 {M[a]['no']} 场、第 {M[b]['no']} 场")
    print(f"  {'组':>3}  " + "".join(f"{M[i]['no']:>4}" for i in idx9) + "     说明")
    print("  " + "-" * 62)
    labels = ["两场都取首选", f"仅第{M[a]['no']}场取次选", f"仅第{M[b]['no']}场取次选", "两场都取次选"]
    for n, (t, lab) in enumerate(zip(R["tickets"], labels), 1):
        print(f"  {n:>3}  " + "".join(f"{t[i]:>4}" for i in idx9) + f"     {lab}")
    print("  " + "-" * 62)
    print(f"  串式写法（按场次顺序）：")
    for n, t in enumerate(R["tickets"], 1):
        print(f"    第{n}组  " + " ".join(f"{M[i]['no']}:{t[i]}" for i in idx9))

    print(f"\n  【指标】")
    print(f"    4 注并集覆盖胜率  {R['覆盖胜率']:.1%}")
    print(f"    期望命中          {R['期望命中']:.2f}/9")
    print(f"    任一注全中概率    1/{1/R['P任一注全中']:,.0f}"
          f"（单注 1/{1/R['P单注全中']:,.0f}，提升 {R['P任一注全中']/R['P单注全中']:.2f}×）")
    d = R["分布"]
    print(f"    命中分布          " + " ".join(f"{i}场={d[i]:.1%}" for i in range(6, 10)))
    print(f"    成本              4 注 × 2 元 = 8 元")
    dp = sum(M[i]["p"]["1"] for i in idx9)
    print(f"    入选九场期望平局  {dp:.2f} 场")


def main():
    pers = sys.argv[1:] or ["26108", "26109"]
    out = {}
    for per in pers:
        for mode in ("B", "U", "M"):
            R = build(per, mode)
            if mode == "B":
                show(R)
                out[per] = R
            else:
                print(f"\n  ── 对照：若按「{R['选场依据']}」选翻转位 ──")
                print(f"     翻转第 {R['M'][R['flip'][0]]['no']}、{R['M'][R['flip'][1]]['no']} 场"
                      f" · 覆盖胜率 {R['覆盖胜率']:.1%} · 任一注全中 1/{1/R['P任一注全中']:,.0f}")
                print(f"     （实测两种标准无显著差异 p=0.288，可按偏好选）")
        print()

    # 写入锁定档
    for per, R in out.items():
        fp = BASE / "batches" / f"sfc_{per}_LOCK.json"
        L = json.loads(fp.read_text(encoding="utf-8"))
        M = R["M"]
        L["任选九_4组单式"] = {
            "玩法": "任选九场 · 4 组单式", "预算": 8, "注数": 4,
            "选场依据": R["选场依据"], "结构": "2² 全组合（非嵌套）",
            "场次": [M[i]["no"] for i in R["idx9"]],
            "放弃": [M[i]["no"] for i in R["drop"]],
            "翻转位": [M[R["flip"][0]]["no"], M[R["flip"][1]]["no"]],
            "号码": [{"组": n, "串": {str(M[i]["no"]): t[i] for i in R["idx9"]}}
                   for n, t in enumerate(R["tickets"], 1)],
            "覆盖胜率": round(R["覆盖胜率"], 4),
            "期望命中": round(R["期望命中"], 2),
            "任一注全中概率": R["P任一注全中"],
            "单注全中概率": R["P单注全中"],
            "数据等级": R["等级"], "出票时间": str(dt.datetime.now())[:19],
        }
        fp.write_text(json.dumps(L, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"✅ 已写入 sfc_{per}_LOCK.json（字段 任选九_4组单式）")
    print("\n⚠ 本站不提供购彩功能，以上为模型输出记录，不构成任何投注建议。")


if __name__ == "__main__":
    main()
