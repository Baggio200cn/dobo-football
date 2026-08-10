"""胜负彩（14 场）预测生成

数据源：500.com 每期 14 场对阵（人工录入或抓取）
模型：Elo（全库 10 万+ 场滚动）+ 市场赔率融合
输出：batches/sfc_<期号>.json + 网站数据

用法：python sfc.py
"""
import sys, json, datetime as dt
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE, ALL_LEAGUES
import data as D
from model import elo_run, elo_1x2, implied_1x2

OUT = BASE / "batches"
OUT.mkdir(exist_ok=True)

# ===== 第 26101 期（截止 2026-08-08 21:30）=====
PERIOD = "26101"
DEADLINE = "2026-08-08 21:30"
# (场次, 赛事, 开赛, 主队中文, 主队EN, 客队中文, 客队EN, 官方赔率[胜,平,负] 或 None)
FIXTURES = [
    (1,  "英联杯", "08-08 22:00", "斯旺西", "Swansea",          "伯明翰",   "Birmingham",   [2.60, 3.20, 2.32]),
    (2,  "英联杯", "08-08 22:00", "西汉姆联", "West Ham",       "朴茨茅斯", "Portsmouth",   [1.40, 4.20, 5.65]),
    (3,  "英联杯", "08-08 22:00", "德比郡", "Derby",            "林肯城",   "Lincoln",      None),
    (4,  "荷甲",  "08-08 22:30", "奈梅亨", "Nijmegen",          "特尔斯达", "Telstar",      [1.37, 4.45, 5.75]),
    (5,  "荷甲",  "08-09 00:45", "前进之鹰", "Go Ahead Eagles", "威廉二世", "Willem II",    [1.43, 4.15, 5.30]),
    (6,  "荷甲",  "08-09 02:00", "埃因霍温", "PSV Eindhoven",   "福图纳",   "For Sittard",  None),
    (7,  "荷甲",  "08-09 03:00", "阿尔克马尔", "AZ Alkmaar",    "海牙",     "Den Haag",     [1.37, 4.48, 5.70]),
    (8,  "葡超",  "08-08 22:30", "马里迪莫", "Maritimo",        "卡萨皮亚", "Casa Pia",     [1.98, 3.10, 3.32]),
    (9,  "葡超",  "08-09 01:00", "吉马良斯", "Guimaraes",       "阿罗卡",   "Arouca",       [1.75, 3.35, 3.85]),
    (10, "葡超",  "08-09 03:30", "阿马多拉", "Estrela",         "里斯本竞技", "Sp Lisbon",  None),
    (11, "瑞典超", "08-08 23:30", "米亚尔比", "Mjallby",        "埃尔夫斯堡", "Elfsborg",   None),
    (12, "挪超",  "08-08 22:00", "维京", "Viking",              "萨普斯堡", "Sarpsborg 08", None),
    (13, "挪超",  "08-09 00:00", "斯达", "Start",               "腓特烈斯塔", "Fredrikstad",None),
    (14, "芬超",  "08-09 00:00", "雅罗", "Jaro",                "瓦萨",     "VPS",          None),
]
ZH = {"H": "胜(3)", "D": "平(1)", "A": "负(0)"}


def main():
    print(f"胜负彩第 {PERIOD} 期 · 截止 {DEADLINE}\n")
    m = D.load_matches()
    print(f"训练库：{len(m):,} 场 · {m.Div.nunique()} 个联赛 · {m.Date.min().date()} → {m.Date.max().date()}")
    r, _ = elo_run(m)
    print(f"Elo 评分：{len(r)} 支球队\n")

    # 各队最近比赛数（衡量数据充分度）
    cnt = pd.concat([m.HomeTeam, m.AwayTeam]).value_counts()

    picks = []
    for no, lg, ko, hz, he, az, ae, odds in FIXTURES:
        rh, ra = r.get(he, 1500), r.get(ae, 1500)
        pe = elo_1x2(rh, ra)
        mk = implied_1x2(*odds) if odds else None
        if mk:
            P = {k: 0.55 * mk[k] + 0.45 * pe[k] for k in "HDA"}   # 有赔率：市场为主
            src = "Elo+市场"
        else:
            P = pe; src = "仅Elo"
        s = sum(P.values()); P = {k: v / s for k, v in P.items()}
        best = max(P, key=P.get)
        edge = (P[best] - mk[best]) if mk else None
        nh, na = int(cnt.get(he, 0)), int(cnt.get(ae, 0))
        picks.append({
            "场次": no, "赛事": lg, "开赛": ko,
            "主队": hz, "主队EN": he, "客队": az, "客队EN": ae,
            "Elo主": round(rh), "Elo客": round(ra), "Elo差": round(rh - ra),
            "样本主": nh, "样本客": na,
            "胜": round(P["H"], 3), "平": round(P["D"], 3), "负": round(P["A"], 3),
            "推荐": ZH[best], "推荐码": {"H": "3", "D": "1", "A": "0"}[best],
            "置信": round(P[best], 3),
            "市场胜": round(mk["H"], 3) if mk else None,
            "偏差": round(edge, 3) if edge is not None else None,
            "来源": src,
        })

    df = pd.DataFrame(picks)
    print(df[["场次", "赛事", "主队", "客队", "Elo差", "胜", "平", "负", "推荐", "置信", "来源"]].to_string(index=False))

    # ===== 单式（每场命中率最优，但全中概率极低）=====
    single = "".join(p["推荐码"] for p in picks)
    p_single = float(np.prod([p["置信"] for p in picks]))
    print(f"\n单式：{single}")
    print(f"  全中概率 {p_single:.2e}（约 {1/p_single:,.0f} 分之一）—— 数学上每场最优，但单式中奖希望渺茫")

    # ===== 复式（贪心：优先给「次选最接近首选」的场次加保）=====
    CODE = {"H": "3", "D": "1", "A": "0"}
    ranked = []
    for p in picks:
        pr = sorted([(p["胜"], "H"), (p["平"], "D"), (p["负"], "A")], reverse=True)
        ranked.append({"no": p["场次"], "pr": pr, "ratio": pr[1][0] / pr[0][0]})
    order = sorted(range(14), key=lambda i: -ranked[i]["ratio"])

    def build(n_double):
        """给前 n_double 个最不确定的场次加第二选择"""
        dbl = set(order[:n_double])
        line, prob = [], 1.0
        for i in range(14):
            pr = ranked[i]["pr"]
            k = 2 if i in dbl else 1
            sel = "".join(sorted(CODE[pr[j][1]] for j in range(k)))
            line.append(sel); prob *= sum(pr[j][0] for j in range(k))
        return line, prob, 2 ** n_double

    print(f"\n复式方案（每注 2 元）：")
    print(f"{'加保场数':>6}{'注数':>6}{'金额':>8}{'全中概率':>12}{'提升':>8}")
    plans = []
    for nd in (0, 2, 3, 4, 5, 6, 7):
        line, prob, bets = build(nd)
        plans.append({"加保": nd, "注数": bets, "金额": bets * 2,
                      "全中概率": prob, "提升": prob / p_single, "串": line})
        print(f"{nd:>6}{bets:>6}{bets*2:>7}元{prob:>12.2e}{prob/p_single:>7.1f}×")

    rec = plans[4]   # 5 场加保 = 32 注 = 64 元，性价比拐点
    print(f"\n⭐ 推荐方案：加保 {rec['加保']} 场 · {rec['注数']} 注 · {rec['金额']} 元 · 全中概率提升 {rec['提升']:.1f} 倍")
    print("   " + " ".join(f"{i+1}:{s}" for i, s in enumerate(rec["串"])))
    print(f"   加保场次：" + "、".join(
        f"第{ranked[i]['no']}场({picks[i]['主队']}vs{picks[i]['客队']})" for i in order[:rec["加保"]]))

    # ===== 平局候选（按平局概率排序，供参考）=====
    dr = sorted(picks, key=lambda p: -p["平"])[:4]
    print(f"\n平局概率最高 4 场（真实平局率约 26%，14 场预期 3.7 场平局）：")
    for p in dr:
        print(f"  {p['场次']:>2} {p['主队']}vs{p['客队']:<12} 平 {p['平']:.0%} · Elo差 {p['Elo差']:+d}")

    obj = {"期号": PERIOD, "截止": DEADLINE, "生成时间": str(dt.datetime.now())[:19],
           "训练库场次": int(len(m)), "训练库联赛": int(m.Div.nunique()),
           "单式": single, "单式全中概率": p_single,
           "复式方案": plans, "推荐方案": rec,
           "平局候选": [{"场次": p["场次"], "对阵": f"{p['主队']}vs{p['客队']}",
                       "平局概率": p["平"], "Elo差": p["Elo差"]} for p in dr],
           "picks": picks}
    fp = OUT / f"sfc_{PERIOD}.json"
    fp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 已存档 {fp.name}")
    return obj


if __name__ == "__main__":
    main()
