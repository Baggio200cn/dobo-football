"""通用自动预测：检测在售期次 → 够格就出票，不够格就拒绝并写明原因。

替代此前一期一个脚本的做法（sfc26104.py / sfc26106.py 都是手写 FIX 对照表）。

数据链路
    ONSALE.json（赛程 + 平均赔率，来自 500 XML）
  + TEAM_MAP.json（中→英队名，由 build_teammap.py 自举）
  + Elo 表（来自我们 36 联赛 10.7 万场数据库）
  → 逐场 胜平负概率 → 单式 + 16 元复式

⚠ 覆盖率闸门（这是本脚本的核心设计）
    26103 期（欧战）只中 5/14、偏差 -2.22，是历史最差。
    根因：欧战对手来自我们不覆盖的联赛，Elo 只能靠先验瞎猜。
    因此：双方球队都能在 Elo 表里找到的场次数 < MIN_COVER 时，
    **拒绝出票**，在网站上标注「数据不足，需人工介入」。
    宁可不出，不出垃圾。
"""
import sys, json, datetime as dt
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
import data as D
from model import elo_run, elo_1x2, implied_1x2, goal_tendency
import sfc_lock as SL
from lock26106 import best14, build_detail

W_MKT = 0.55          # 市场权重
MIN_COVER = 12        # 14 场里至少要有几场「双方均有实测 Elo」
NEUTRAL_KW = ("社区盾", "超级杯", "欧洲超级杯")   # 中立场地赛事关键词
CODE = {"H": "3", "D": "1", "A": "0"}
ZH = {"H": "胜", "D": "平", "A": "负"}


def resolve(zh, TM, elo):
    """中文队名 → (英文名, 是否有实测 Elo)"""
    v = TM.get(zh)
    if v and v["en"] in elo:
        return v["en"], True
    if v:
        return v["en"], False
    return None, False


def predict(period, sched, TM, elo, GT):
    picks, cover = [], 0
    for g in sched:
        neutral = any(k in (g.get("赛事") or "") for k in NEUTRAL_KW)
        th, okh = resolve(g["主队"], TM, elo)
        ta, oka = resolve(g["客队"], TM, elo)
        both = okh and oka
        cover += both
        eh = elo.get(th, 1500.0); ea = elo.get(ta, 1500.0)
        gts = [GT[x] for x in (th, ta) if x in GT]
        gt = float(np.mean(gts)) if gts else None
        pe = elo_1x2(eh, ea, hfa=0 if neutral else 60, goal_tend=gt)
        od = g.get("赔率")
        mk = implied_1x2(*od) if (od and len(od) == 3) else None
        if mk:
            p = {k: W_MKT * mk[k] + (1 - W_MKT) * pe[k] for k in "HDA"}
            s = sum(p.values()); p = {k: v / s for k, v in p.items()}
            src = "Elo+市场"
        else:
            p = pe; src = "仅Elo"
        code = max(p, key=p.get)
        picks.append({
            "场次": g["场次"], "赛事": g["赛事"], "主队": g["主队"], "客队": g["客队"],
            "主队EN": th, "客队EN": ta, "开赛": g["开赛"], "中立场": neutral,
            "Elo主": round(eh), "Elo客": round(ea), "Elo差": round(eh - ea),
            "胜": round(p["H"], 3), "平": round(p["D"], 3), "负": round(p["A"], 3),
            "推荐码": CODE[code], "推荐": f"{ZH[code]}({CODE[code]})",
            "置信": round(p[code], 3), "来源": src,
            "匹配": "实测" if both else ("单方" if (okh or oka) else "先验"),
        })
    return picks, cover


def make_lock(period, deadline, picks, cover, m):
    P = picks
    conf = np.array([p["置信"] for p in P])
    M = []
    for p in P:
        pr = sorted([(p["胜"], "H"), (p["平"], "D"), (p["负"], "A")], reverse=True)
        M.append({"no": p["场次"], "nm": f"{p['主队']}vs{p['客队']}", "pr": pr, "c": p["置信"]})

    single = "".join(p["推荐码"] for p in P)
    dist = SL.hit_distribution(conf)
    p14 = float(np.prod(conf))
    pr_fu, bets, ks = best14(M)
    detail, line = build_detail(M, ks, range(14))
    cov = np.array([d["覆盖概率"] for d in detail])
    distf = SL.hit_distribution(cov)

    # 任选九对照
    order = sorted(range(14), key=lambda i: -conf[i])
    top9 = sorted(order[:9])
    c9 = np.array([conf[i] for i in top9])

    return {
        "期号": period, "截止": deadline, "锁定时间": str(dt.datetime.now())[:19],
        "玩法建议": "胜负彩14场", "自动生成": True,
        "数据覆盖": {"双方实测": cover, "总场次": 14, "闸门": MIN_COVER},
        "训练库": {"场次": int(len(m)), "联赛": int(m.Div.nunique())},
        "单式": {"串": single, "期望命中": round(float(conf.sum()), 2),
                "预测胜率": round(float(conf.mean()), 4),
                "分布": [round(float(x), 5) for x in dist],
                "累计": [round(float(x), 5) for x in dist[::-1].cumsum()[::-1]],
                "最可能命中": int(dist.argmax()), "全中概率": p14},
        "复式16元": {"注数": bets, "金额": bets * 2, "串": line, "明细": detail,
                   "全中概率": pr_fu, "覆盖胜率": round(float(cov.mean()), 4),
                   "期望命中": round(float(cov.sum()), 2),
                   "分布": [round(float(x), 5) for x in distf],
                   "累计": [round(float(x), 5) for x in distf[::-1].cumsum()[::-1]],
                   "加保场": [d["场次"] for d in detail if d["选项数"] > 1],
                   "较单式提升": round(pr_fu / p14, 2)},
        "任选九对照": {"场次": [M[i]["no"] for i in top9],
                   "放弃": [M[i]["no"] for i in sorted(order[9:])],
                   "预测胜率": round(float(c9.mean()), 4),
                   "期望命中": round(float(c9.sum()), 2),
                   "全中概率": float(np.prod(c9))},
        "逐场": [{"场次": p["场次"], "赛事": p["赛事"],
                "对阵": f"{p['主队']}vs{p['客队']}", "胜": p["胜"], "平": p["平"],
                "负": p["负"], "单选": p["推荐码"], "置信": p["置信"],
                "Elo差": p["Elo差"], "来源": p["来源"], "匹配": p["匹配"],
                "实际": None, "单式命中": None} for p in P],
        "已复盘": False,
    }


def main():
    ON = json.loads((BASE / "batches" / "ONSALE.json").read_text(encoding="utf-8"))
    TM = json.loads((BASE / "batches" / "TEAM_MAP.json").read_text(encoding="utf-8"))
    m = D.load_matches()
    elo, _ = elo_run(m)
    GT = goal_tendency(m)

    todo = [p for p in ON["期次"] if p["状态"] == "在售" and not p["已预测"]]
    if not todo:
        print("没有需要预测的在售期次。"); return []

    made, skipped = [], []
    for p in todo:
        per = p["期号"]
        print("=" * 84)
        print(f"  第 {per} 期  截止 {p['截止']}")
        print("=" * 84)
        picks, cover = predict(per, p["赛程"], TM, elo, GT)
        q = {}
        for x in picks: q[x["匹配"]] = q.get(x["匹配"], 0) + 1
        print(f"  数据覆盖：双方实测 {cover}/14   明细 {q}")

        if cover < MIN_COVER:
            miss = [f"第{x['场次']}场 {x['主队']}vs{x['客队']}"
                    for x in picks if x["匹配"] != "实测"]
            reason = (f"双方实测仅 {cover}/14，低于闸门 {MIN_COVER}。"
                      f"以下场次球队不在我们的 36 联赛数据库内，Elo 只能靠先验估算：")
            print(f"\n  ⛔ 拒绝出票：{reason}")
            for s in miss[:8]: print(f"       {s}")
            if len(miss) > 8: print(f"       …… 另有 {len(miss)-8} 场")
            print(f"\n  依据：26103 期（同为欧战）双方实测不足，实际只中 5/14，偏差 -2.22，为历史最差。")
            skipped.append({"期号": per, "截止": p["截止"], "覆盖": cover,
                            "闸门": MIN_COVER, "原因": reason, "缺数据场次": miss,
                            "赛事构成": p.get("联赛构成", {})})
            print()
            continue

        obj = {"期号": per, "截止": p["截止"], "生成时间": str(dt.datetime.now())[:19],
               "单式": "".join(x["推荐码"] for x in picks),
               "训练库场次": int(len(m)), "训练库联赛": int(m.Div.nunique()),
               "picks": picks}
        (BASE / "batches" / f"sfc_{per}.json").write_text(
            json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
        lock = make_lock(per, p["截止"], picks, cover, m)
        (BASE / "batches" / f"sfc_{per}_LOCK.json").write_text(
            json.dumps(lock, ensure_ascii=False, indent=1), encoding="utf-8")

        S = lock["单式"]; F = lock["复式16元"]
        print(f"\n  {'场':>3} {'赛事':<6}{'对阵':<26}{'胜':>6}{'平':>6}{'负':>6}{'荐':>4}{'置信':>7}")
        for x in picks:
            print(f"  {x['场次']:>3} {x['赛事']:<6}{x['主队']+'vs'+x['客队']:<26}"
                  f"{x['胜']:>6.2f}{x['平']:>6.2f}{x['负']:>6.2f}"
                  f"{x['推荐码']:>4}{x['置信']:>7.1%}")
        print(f"\n  单式 {S['串']}  预测胜率 {S['预测胜率']:.1%} · 期望命中 {S['期望命中']}/14")
        print(f"  复式 {F['注数']} 注 {F['金额']} 元 · 加保第 {F['加保场']} 场 · "
              f"覆盖胜率 {F['覆盖胜率']:.1%} · 期望命中 {F['期望命中']}/14")
        print(f"  ✅ 已锁定 sfc_{per}_LOCK.json\n")
        made.append(per)

    fp = BASE / "batches" / "SKIPPED.json"
    fp.write_text(json.dumps({"更新时间": str(dt.datetime.now())[:19], "期次": skipped},
                             ensure_ascii=False, indent=1), encoding="utf-8")
    print("=" * 84)
    print(f"  自动出票 {len(made)} 期" + (f"：{', '.join(made)}" if made else "") +
          f" · 拒绝 {len(skipped)} 期" +
          (f"：{', '.join(s['期号'] for s in skipped)}" if skipped else ""))
    return made


if __name__ == "__main__":
    main()
