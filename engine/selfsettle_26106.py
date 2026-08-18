"""自行结算第 26106 期（官方开奖被推迟，但比赛已全部踢完）

背景
  26106 售彩 8/16 21:30 截止，14 场于 8/16–8/17 全部踢完。
  但官方开奖 XML 至今仍标记 14 场「未完赛」(Result=-1)、销量 0，
  且 26106 不在 500 的已开奖期号列表中（最新已开奖为 26105）。
  据用户获悉：26106 将与 26107 一并开奖（26107 开奖时间 2026-08-21）。

  然而 500 自家的**联赛数据**（liansai）里这 14 场比分俱全、status=5（完赛）。
  因此可以自行结算，不必干等官方回填。

⚠ 本结算为**非官方**，仅用于提前复盘。官方赛果公布后由 auto_review.py
  覆盖写入并以官方为准；本文件只写 预复盘 字段，不写 已复盘=True，
  也不进任何台账，避免污染统计。
"""
import sys, json, re, html, datetime as dt
import xml.etree.ElementTree as ET
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
import build_l500 as B

PERIOD = "26106"
ZH = {"3": "胜", "1": "平", "0": "负"}


def fixtures():
    raw = B._curl(f"https://kaijiang.500.com/static/info/kaijiang/xml/sfc/{PERIOD}.xml")
    r = ET.fromstring(raw.decode("utf-8", "replace"))
    out = []
    for m in sorted(r.findall(".//MatchTeam"), key=lambda x: int(x.attrib.get("OrderNum", 0))):
        a = m.attrib
        out.append({"no": int(a["OrderNum"]), "赛事": a.get("SimpleGBName", ""),
                    "主ID": a["HomeTeamID"], "客ID": a["GuestTeamID"],
                    "主队": a.get("HomeTeamName", ""), "客队": a.get("GuestTeamName", ""),
                    "开赛": a.get("MatchTime", "")})
    return out


def team_results(tid):
    h = B.get(f"https://liansai.500.com/team/{tid}/teamfixture", f"ss_{tid}.html", refetch=True)
    if not h: return {}
    out = {}
    for blob in re.findall(r"data='(\{.*?\})'", h, re.S):
        try: d = json.loads(html.unescape(blob))
        except Exception: continue
        if d.get("STATUSID") == 5 and d.get("HOMESCORE") is not None:
            out[(str(d.get("HOMETEAMID")), str(d.get("AWAYTEAMID")), d.get("MATCHDATE"))] = d
    return out


def main():
    fx = fixtures()
    print("=" * 84)
    print(f"  第 {PERIOD} 期 · 自行结算（官方开奖推迟，比赛已踢完）")
    print("=" * 84)

    res, rows = [], []
    for g in fx:
        tr = team_results(g["主ID"])
        hit = None
        for (h, a, d), v in tr.items():
            if h == g["主ID"] and a == g["客ID"] and (d or "")[:10] >= "2026-08-15":
                hit = v; break
        if not hit:
            # 允许部分结算：未完赛场次记为 ?，不阻断其余 13 场的复盘
            print(f"  ⚠ 第{g['no']}场 {g['主队']}vs{g['客队']} 尚未完赛（这正是整期无法开奖的原因）")
            res.append("?")
            rows.append({**g, "比分": "未赛", "码": "?", "日期": None})
            continue
        hs, as_ = hit["HOMESCORE"], hit["AWAYSCORE"]
        code = "3" if hs > as_ else ("1" if hs == as_ else "0")
        res.append(code)
        rows.append({**g, "比分": f"{hs}:{as_}", "码": code, "日期": hit.get("MATCHDATE")})

    R = "".join(res)
    print(f"\n  自算赛果：{R}\n")

    L = json.loads((BASE / "batches" / f"sfc_{PERIOD}_LOCK.json").read_text(encoding="utf-8"))
    S = L["单式"]; F = L.get("复式16元")
    fu = {d["场次"]: d for d in (F or {}).get("明细", [])}
    N = L.get("任选九对照") or {}
    n_idx = set(N.get("场次", []))

    s_hit = f_hit = n_hit = 0
    print(f"  {'场':>3} {'赛事':<6}{'对阵':<26}{'比分':>7}{'实际':>5}{'预测':>5}{'置信':>7}"
          f"{'单式':>5}{'复式':>7}{'任九':>5}")
    print("  " + "-" * 82)
    for i, g in enumerate(L["逐场"]):
        r = rows[i]; act = res[i]
        if act == "?":
            print(f"  {g['场次']:>3} {rows[i]['赛事']:<6}{g['对阵']:<26}{'未赛':>7}{'?':>5}{ZH[g['单选']]:>5}{g['置信']:>7.0%}{'待定':>5}{'待定':>7}{'待定':>5}")
            continue
        ok = (g["单选"] == act); s_hit += ok
        d = fu.get(g["场次"])
        fok = (act in d["选项"]) if d else None
        if d: f_hit += bool(fok)
        nin = g["场次"] in n_idx
        if nin and ok: n_hit += 1
        mark = "  ← 高置信翻车" if (not ok and g["置信"] >= 0.55) else ""
        print(f"  {g['场次']:>3} {r['赛事']:<6}{g['对阵']:<26}{r['比分']:>7}{ZH[act]:>5}"
              f"{ZH[g['单选']]:>5}{g['置信']:>7.0%}{'✓' if ok else '✗':>5}"
              f"{(('✓' if fok else '✗')+' '+d['选项']) if d else '—':>7}"
              f"{('✓' if ok else '✗') if nin else '—':>5}{mark}")
    print("  " + "-" * 82)

    exp = S["期望命中"]
    n_done = sum(1 for c in res if c != "?")
    n_pend = 14 - n_done
    print(f"\n  【核心结果 · 非官方】")
    print(f"    单式（已完赛 {n_done} 场）  {s_hit}/{n_done} = {s_hit/max(n_done,1):.1%}"
          f"   全期期望 {exp}")
    if F:
        print(f"    16 元复式（已完赛 {n_done} 场） {f_hit}/{n_done}"
              f"   全期期望 {F['期望命中']}   加保救回 {f_hit-s_hit} 场")
    if n_idx:
        print(f"    任选九对照   {n_hit}/9    期望 {N.get('期望命中')}"
              f"   {'🎉 全中' if n_hit==9 else ''}")
    nd = R.count("1")
    print(f"\n  【平局】实际 {nd} 场（{nd/14:.0%}）· 模型期望 "
          f"{sum(g['平'] for g in L['逐场']):.2f} 场")
    dm = [(g, res[i]) for i, g in enumerate(L["逐场"]) if res[i] == "1" and g["单选"] != "1"]
    if dm:
        print(f"    被平局打掉 {len(dm)} 场：" +
              "、".join(f"第{g['场次']}场({g['置信']:.0%})" for g, _ in dm))

    L["预复盘_非官方"] = {
        "赛果": R, "来源": "500 联赛数据自行结算（官方开奖推迟）",
        "结算时间": str(dt.datetime.now())[:19],
        "单式命中": s_hit, "复式命中": f_hit, "任九命中": n_hit,
        "实际平局数": nd, "偏差": round(s_hit - exp, 2),
        "说明": "非官方结果，官方公布后由 auto_review 覆盖；不计入任何台账",
        "逐场": rows,
    }
    (BASE / "batches" / f"sfc_{PERIOD}_LOCK.json").write_text(
        json.dumps(L, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  ⚠ 以上为非官方自行结算，仅供提前复盘。")
    print(f"    官方赛果公布后 auto_review.py 会覆盖并以官方为准，本结果不进台账。")
    print(f"  ✅ 已写入 sfc_{PERIOD}_LOCK.json（字段 预复盘_非官方）")
    return R


if __name__ == "__main__":
    main()
