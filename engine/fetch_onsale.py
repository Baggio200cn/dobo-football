"""抓取「在售 / 待开奖」期次，供网站实时公布赛程与投注截止时间。

同一个 XML 源在开奖前就已挂出完整赛程与平均赔率：
    StartTime        开始售卖
    EndTime          官方售彩截止   ← 网站要突出显示的
    FirstMatchTime   首场开赛
    AllMatchFinished 0=未踢完
    MatchTeam@AverageOdds  "胜 平 负"

状态判定（以运行时刻为准，北京时间）：
    now < EndTime                     → 在售
    EndTime <= now, 未开奖            → 已截止·待开奖
    已开奖                            → 不收录（走复盘）
"""
import sys, json, datetime as dt
import xml.etree.ElementTree as ET
import requests
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE

XML = "https://kaijiang.500.com/static/info/kaijiang/xml/sfc/{}.xml"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
LOOKAHEAD = 6        # 从最新已知期次往后探几期


def parse(period):
    try:
        r = requests.get(XML.format(period), headers=UA, timeout=30)
        if r.status_code != 200: return None
        root = ET.fromstring(r.content.decode("utf-8", "replace"))
    except Exception:
        return None
    g = lambda t: (root.findtext(t) or "").strip()
    if g("PeriodicalNO") != str(period): return None
    mts = root.findall(".//MatchTeam")
    if not mts: return None

    games = []
    for m in sorted(mts, key=lambda x: int(x.attrib.get("OrderNum", 0))):
        a = m.attrib
        od = (a.get("AverageOdds") or "").split()
        games.append({
            "场次": int(a.get("OrderNum", 0)),
            "赛事": a.get("SimpleGBName", ""),
            "主队": a.get("HomeTeamName", ""),
            "客队": a.get("GuestTeamName", ""),
            "开赛": a.get("MatchTime", ""),
            "赔率": [float(x) for x in od] if len(od) == 3 else None,
            "比分": a.get("MatchScore", ""),
        })
    return {"期号": str(period), "开始": g("StartTime"), "截止": g("EndTime"),
            "首场": g("FirstMatchTime"), "开奖时间": g("ResultTime"),
            "已开奖": bool(g("Result").strip()) and g("AllMatchFinished") == "1",
            "赛程": games}


def kaijiang(period):
    """已开奖期次的赛果 + 中奖信息（销量、三档奖金、逐场比分）"""
    import re as _re
    try:
        r = requests.get(XML.format(period), headers=UA, timeout=30)
        root = ET.fromstring(r.content.decode("utf-8", "replace"))
    except Exception:
        return None
    g = lambda t: (root.findtext(t) or "").strip()
    res = "".join(g("Result").split(","))
    if len(res) != 14: return None
    num = lambda t: int(_re.sub(r"[^\d]", "", g(t) or "0") or 0)
    games = []
    for m in sorted(root.findall(".//MatchTeam"),
                    key=lambda x: int(x.attrib.get("OrderNum", 0))):
        a = m.attrib
        games.append({"场次": int(a.get("OrderNum", 0)), "赛事": a.get("SimpleGBName", ""),
                      "主队": a.get("HomeTeamName", ""), "客队": a.get("GuestTeamName", ""),
                      "比分": a.get("MatchScore", ""), "果": a.get("Result", "")})
    return {"期号": str(period), "开奖": res, "开奖时间": g("ResultTime"),
            "销量": num("TotalMoney"), "任九销量": num("NineTotalMoney"),
            "奖金": {"一等奖": {"注数": num("Num1"), "单注": num("Money1")},
                    "二等奖": {"注数": num("Num2"), "单注": num("Money2")},
                    "任九": {"注数": num("NineNum"), "单注": num("NineMoney")}},
            "赛程": games}


def latest_known():
    """从已有 LOCK / 预测文件推断最新期号"""
    import glob, re
    ns = []
    for f in glob.glob(str(BASE / "batches" / "sfc_*.json")):
        m = re.search(r"sfc_(\d{5})", f)
        if m: ns.append(int(m.group(1)))
    return max(ns) if ns else 26106


def main():
    now = dt.datetime.now()
    start = latest_known()
    out = []
    print(f"从 {start} 起向后探测 {LOOKAHEAD} 期（当前 {now:%Y-%m-%d %H:%M}）\n")
    # 起点回退 2 期：已截止但未开奖的期次仍要留在看板上
    for p in range(start - 2, start + LOOKAHEAD):
        d = parse(p)
        if d is None:
            print(f"  {p}  —— 尚未挂出")
            continue
        if d["已开奖"]:
            print(f"  {p}  已开奖，跳过（走复盘）")
            continue
        try:
            end = dt.datetime.strptime(d["截止"], "%Y-%m-%d %H:%M:%S")
        except Exception:
            end = None
        d["状态"] = "在售" if (end and now < end) else "已截止·待开奖"
        d["剩余秒"] = int((end - now).total_seconds()) if (end and now < end) else 0
        # 是否已有我们的预测
        d["已预测"] = (BASE / "batches" / f"sfc_{p}_LOCK.json").exists()
        lg = {}
        for g in d["赛程"]: lg[g["赛事"]] = lg.get(g["赛事"], 0) + 1
        d["联赛构成"] = lg
        out.append(d)
        hrs = d["剩余秒"] / 3600
        print(f"  {p}  {d['状态']:<12} 截止 {d['截止']}"
              f"{f'（剩 {hrs:.1f} 小时）' if hrs > 0 else ''}"
              f"  {'✓ 已预测' if d['已预测'] else '✗ 未预测'}")
        print(f"        联赛：{'、'.join(f'{k}×{v}' for k, v in lg.items())}")

    # 最近已开奖期次的赛果与中奖信息（供网站「开奖公布」实时展示）
    drawn = []
    for p in range(start, start - 5, -1):
        d = parse(p)
        if not d or not d["已开奖"]:
            continue
        kj = kaijiang(p)
        if kj: drawn.append(kj)
        if len(drawn) >= 3: break
    if drawn:
        print(f"\n开奖公布：{'、'.join(x['期号'] for x in drawn)}")

    fp = BASE / "batches" / "ONSALE.json"
    fp.write_text(json.dumps({"更新时间": str(now)[:19], "期次": out, "开奖": drawn},
                             ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 收录在售 {len(out)} 期 · 开奖 {len(drawn)} 期 → {fp.name}")
    return out


if __name__ == "__main__":
    main()
