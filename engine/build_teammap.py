"""自举中文→英文队名对照表。

问题：500.com 给中文队名，我们的 Elo 表是 football-data 的英文名。
      此前靠手写 FIX 对照表（sfc26104.py / sfc26106.py），无法自动化。

解法：历史期次的 XML 里同时有【中文队名 + 比分 + 开赛时间 + 联赛】，
      我们的数据库里有【英文队名 + 比分 + 日期 + Div】。
      用「日期 ± 1 天 + 联赛 + 精确比分」三元组对撞，唯一命中即可确定中英对应。

      注意时区：500 的 MatchTime 是北京时间，欧洲深夜球在北京是次日凌晨，
      所以窗口取 ±1 天，并要求比分精确匹配来消歧。

产出 batches/TEAM_MAP.json： {"中文名": {"en": "英文名", "div": "P1", "n": 出现次数}}

用法：
  python build_teammap.py              # 增量扫描最近 80 期
  python build_teammap.py 60 26106     # 扫描截至 26106 的 60 期
"""
import sys, json, re, time
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
import datetime as dt
import pandas as pd
import requests
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE, LEAGUES, NEW_LEAGUES
from data import load_matches

XML = "https://kaijiang.500.com/static/info/kaijiang/xml/sfc/{}.xml"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
CACHE = BASE / "cache_xml"
MAP_FP = BASE / "batches" / "TEAM_MAP.json"

# 联赛中文名 → Div（合并两张表，并补 500 站的别名）
ZH2DIV = {v: k for k, v in {**LEAGUES, **NEW_LEAGUES}.items()}
ZH2DIV.update({"苏超": "SC0", "英超": "E0", "德甲": "D1", "意甲": "I1",
               "丹麦超": "DNK", "奥超": "AUT", "瑞士超": "SWZ", "波兰甲": "POL"})


def get_xml(period, ttl_days=3650):
    """带磁盘缓存的抓取（历史期次不会变，可永久缓存）"""
    CACHE.mkdir(exist_ok=True)
    fp = CACHE / f"sfc_{period}.xml"
    if fp.exists() and fp.stat().st_size > 200:
        return fp.read_text(encoding="utf-8", errors="replace")
    try:
        r = requests.get(XML.format(period), headers=UA, timeout=25)
        if r.status_code != 200: return None
        t = r.content.decode("utf-8", "replace")
        if "<PeriodicalNO>" not in t: return None
        fp.write_text(t, encoding="utf-8")
        time.sleep(0.25)
        return t
    except Exception:
        return None


def parse_finished(period):
    """只取已全部踢完的期次，返回逐场 (中主, 中客, 主进, 客进, 日期, Div)"""
    t = get_xml(period)
    if not t: return []
    try: root = ET.fromstring(t)
    except Exception: return []
    if (root.findtext("AllMatchFinished") or "") != "1": return []
    out = []
    for m in root.findall(".//MatchTeam"):
        a = m.attrib
        sc = a.get("MatchScore", "")
        mm = re.match(r"^(\d+):(\d+)$", sc)
        if not mm: continue
        div = ZH2DIV.get(a.get("SimpleGBName", ""))
        if not div: continue
        try:
            d = dt.datetime.strptime(a.get("MatchTime", "")[:10], "%Y-%m-%d").date()
        except Exception:
            continue
        out.append((a.get("HomeTeamName", ""), a.get("GuestTeamName", ""),
                    int(mm.group(1)), int(mm.group(2)), d, div))
    return out


def main():
    n_scan = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    end = int(sys.argv[2]) if len(sys.argv) > 2 else None

    m = load_matches()
    m = m[["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].dropna()
    m["d"] = pd.to_datetime(m["Date"]).dt.date
    # 按 (Div, 比分) 建索引，加速
    idx = defaultdict(list)
    for x in m.itertuples():
        idx[(x.Div, int(x.FTHG), int(x.FTAG))].append((x.d, x.HomeTeam, x.AwayTeam))
    print(f"数据库 {len(m):,} 场 / {m.Div.nunique()} 联赛\n")

    if end is None:
        # 从现有文件推断最新期
        import glob
        ns = [int(g) for f in glob.glob(str(BASE / "batches" / "sfc_*.json"))
              for g in re.findall(r"sfc_(\d{5})", f)]
        end = max(ns) if ns else 26106

    votes = defaultdict(lambda: defaultdict(int))   # zh -> en -> count
    divof = defaultdict(lambda: defaultdict(int))   # zh -> div -> count
    n_ok = n_amb = n_miss = n_per = 0

    for p in range(end - n_scan, end + 1):
        games = parse_finished(p)
        if not games: continue
        n_per += 1
        for zh_h, zh_a, gh, ga, d, div in games:
            cands = [c for c in idx.get((div, gh, ga), []) if abs((c[0] - d).days) <= 1]
            if len(cands) == 1:
                _, en_h, en_a = cands[0]
                votes[zh_h][en_h] += 1; divof[zh_h][div] += 1
                votes[zh_a][en_a] += 1; divof[zh_a][div] += 1
                n_ok += 1
            elif len(cands) > 1:
                n_amb += 1
            else:
                n_miss += 1

    print(f"扫描 {n_per} 个已开奖期次 · 唯一命中 {n_ok} 场 · 多解 {n_amb} · 未匹配 {n_miss}")

    # 合并旧表 + 手写种子
    old = {}
    if MAP_FP.exists():
        old = json.loads(MAP_FP.read_text(encoding="utf-8"))

    out = dict(old)
    new_cnt = upd_cnt = 0
    for zh, cnt in votes.items():
        en = max(cnt, key=cnt.get)
        n = cnt[en]
        div = max(divof[zh], key=divof[zh].get)
        conf = n / sum(cnt.values())
        if zh not in out:
            new_cnt += 1
        elif out[zh].get("en") != en and n > out[zh].get("n", 0):
            upd_cnt += 1
        elif zh in out and out[zh].get("n", 0) >= n:
            continue
        out[zh] = {"en": en, "div": div, "n": n, "conf": round(conf, 2)}

    MAP_FP.write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True),
                      encoding="utf-8")
    solid = sum(1 for v in out.values() if v["n"] >= 2)
    print(f"\n对照表：{len(out)} 支球队（新增 {new_cnt} · 更新 {upd_cnt}）")
    print(f"  ≥2 次印证的高可信条目：{solid}（{solid/max(len(out),1):.0%}）")
    print(f"  → {MAP_FP.name}")

    # 抽样展示
    sample = sorted(out.items(), key=lambda x: -x[1]["n"])[:12]
    print(f"\n  {'中文':<14}{'英文':<22}{'联赛':<6}{'印证'}")
    print("  " + "-" * 50)
    for zh, v in sample:
        print(f"  {zh:<14}{v['en']:<22}{v['div']:<6}{v['n']}")
    return out


if __name__ == "__main__":
    main()
