"""从 500.com 建设扩展联赛库（自举发现 + 整季抓取）

背景见 vault《数据源调研_扩建联赛库_v1.md》。
现有 football-data.co.uk 的 36 联赛不含欧战常客（克罗地亚/塞尔维亚/捷克/
斯洛伐克/保加利亚/匈牙利/斯洛文尼亚/哈萨克斯坦/亚美尼亚等），导致 26107
期双方实测只有 1/14，模型被迫全程弃权。

发现链路（全自动，不需人工维护联赛表）
  1. 彩票开奖 XML          → 每场的 HomeTeamID / GuestTeamID（500 全站统一 ID）
  2. /team/{tid}/          → 该队所属联赛（赛季）ID —— 首页联赛列表不可靠，
                             克罗地亚首页写「克罗甲 9198」实为低级别，
                             真正顶级是「克亚联 19880」，必须反查
  3. /team/{tid}/teamfixture → data 属性 JSON 里的 SEASONID，得到历史赛季
  4. /zuqiu-{seasonId}/    → 页面 JS 里的 stageId
  5. index.php?c=score&a=getmatch&stid={stageId}&round={N}
                           → 整轮 JSON：比分/半场/欧赔/队ID，逐轮取完即整季

产出 batches/L500_MATCHES.json：以 500 team ID 为主键的比赛库。
不需要中英翻译 —— 彩票赛程与本库共用同一套 team ID。
"""
import sys, json, re, time, html
from collections import defaultdict
import requests
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0",
      "Referer": "https://liansai.500.com/"}
CACHE = BASE / "cache_500"
DELAY = 0.25
# 只收这些赛事类型（排除杯赛/友谊赛/预备队；跨国赛事不建 Elo 池）
SKIP_KW = ("杯", "友谊", "U19", "U21", "U23", "预备", "女", "青", "欧冠", "欧罗巴",
           "欧协联", "欧超杯", "世预", "亚冠", "解放者", "世外", "欧国联", "夏季赛",
           "全联", "外围")

# football-data.co.uk 已覆盖的联赛（500 的中文简名）—— 不重复抓，省请求额度。
# 名称对照 config.LEAGUES / NEW_LEAGUES。
EXISTING = {
    "英超", "西甲", "意甲", "德甲", "法甲", "荷甲", "葡超", "比甲", "土超", "希腊超",
    "英冠", "英甲", "英乙", "西乙", "意乙", "德乙", "法乙", "苏超",
    "瑞典超", "挪超", "芬超", "丹超", "奥甲", "瑞士超", "波甲", "爱超", "俄超",
    "罗甲", "美职联", "巴甲", "阿甲", "日职", "中超", "墨超",
}


FAILS = {"n": 0}


def _curl(url, params=None):
    """用 curl 子进程取页面。

    ⚠ 关键：liansai.500.com 会做 TLS 指纹识别（JA3）。
      同一时刻同一 URL：curl → HTTP 200 / 147KB，
                        python-requests → HTTP 503 / 0 字节。
      补齐 UA / Referer / Accept / Accept-Language / Sec-Fetch-* 全套浏览器头
      也无效（已逐一实测），因为差异在 TLS 握手层而非 HTTP 头层。
      故此处不用 requests，改调系统 curl。
    """
    import subprocess, urllib.parse
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    try:
        p = subprocess.run(
            ["curl", "-sS", "-m", "30", "--compressed",
             "-A", UA["User-Agent"], "-H", f"Referer: {UA['Referer']}", url],
            capture_output=True, timeout=40)
        return p.stdout if (p.returncode == 0 and len(p.stdout) > 200) else None
    except Exception:
        return None


def get(url, fname=None, params=None, as_json=False, refetch=False, tries=4):
    """带磁盘缓存 + 指数退避重试。

    500.com 在连续数百次请求后会限流（非 200 或空响应）。第一版没有重试，
    结果第 1 步 888 次请求把额度耗尽，第 2 步 112 个赛季全部取不到 → 0 场。
    """
    CACHE.mkdir(exist_ok=True)
    fp = (CACHE / fname) if fname else None
    if fp and fp.exists() and fp.stat().st_size > 200 and not refetch:
        t = fp.read_text(encoding="utf-8", errors="replace")
        try:
            return json.loads(t) if as_json else t
        except Exception:
            pass
    back = DELAY
    for k in range(tries):
        raw = _curl(url, params)
        if raw:
            if as_json:
                try:
                    d = json.loads(raw.decode("utf-8", "replace"))
                except Exception:
                    d = None
                if d is not None:
                    if fp: fp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
                    time.sleep(DELAY); return d
            else:
                for enc in ("gbk", "utf-8"):
                    try:
                        t = raw.decode(enc); break
                    except Exception: continue
                else:
                    t = raw.decode("utf-8", "replace")
                if fp: fp.write_text(t, encoding="utf-8")
                time.sleep(DELAY); return t
        back *= 2.5
        time.sleep(min(back, 20))
    FAILS["n"] += 1
    return None


def team_leagues(tid):
    """球队页 → [(联赛赛季ID, 赛事简名)]"""
    h = get(f"https://liansai.500.com/team/{tid}/", f"tm_{tid}.html")
    if not h: return []
    out, seen = [], set()
    for lid, nm in re.findall(r'/zuqiu-(\d+)/"[^>]*>([^<]{2,14})<', h):
        nm = nm.strip()
        if nm in seen: continue
        seen.add(nm); out.append((lid, nm))
    return out


def team_seasons(tid):
    """球队赛程页 → {赛事简名: set(SEASONID)}，用于挖历史赛季"""
    h = get(f"https://liansai.500.com/team/{tid}/teamfixture", f"tf_{tid}.html")
    if not h: return {}
    out = defaultdict(set)
    for raw in re.findall(r"data='(\{.*?\})'", h, re.S):
        try: m = json.loads(html.unescape(raw))
        except Exception: continue
        if m.get("SIMPLEGBNAME") and m.get("SEASONID"):
            out[m["SIMPLEGBNAME"]].add(m["SEASONID"])
    return out


def stage_of(season_id):
    """赛季页 → stageId（getmatch 的 stid 参数）

    注意：stageId 的 JS 变量只出现在 jifen 页，不在联赛落地页；
    但落地页含 /zuqiu-{sid}/jifen-{stid}/ 链接，直接从 URL 取更稳。"""
    h = get(f"https://liansai.500.com/zuqiu-{season_id}/", f"lg_{season_id}.html")
    if not h: return None, None
    m = (re.search(rf"/zuqiu-{season_id}/jifen-(\d+)/", h)
         or re.search(r"/jifen-(\d+)/", h)
         or re.search(r"stageId\s*=\s*(\d+)", h))
    t = re.search(r"<title>【([^】]*)】", h)
    return (m.group(1) if m else None), (t.group(1) if t else "")


def season_matches(stid, max_round=60):
    """逐轮取完整赛季"""
    out = []
    for r in range(1, max_round + 1):
        d = get("https://liansai.500.com/index.php", f"gm_{stid}_{r}.json",
                params={"c": "score", "a": "getmatch", "stid": stid, "round": r},
                as_json=True)
        if not d: break
        for m in d:
            if m.get("status") != 5 or m.get("hscore") is None: continue
            out.append({
                "fid": m["fid"], "stid": int(stid), "round": m.get("round"),
                "日期": (m.get("stime") or "")[:10],
                "主ID": m.get("hid"), "客ID": m.get("gid"),
                "主队": m.get("hname") or m.get("hsxname"),
                "客队": m.get("gname") or m.get("gsxname"),
                "主进": m.get("hscore"), "客进": m.get("gscore"),
                "主半": m.get("hhalfscore"), "客半": m.get("ghalfscore"),
                "胜赔": m.get("win"), "平赔": m.get("draw"), "负赔": m.get("lost"),
            })
    return out


def discover(seed_tids):
    """从种子球队 ID 反查出需要建库的国内联赛（赛季ID → 赛事名）"""
    found = {}
    def ok(nm):
        return nm and nm not in EXISTING and not any(k in nm for k in SKIP_KW)
    for tid in seed_tids:
        for lid, nm in team_leagues(tid):
            if ok(nm): found[lid] = nm
        for nm, sids in team_seasons(tid).items():
            if ok(nm):
                for s in sids: found[str(s)] = nm
    return found


def main():
    # 种子：所有已缓存彩票期次里出现过的球队 ID
    seeds, seen_lg = set(), {}
    import xml.etree.ElementTree as ET
    for f in sorted((BASE / "cache_xml").glob("sfc_*.xml")):
        try: root = ET.fromstring(f.read_text(encoding="utf-8", errors="replace"))
        except Exception: continue
        for mt in root.findall(".//MatchTeam"):
            a = mt.attrib
            for k in ("HomeTeamID", "GuestTeamID"):
                if a.get(k): seeds.add(a[k])
    print(f"种子球队 {len(seeds)} 支（来自 {len(list((BASE/'cache_xml').glob('sfc_*.xml')))} 期彩票赛程）\n")

    print("=" * 82)
    print("  第 1 步 · 反查国内联赛")
    print("=" * 82)
    leagues = discover(sorted(seeds))
    byname = defaultdict(list)
    for sid, nm in leagues.items(): byname[nm].append(sid)
    print(f"  发现 {len(byname)} 个联赛 / {len(leagues)} 个赛季")
    for nm in sorted(byname, key=lambda x: -len(byname[x]))[:40]:
        print(f"    {nm:<12}赛季 {sorted(byname[nm])}")

    # 优先级：欧战常客 > 其它欧洲 > 其余。限流时先保住关键联赛。
    P1 = ["克亚联", "捷甲", "保超", "斯伐超", "斯洛文甲", "匈甲", "亚美联", "阿塞联",
          "塞甲联", "塞浦甲", "冰岛超", "哈萨超", "黑山甲", "卢森甲", "威超", "法罗超"]
    P2 = ["葡甲", "荷乙", "德丙联", "希腊超A", "挪甲", "瑞典超甲", "法丙", "社区盾"]
    def prio(nm):
        if nm in P1: return (0, P1.index(nm))
        if nm in P2: return (1, P2.index(nm))
        return (2, nm)

    print("\n" + "=" * 82)
    print("  第 2 步 · 逐赛季抓取（P1 欧战常客 → P2 其它欧洲 → P3 其余）")
    print("=" * 82)
    allm, stats = {}, defaultdict(lambda: [0, 0])
    for nm in sorted(byname, key=prio):
        for sid in sorted(byname[nm]):
            stid, title = stage_of(sid)
            if not stid: continue
            ms = season_matches(stid)
            for m in ms:
                m["联赛"] = nm
                allm[m["fid"]] = m
            stats[nm][0] += 1; stats[nm][1] += len(ms)
            print(f"  {nm:<12}赛季{sid:<7}stid={stid:<7}{len(ms):>4} 场   {title[:26]}", flush=True)
            # 增量落盘：限流中断也不丢已抓到的
            if len(allm) % 500 < len(ms):
                (BASE / "batches" / "L500_MATCHES.json").write_text(
                    json.dumps(sorted(allm.values(), key=lambda x: x["日期"]),
                               ensure_ascii=False, indent=1), encoding="utf-8")

    rows = sorted(allm.values(), key=lambda x: x["日期"])
    fp = BASE / "batches" / "L500_MATCHES.json"
    fp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  抓取失败次数（含重试后仍失败）：{FAILS['n']}")

    print("\n" + "=" * 82)
    print(f"  ✅ 扩展库建成：{len(rows):,} 场 / {len(stats)} 联赛")
    print("=" * 82)
    print(f"  {'联赛':<12}{'赛季数':>7}{'场次':>8}")
    for nm, (ns, nm_) in sorted(stats.items(), key=lambda x: -x[1][1]):
        print(f"  {nm:<12}{ns:>7}{nm_:>8}")
    if rows:
        od = sum(1 for r in rows if r["胜赔"])
        tids = set([r["主ID"] for r in rows]) | set([r["客ID"] for r in rows])
        print(f"\n  时间跨度 {rows[0]['日期']} ~ {rows[-1]['日期']}")
        print(f"  含欧赔 {od:,} 场（{od/len(rows):.0%}） · 球队 {len(tids)} 支")
    print(f"  → {fp.name}")
    return rows


if __name__ == "__main__":
    main()
