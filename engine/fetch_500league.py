"""从 500.com 抓取联赛历史赛果（打样版：克罗地亚顶级联赛）

数据源发现过程见 vault《数据源调研_扩建联赛库_v1.md》。

关键发现：球队赛程页 liansai.500.com/team/{tid}/teamfixture
          每场比赛是一个 <tr data='{...json...}'>，含：
            MATCHDATE / HOMETEAMID / AWAYTEAMID
            HOMESCORE / AWAYSCORE / HOMEHTSCORE / AWAYHTSCORE
            HOMETEAMSXNAME / AWAYTEAMSXNAME
            SIMPLEGBNAME / SEASONID
            WIN / DRAW / LOSE（欧赔）
          —— 比分与赔率一次到手，正是我们模型需要的两样。

策略：抓该联赛全部球队的 teamfixture，按 FIXTUREID 去重，即得整个联赛历史。
      联赛 ID 用「从彩票赛程的 team ID 反查」得到（首页列表的 ID 不可靠：
      克罗地亚首页写「克罗甲 9198」实为低级别联赛，真正顶级是「克亚联 19880」）。

用法：
  python fetch_500league.py 19880           # 打样：克亚联
  python fetch_500league.py 19880 --seed 448
"""
import sys, json, re, time, html
from pathlib import Path
from collections import defaultdict
import requests
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0"}
CACHE = BASE / "cache_500"
DELAY = 0.5


def get(url, fname, ttl_ok=True):
    CACHE.mkdir(exist_ok=True)
    fp = CACHE / fname
    if ttl_ok and fp.exists() and fp.stat().st_size > 2000:
        return fp.read_text(encoding="utf-8", errors="replace")
    try:
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code != 200: return None
        for enc in ("gbk", "utf-8"):
            try:
                t = r.content.decode(enc); break
            except Exception:
                continue
        else:
            t = r.content.decode("utf-8", "replace")
        fp.write_text(t, encoding="utf-8")
        time.sleep(DELAY)
        return t
    except Exception as e:
        print(f"    ⚠ {url} 失败：{e}")
        return None


def league_teams(lid):
    """联赛页 → 参赛球队 {tid: 名称}"""
    h = get(f"https://liansai.500.com/zuqiu-{lid}/", f"lg_{lid}.html")
    if not h: return {}
    out = {}
    for tid, nm in re.findall(r'/team/(\d+)/"[^>]*title="([^"]+)"', h):
        out[tid] = nm
    return out


def league_name(lid):
    """联赛页标题 → 赛事简名（teamfixture 用 SIMPLEGBNAME 而非 SEASONID 标识联赛）"""
    h = get(f"https://liansai.500.com/zuqiu-{lid}/", f"lg_{lid}.html") or ""
    m = re.search(r"<title>【[^_]*_([^_]+)_", h)
    return m.group(1) if m else ""


def team_matches(tid):
    """球队赛程页 → 逐场 dict（data 属性里的 JSON）"""
    h = get(f"https://liansai.500.com/team/{tid}/teamfixture", f"tf_{tid}.html")
    if not h: return []
    out = []
    for raw in re.findall(r"data='(\{.*?\})'", h, re.S):
        try:
            out.append(json.loads(html.unescape(raw)))
        except Exception:
            continue
    return out


def main():
    lid = sys.argv[1] if len(sys.argv) > 1 else "19880"
    teams, LNAME = league_teams(lid), league_name(lid)
    print("=" * 78)
    print(f"  打样：联赛 {lid} · 参赛球队 {len(teams)} 支")
    print("=" * 78)
    if not teams:
        print("  ⛔ 未取到球队列表"); return
    print("  " + "、".join(list(teams.values())[:12]))

    seen, rows = set(), []
    other = defaultdict(int)
    for i, (tid, nm) in enumerate(teams.items(), 1):
        ms = team_matches(tid)
        new = 0
        for m in ms:
            fid = m.get("FIXTUREID")
            if not fid or fid in seen: continue
            seen.add(fid)
            if m.get("SIMPLEGBNAME") != LNAME:
                other[m.get("SIMPLEGBNAME", "?")] += 1
                continue
            if m.get("HOMESCORE") is None or m.get("STATUSID") != 5: continue
            rows.append({
                "日期": m.get("MATCHDATE"), "联赛": m.get("SIMPLEGBNAME"),
                "主ID": m.get("HOMETEAMID"), "客ID": m.get("AWAYTEAMID"),
                "主队": m.get("HOMETEAMSXNAME"), "客队": m.get("AWAYTEAMSXNAME"),
                "主进": m.get("HOMESCORE"), "客进": m.get("AWAYSCORE"),
                "主半": m.get("HOMEHTSCORE"), "客半": m.get("AWAYHTSCORE"),
                "胜赔": m.get("WIN"), "平赔": m.get("DRAW"), "负赔": m.get("LOST"),
                "fid": fid,
            })
            new += 1
        print(f"  [{i:>2}/{len(teams)}] {nm:<16} 该队页 {len(ms):>4} 场 · 本联赛新增 {new:>3}")

    rows.sort(key=lambda x: (x["日期"] or ""))
    print("\n" + "=" * 78)
    print(f"  ✅ 本联赛去重后 {len(rows)} 场")
    if rows:
        print(f"  时间跨度 {rows[0]['日期']} ~ {rows[-1]['日期']}")
        yrs = defaultdict(int)
        for r in rows: yrs[(r["日期"] or "----")[:4]] += 1
        print(f"  按年分布：" + " · ".join(f"{k} {v}场" for k, v in sorted(yrs.items())))
        od = sum(1 for r in rows if r["胜赔"])
        ht = sum(1 for r in rows if r["主半"] is not None)
        print(f"  含欧赔 {od} 场（{od/len(rows):.0%}） · 含半场比分 {ht} 场（{ht/len(rows):.0%}）")
        tm = set([r["主队"] for r in rows]) | set([r["客队"] for r in rows])
        print(f"  涉及球队 {len(tm)} 支")
        print(f"\n  样例：")
        for r in rows[:3] + rows[-3:]:
            o = f"{r['胜赔']}/{r['平赔']}/{r['负赔']}" if r["胜赔"] else "无赔率"
            print(f"    {r['日期']}  {r['主队']:<16}{r['主进']}:{r['客进']:<4}{r['客队']:<16}{o}")
        if other:
            print(f"\n  （顺带抓到其它赛事 {sum(other.values())} 场，未计入："
                  + "、".join(f"{k}{v}" for k, v in sorted(other.items(), key=lambda x: -x[1])[:6]) + "）")

        fp = BASE / "batches" / f"L500_{lid}.json"
        fp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n  → {fp.name}")
    return rows


if __name__ == "__main__":
    main()
