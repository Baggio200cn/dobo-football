"""从磁盘缓存重建扩展联赛库（缓存即真相）

为什么需要它：
  L500_MATCHES.json 是「增量落盘」的产物，一旦某轮抓取被中断/重启，
  下一轮会从一个不完整的快照续写，之前抓到的会被覆盖丢失
  （实测：荷乙 140→22、葡甲 138→20，凭空少了约 250 场）。

  但 cache_500/gm_{stid}_{round}.json 是不可变的原始响应，永远不丢。
  因此正确做法是：**每次都从缓存全量重建**，JSON 只是派生产物。

联赛名从 cache_500/lg_{seasonId}.html 反查：
  页面标题给赛事名，页面里的 /jifen-{stid}/ 给 stageId → 建立 stid → 联赛名 映射。
"""
import sys, json, re
from collections import Counter, defaultdict
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE

CACHE = BASE / "cache_500"


def stid_index():
    """cache_500/lg_*.html → {stid: (联赛名, 赛季标题)}"""
    idx = {}
    for f in CACHE.glob("lg_*.html"):
        try: h = f.read_text(encoding="utf-8", errors="replace")
        except Exception: continue
        t = re.search(r"<title>【([^】]*)】", h)
        title = t.group(1) if t else ""
        # 标题形如「2025/2026赛季_克亚联_克亚联」
        parts = title.split("_")
        nm = parts[1] if len(parts) > 1 else ""
        for stid in set(re.findall(r"/jifen-(\d+)/", h)):
            if nm: idx[stid] = (nm, title)
    return idx


def main():
    idx = stid_index()
    print(f"stid → 联赛 索引：{len(idx)} 条\n")

    rows, unknown = {}, Counter()
    per = defaultdict(set)
    for f in sorted(CACHE.glob("gm_*.json")):
        m = re.match(r"gm_(\d+)_(\d+)\.json$", f.name)
        if not m: continue
        stid, rnd = m.group(1), int(m.group(2))
        try: d = json.loads(f.read_text(encoding="utf-8"))
        except Exception: continue
        if not isinstance(d, list): continue
        nm = idx.get(stid, (None, None))[0]
        if not nm:
            unknown[stid] += len(d); continue
        for x in d:
            if x.get("status") != 5 or x.get("hscore") is None: continue
            rows[x["fid"]] = {
                "fid": x["fid"], "stid": int(stid), "round": x.get("round"),
                "联赛": nm, "日期": (x.get("stime") or "")[:10],
                "主ID": x.get("hid"), "客ID": x.get("gid"),
                "主队": x.get("hname") or x.get("hsxname"),
                "客队": x.get("gname") or x.get("gsxname"),
                "主进": x.get("hscore"), "客进": x.get("gscore"),
                "主半": x.get("hhalfscore"), "客半": x.get("ghalfscore"),
                "胜赔": x.get("win"), "平赔": x.get("draw"), "负赔": x.get("lost"),
            }
            per[nm].add(stid)

    out = sorted(rows.values(), key=lambda x: x["日期"])
    fp = BASE / "batches" / "L500_MATCHES.json"
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    P1 = {"克亚联", "捷甲", "保超", "斯伐超", "斯洛文甲", "匈甲", "亚美联", "阿塞联",
          "塞甲联", "塞浦甲", "冰岛超", "哈萨超", "黑山甲", "卢森甲", "威超"}
    c = Counter(r["联赛"] for r in out)
    print("=" * 68)
    print(f"  从缓存重建：{len(out):,} 场 / {len(c)} 联赛")
    print("=" * 68)
    print(f"  {'联赛':<12}{'赛季':>5}{'场次':>7}   {'球队':>5}")
    for nm, n in c.most_common():
        tids = set()
        for r in out:
            if r["联赛"] == nm: tids.add(r["主ID"]); tids.add(r["客ID"])
        print(f"  {nm:<12}{len(per[nm]):>5}{n:>7}   {len(tids):>5}{'  ★P1' if nm in P1 else ''}")
    p1n = sum(v for k, v in c.items() if k in P1)
    print(f"\n  ★ P1 欧战常客合计 {p1n:,} 场 / {len([k for k in c if k in P1])} 个联赛")
    if out:
        od = sum(1 for r in out if r["胜赔"])
        print(f"  时间跨度 {out[0]['日期']} ~ {out[-1]['日期']} · 含欧赔 {od/len(out):.0%}")
    if unknown:
        print(f"\n  ⚠ {len(unknown)} 个 stid 无联赛名（lg_*.html 缺失），"
              f"涉及 {sum(unknown.values())} 条原始记录，可重跑 stage_of 补齐")
    print(f"  → {fp.name}")
    return out


if __name__ == "__main__":
    main()
