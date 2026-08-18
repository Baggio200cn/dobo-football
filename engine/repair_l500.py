"""修复轮：只补 P1 欧战常客联赛，慢速抓取避免限流。

上一轮全量跑的教训（59 次抓取彻底失败）：
  · P1 联赛排在最前面执行，恰好撞上前一轮 888 次请求留下的限流窗口，
    重试 4 次全败被跳过 —— 最重要的联赛成了限流的牺牲品。
  · 本赛季（2026/27）欧洲联赛刚开踢，stid 拿到也是 0 场；
    真正有价值的是**上赛季**，必须显式指定。
  · 产出 196 场里大半是巴西州联赛，对胜负彩无用。

本轮策略：
  · 只打 P1 名单，赛季 ID 用已发现的历史赛季（不是当前赛季）
  · DELAY 提到 1.5s，重试 6 次、退避到 60s
  · 每个联赛抓完立即落盘，中断不丢
"""
import sys, json, time
import requests
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
import build_l500 as B

# 联赛 → 赛季ID 列表（含历史赛季）。来自 build_l500 第 1 步的反查结果。
TARGETS = {
    "克亚联":   ["14539", "19880"],
    "捷甲":     ["9163", "19949"],
    "保超":     ["9170", "19887"],
    "斯伐超":   ["9130", "19919"],
    "斯洛文甲": ["9183", "19929"],
    "匈甲":     ["9180", "19975"],
    "亚美联":   ["9199", "19998"],
    "阿塞联":   ["15459", "20073"],
    "塞甲联":   ["9182", "19892"],
    "塞浦甲":   ["9176"],
    "冰岛超":   ["19561"],
    "哈萨超":   ["19670"],
    "黑山甲":   ["9157", "19920"],
    "卢森甲":   ["9145", "20087"],
    "威超":     ["9162", "19974"],
}

B.DELAY = 1.5          # 慢下来


def main():
    fp = BASE / "batches" / "L500_MATCHES.json"
    allm = {}
    if fp.exists():
        for m in json.loads(fp.read_text(encoding="utf-8")):
            allm[m["fid"]] = m
    print(f"已有 {len(allm)} 场，开始修复轮（慢速 {B.DELAY}s/请求）\n")

    for nm, sids in TARGETS.items():
        for sid in sids:
            stid, title = B.stage_of(sid)
            if not stid:
                print(f"  {nm:<10}赛季{sid:<7} ⛔ 取不到 stid（仍被限流？）", flush=True)
                continue
            ms = B.season_matches(stid)
            new = 0
            for m in ms:
                if m["fid"] not in allm:
                    m["联赛"] = nm; allm[m["fid"]] = m; new += 1
            print(f"  {nm:<10}赛季{sid:<7}stid={stid:<7}{len(ms):>4} 场（新增 {new:>3}）  {title[:24]}",
                  flush=True)
            fp.write_text(json.dumps(sorted(allm.values(), key=lambda x: x["日期"]),
                                     ensure_ascii=False, indent=1), encoding="utf-8")

    rows = sorted(allm.values(), key=lambda x: x["日期"])
    from collections import Counter
    c = Counter(r["联赛"] for r in rows)
    print("\n" + "=" * 70)
    print(f"  修复后：{len(rows):,} 场 / {len(c)} 联赛 · 失败 {B.FAILS['n']} 次")
    print("=" * 70)
    for k, v in c.most_common():
        mark = " ★P1" if k in TARGETS else ""
        print(f"  {k:<12}{v:>6} 场{mark}")
    p1 = sum(v for k, v in c.items() if k in TARGETS)
    print(f"\n  P1 欧战常客合计 {p1:,} 场")
    return rows


if __name__ == "__main__":
    main()
