"""扩展联赛 Elo 池（以 500 team ID 为主键）

与主 Elo 表（football-data 英文名）并行的第二个池子，专供
football-data 未覆盖的联赛（克罗地亚/捷克/保加利亚/斯洛伐克…）。

为什么不合并进主表：
  两边球队从无交手，Elo 分数不可比（同 1500 分在不同池子里含义不同）。
  跨池对局（欧战）走 UEFA 国家系数校准，见 euro.league_offset。

为什么用 team ID 而非队名：
  500 站内同一支队有两套名字（联赛页「依斯特拉」/ 赛程 JSON「伊斯特拉1961」），
  按名字建池会把一支队拆成两支。而 team ID 全站统一，
  且**与胜负彩赛程共用同一套 ID** —— 这正是选 500 作数据源的核心理由。

产出 batches/L500_ELO.json：
  {"elo": {teamId: 分数}, "name": {teamId: 名称}, "league": {teamId: 联赛},
   "gt": {teamId: 近20场场均总进球}, "n": {teamId: 参赛场次}}
"""
import sys, json
from collections import defaultdict, deque
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE

K, HFA = 20.0, 60.0
MIN_MATCH = 10          # 少于这么多场的球队视为「数据不足」，不参与实测判定


def build(rows):
    R, name, lg, cnt = {}, {}, {}, defaultdict(int)
    gf = defaultdict(lambda: deque(maxlen=20))
    for m in sorted(rows, key=lambda x: x["日期"] or ""):
        h, a = str(m["主ID"]), str(m["客ID"])
        if not h or not a or m["主进"] is None: continue
        name[h] = m["主队"]; name[a] = m["客队"]
        lg[h] = lg[a] = m["联赛"]
        rh, ra = R.get(h, 1500.0), R.get(a, 1500.0)
        eh = 1 / (1 + 10 ** (-(rh + HFA - ra) / 400))
        s = 1.0 if m["主进"] > m["客进"] else (0.5 if m["主进"] == m["客进"] else 0.0)
        R[h] = rh + K * (s - eh)
        R[a] = ra + K * ((1 - s) - (1 - eh))
        cnt[h] += 1; cnt[a] += 1
        g = m["主进"] + m["客进"]
        gf[h].append(g); gf[a].append(g)
    gt = {t: float(np.mean(v)) for t, v in gf.items() if len(v) >= 5}
    return R, name, lg, gt, cnt


def main():
    fp = BASE / "batches" / "L500_MATCHES.json"
    if not fp.exists():
        print("⛔ L500_MATCHES.json 不存在，先跑 build_l500 / repair_l500 / consolidate_l500")
        return
    rows = json.loads(fp.read_text(encoding="utf-8"))
    R, name, lg, gt, cnt = build(rows)

    solid = {t for t, c in cnt.items() if c >= MIN_MATCH}
    out = {"elo": {t: round(v, 1) for t, v in R.items()},
           "name": name, "league": lg,
           "gt": {t: round(v, 2) for t, v in gt.items()},
           "n": dict(cnt), "可用": sorted(solid), "MIN_MATCH": MIN_MATCH}
    op = BASE / "batches" / "L500_ELO.json"
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    from collections import Counter
    byl = defaultdict(list)
    for t in R: byl[lg.get(t, "?")].append(t)
    print("=" * 74)
    print(f"  扩展 Elo 池：{len(rows):,} 场 → {len(R)} 支球队")
    print(f"  其中 ≥{MIN_MATCH} 场（可用于实测判定）：{len(solid)} 支（{len(solid)/max(len(R),1):.0%}）")
    print("=" * 74)
    print(f"  {'联赛':<12}{'球队':>5}{'可用':>5}{'场均':>7}   {'最强 3 队'}")
    for l in sorted(byl, key=lambda x: -len(byl[x])):
        ts = byl[l]
        ok = [t for t in ts if t in solid]
        avg = np.mean([cnt[t] for t in ts])
        top = sorted(ts, key=lambda t: -R[t])[:3]
        s = "、".join(f"{name[t]}({R[t]:.0f})" for t in top)
        print(f"  {l:<12}{len(ts):>5}{len(ok):>5}{avg:>7.1f}   {s}")
    print(f"\n  → {op.name}")
    return out


if __name__ == "__main__":
    main()
