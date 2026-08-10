"""⑤ 主力缺阵指数（伤停 / 停赛 / 轮换的综合代理）

数据源：FPL 官方 gameweek 快照（GitHub 镜像，无需 key）
        每球员每轮的 minutes / starts —— 可反推「谁没上场」

核心指标：对第 N 轮，取该队在前 6 轮出场时间最多的 11 人为「常规主力」，
          统计他们在**第 N−1 轮**有几人 0 分钟 → 缺阵指数

⚠️ 防前视：特征用的是「上一轮」的缺阵，不是本场首发（本场首发赛前不可知）。
   伤病有持续性，上一轮缺阵是对本场可用性的合理代理。

⚠️ 局限：仅英超（FPL 只覆盖 Premier League）
用法：python fetch_availability.py
"""
import sys, io, urllib.request
from collections import defaultdict
import numpy as np, pandas as pd
from config import BASE
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

SEASONS = ["2023-24", "2024-25", "2025-26", "2026-27"]
URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{}/gws/merged_gw.csv"
OUT = BASE / "availability.csv"

# FPL 队名 → football-data 队名
NAME_MAP = {
    "Man Utd": "Man United", "Spurs": "Tottenham", "Nott'm Forest": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United", "Luton": "Luton", "Wolves": "Wolves",
    "Newcastle": "Newcastle", "Man City": "Man City", "Brighton": "Brighton",
}


def fetch(season):
    try:
        r = urllib.request.urlopen(URL.format(season), timeout=60)
        df = pd.read_csv(io.StringIO(r.read().decode("utf-8", errors="replace")))
        need = {"name", "team", "GW", "minutes", "kickoff_time"}
        if not need.issubset(df.columns):
            print(f"  ⚠ {season}: 缺列，跳过"); return None
        df["season"] = season
        return df[["name", "team", "GW", "minutes", "kickoff_time", "season"]]
    except Exception as e:
        print(f"  ⚠ {season}: {e}"); return None


def main(window=6, topn=11):
    frames = []
    for s in SEASONS:
        d = fetch(s)
        if d is not None:
            print(f"  {s}: {len(d)} 行 · GW {d.GW.min()}–{d.GW.max()}", flush=True)
            frames.append(d)
    if not frames:
        print("❌ 无数据"); return None
    P = pd.concat(frames, ignore_index=True)
    P["team"] = P["team"].replace(NAME_MAP)
    P["date"] = pd.to_datetime(P["kickoff_time"], errors="coerce", utc=True).dt.date

    rows = []
    for (season, team), g in P.groupby(["season", "team"]):
        g = g.sort_values("GW")
        gws = sorted(g.GW.unique())
        # 每轮：球员 → 分钟
        by_gw = {gw: dict(zip(sub.name, sub.minutes)) for gw, sub in g.groupby("GW")}
        gw_date = g.groupby("GW")["date"].first().to_dict()
        for i, gw in enumerate(gws):
            if i < 2: continue                       # 前两轮无历史，跳过
            hist = [x for x in gws[max(0, i - window):i]]   # 前 window 轮（不含本轮）
            tot = defaultdict(float)
            for h in hist:
                for pl, mn in by_gw.get(h, {}).items():
                    tot[pl] += float(mn or 0)
            if not tot: continue
            regulars = [p for p, _ in sorted(tot.items(), key=lambda kv: -kv[1])[:topn]]
            prev = by_gw.get(gws[i - 1], {})          # ★ 上一轮（赛前可知）
            absent = sum(1 for p in regulars if float(prev.get(p, 0) or 0) == 0)
            # 缺阵权重：按其历史分钟占比
            wtot = sum(tot[p] for p in regulars) or 1
            wabs = sum(tot[p] for p in regulars if float(prev.get(p, 0) or 0) == 0) / wtot
            rows.append({"season": season, "team": team, "GW": gw,
                         "date": gw_date.get(gw), "absent_n": absent,
                         "absent_w": round(wabs, 4)})

    A = pd.DataFrame(rows).dropna(subset=["date"])
    A.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n✅ 缺阵指数：{len(A)} 条（队×轮）→ {OUT.name}")
    print(f"   跨度 {A.date.min()} → {A.date.max()} · 球队 {A.team.nunique()}")
    print(f"   场均缺阵主力 {A.absent_n.mean():.2f} 人（分钟权重 {A.absent_w.mean():.1%}）")
    print(f"   分布: {A.absent_n.value_counts().sort_index().head(8).to_dict()}")
    return A


if __name__ == "__main__":
    main()
