"""③ 真 xG 抓取（Understat 数据，经 GitHub 镜像 · 无需浏览器驱动）

来源：vaastav/Fantasy-Premier-League 仓库的 understat/ 目录
      每个球员一个 CSV，含逐场 xG / npxG / xA + 比赛标识(h_team,a_team,date,id)
做法：下载全部球员文件 → 按 (比赛id, 球队) 聚合球员 xG → 得到每场每队真 xG

⚠️ 局限：**仅英超**（FPL 仓库只覆盖 Premier League），约占我们数据 22%
用法：python fetch_xg.py
"""
import sys, io, json, time
import pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import BASE
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

REPO = "vaastav/Fantasy-Premier-League"
SEASONS = ["2023-24", "2024-25", "2025-26"]
RAW = "https://raw.githubusercontent.com/{}/master/data/{}/understat/{}"
API = "https://api.github.com/repos/{}/contents/data/{}/understat"
OUT = BASE / "xg_team_match.csv"
UA = {"User-Agent": "Mozilla/5.0"}


def list_files(season):
    try:
        r = requests.get(API.format(REPO, season), headers=UA, timeout=30)
        if r.status_code != 200: return []
        return [x["name"] for x in r.json() if x["name"].endswith(".csv")]
    except Exception:
        return []


def grab(season, name):
    try:
        r = requests.get(RAW.format(REPO, season, name), headers=UA, timeout=25)
        if r.status_code != 200: return None
        df = pd.read_csv(io.StringIO(r.text))
        need = {"xG", "h_team", "a_team", "date", "id", "h_goals", "a_goals"}
        if not need.issubset(df.columns): return None
        # 该球员属于主队还是客队？用 npxG 无法判断，需要 roster 信息 → 用 position 无效
        # 解法：understat 球员文件不直接标队，但可用「该球员所在队 = 出现最多的队」近似
        return df[list(need | {"npxG"} & set(df.columns))] if "npxG" in df.columns else df[list(need)]
    except Exception:
        return None


def main(workers=12):
    all_rows = []
    for season in SEASONS:
        files = list_files(season)
        if not files:
            print(f"  ⚠ {season}: 目录不可用，跳过"); continue
        print(f"  {season}: {len(files)} 个球员文件，下载中…", flush=True)
        got = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(grab, season, f): f for f in files}
            for i, fu in enumerate(as_completed(futs), 1):
                d = fu.result()
                if d is not None and len(d):
                    d = d.copy(); d["season"] = season
                    all_rows.append(d); got += 1
                if i % 150 == 0:
                    print(f"    …{i}/{len(files)}", flush=True)
        print(f"  {season}: 成功 {got}", flush=True)

    if not all_rows:
        print("❌ 未取到任何数据"); return None

    P = pd.concat(all_rows, ignore_index=True)
    P["date"] = pd.to_datetime(P["date"], errors="coerce").dt.date
    # 每场比赛的球员 xG 总和拆到两队：understat 球员文件无队标签，
    # 用「同一场里 xG 之和 ≈ 两队合计」→ 只能得到比赛总 xG，无法分队。
    # 因此这里输出：比赛总 xG（total_xg），仍是有用特征（衡量比赛开放度/机会质量）
    g = P.groupby(["id", "date", "h_team", "a_team", "h_goals", "a_goals"], as_index=False).agg(
        total_xg=("xG", "sum"),
        total_npxg=("npxG", "sum") if "npxG" in P.columns else ("xG", "sum"),
        n_players=("xG", "size"))
    g.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n✅ 聚合完成：{len(g)} 场（英超）→ {OUT.name}")
    print(f"   跨度 {g.date.min()} → {g.date.max()}")
    print(f"   场均总 xG {g.total_xg.mean():.2f} · 场均实际总进球 {(g.h_goals+g.a_goals).mean():.2f}")
    return g


if __name__ == "__main__":
    main()
