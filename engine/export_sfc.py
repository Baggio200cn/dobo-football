"""导出胜负彩数据到网站：预测板块 + 数据集来源 + 训练过程 + 训练结果"""
import sys, json, datetime as dt
import numpy as np, pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE, ALL_LEAGUES, LEAGUES, NEW_LEAGUES, SEASONS, FD_BASE
import data as D
from model import elo_run

OUT = BASE / "web_data.json"


def main():
    m = D.load_matches()
    r, _ = elo_run(m)

    # ===== 数据集来源列表（实时统计）=====
    src = []
    for div, g in m.groupby("Div"):
        is_new = div in NEW_LEAGUES
        src.append({
            "code": div,
            "name": ALL_LEAGUES.get(div, div),
            "n": int(len(g)),
            "from": str(g.Date.min().date()),
            "to": str(g.Date.max().date()),
            "teams": int(len(set(g.HomeTeam) | set(g.AwayTeam))),
            "odds": bool(g["B365H"].notna().mean() > 0.5) if "B365H" in g else False,
            "url": f"{FD_BASE}/new/{div}.csv" if is_new else f"{FD_BASE}/mmz4281/2526/{div}.csv",
            "fmt": "new" if is_new else "main",
        })
    src.sort(key=lambda x: -x["n"])

    # ===== 训练过程（管线步骤 + 实时数值）=====
    n_teams = len(r)
    elo_vals = np.array(list(r.values()))
    pipeline = [
        {"step": 1, "name": "数据采集", "detail": f"{len(src)} 个联赛 CSV 自动下载",
         "metric": f"{len(m):,} 场"},
        {"step": 2, "name": "清洗与对齐", "detail": "统一列名 / 解析日期 / 剔除缺失",
         "metric": f"{m.Date.min().date()} → {m.Date.max().date()}"},
        {"step": 3, "name": "Elo 滚动训练", "detail": "按时间顺序逐场更新评分（K=20，主场 +60）",
         "metric": f"{n_teams} 支球队"},
        {"step": 4, "name": "市场概率解析", "detail": "赔率去水 → 隐含概率",
         "metric": f"{int(m['B365H'].notna().sum()):,} 场有赔率"},
        {"step": 5, "name": "融合预测", "detail": "0.55×市场 + 0.45×Elo（无赔率时纯 Elo）",
         "metric": "14 场输出"},
    ]

    # ===== 训练结果（Elo 分布 + 各联赛强度）=====
    league_str = []
    for div, g in m.groupby("Div"):
        teams = set(g[g.Date >= g.Date.max() - pd.Timedelta(days=400)].HomeTeam)
        vals = [r[t] for t in teams if t in r]
        if len(vals) >= 5:
            league_str.append({"name": ALL_LEAGUES.get(div, div), "code": div,
                               "avg": round(float(np.mean(vals))), "max": round(float(np.max(vals))),
                               "n": len(vals)})
    league_str.sort(key=lambda x: -x["max"])

    top = sorted(r.items(), key=lambda kv: -kv[1])[:15]
    result = {
        "n_teams": n_teams,
        "elo_mean": round(float(elo_vals.mean())),
        "elo_std": round(float(elo_vals.std())),
        "elo_max": round(float(elo_vals.max())),
        "elo_min": round(float(elo_vals.min())),
        "top": [{"team": t, "elo": round(v)} for t, v in top],
        "leagues": league_str[:16],
        "hist": [int(x) for x in np.histogram(elo_vals, bins=10, range=(1200, 1900))[0]],
    }

    # ===== 胜负彩预测 =====
    fp = BASE / "batches" / "sfc_26101.json"
    sfc = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None

    web = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    web.update({
        "updated": str(m.Date.max().date()),
        "matches": int(len(m)),
        "span": f"{m.Date.min().date()} → {m.Date.max().date()}",
        "n_leagues": int(m.Div.nunique()),
        "sources": src,
        "pipeline": pipeline,
        "train_result": result,
        "sfc": sfc,
        "generated": str(dt.datetime.now())[:19],
    })
    # 更新实验室档案里的训练集描述
    if "lab" in web:
        web["lab"]["dataset"].update({
            "n": int(len(m)),
            "span": f"{m.Date.min().date()} → {m.Date.max().date()}",
            "leagues": f"{m.Div.nunique()} 个联赛（含荷甲/葡超/北欧/英格兰各级）",
        })
    OUT.write_text(json.dumps(web, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✅ {len(m):,} 场 · {len(src)} 联赛 · {n_teams} 队 · 胜负彩 {'已载入' if sfc else '无'}")
    print(f"   Elo 均值 {result['elo_mean']} ± {result['elo_std']} · 最高 {top[0][0]} {round(top[0][1])}")
    return web


if __name__ == "__main__":
    main()
