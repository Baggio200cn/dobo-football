"""每期"模拟批次"生成器（核心：模型 vs 市场偏差最大的场次）

逻辑：
  1. 对候选场次算模型概率（Elo + 泊松-DC 融合）
  2. 取市场隐含概率（赔率去水）
  3. edge = 模型概率 - 市场概率，按 edge 排序
  4. 取 TOP N 作为本期"高置信/模拟投注"批次
  5. 写入 batches/batch_<期号>.csv + .json（供网站与复盘用）

诚实说明：edge 大不代表一定赢，只代表"模型与市场分歧大"。
分歧正是检验模型的地方——赢了说明模型看到了市场没看到的，输了说明模型有偏差。
用法：
  python batch.py            # 用最近一轮真实赛程（休赛期用最近比赛做演示）
  python batch.py --n 5      # 指定选几场
"""
import sys, json, glob
import numpy as np, pandas as pd
from pathlib import Path
from config import LEAGUES, BASE
from data import load_matches, load_fixtures
from model import Poisson, elo_run, elo_1x2, implied_1x2
from export_web import ZH
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

BATCH = BASE / "batches"; BATCH.mkdir(exist_ok=True)
ZHO = {"H": "主胜", "D": "平", "A": "客胜"}


def build(n=5):
    m = load_matches()
    pois = Poisson().fit(m).fit_rho(m)
    r, _ = elo_run(m)

    fx = load_fixtures()
    if len(fx):
        cand = [(f.Div, f.Date, f.HomeTeam, f.AwayTeam,
                 getattr(f, "B365H", None), getattr(f, "B365D", None), getattr(f, "B365A", None))
                for f in fx.itertuples()]
        src = "未来赛程"
    else:
        demo = m.tail(20)
        cand = [(x.Div, x.Date, x.HomeTeam, x.AwayTeam, x.B365H, x.B365D, x.B365A)
                for x in demo.itertuples()]
        src = "休赛期 · 用最近 20 场做演示（新赛季自动切真实赛程）"

    rows = []
    for div, date, h, a, oh, od, oa in cand:
        pp = pois.predict(h, a)
        pe = elo_1x2(r.get(h, 1500), r.get(a, 1500))
        blend = {o: (pp[o] + pe[o]) / 2 for o in ("H", "D", "A")}
        mk = implied_1x2(oh, od, oa)
        if not mk:   # 没赔率就无法算偏差，跳过
            continue
        edges = {o: blend[o] - mk[o] for o in ("H", "D", "A")}
        pick = max(edges, key=edges.get)
        rows.append({
            "联赛": LEAGUES.get(div, div),
            "日期": str(pd.to_datetime(date).date()),
            "主队": ZH.get(h, h), "客队": ZH.get(a, a),
            "主队EN": h, "客队EN": a,
            "选择": ZHO[pick], "选择码": pick,
            "模型概率": round(blend[pick], 3),
            "市场概率": round(mk[pick], 3),
            "偏差": round(edges[pick], 3),
            "预计比分": pp["top_score"],
        })

    if not rows:
        raise SystemExit("没有可用场次（缺赔率数据）。")

    df = pd.DataFrame(rows).sort_values("偏差", ascending=False).reset_index(drop=True)
    top = df.head(n).copy()
    top.insert(0, "序号", range(1, len(top) + 1))

    # 期号：用最新日期
    period = str(pd.to_datetime(df["日期"]).max().date())
    top.to_csv(BATCH / f"batch_{period}.csv", index=False, encoding="utf-8-sig")
    payload = {
        "period": period, "source": src, "n": len(top),
        "created_from_matches": int(len(m)),
        "picks": top.to_dict(orient="records"),
    }
    (BATCH / f"batch_{period}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"来源：{src}")
    print(f"\n第 {period} 期 · 模型 vs 市场偏差 TOP{len(top)}：\n")
    print(top[["序号", "联赛", "日期", "主队", "客队", "选择", "模型概率", "市场概率", "偏差", "预计比分"]].to_string(index=False))
    print(f"\n✅ 已写入 batches/batch_{period}.csv / .json")
    print("⚠️ 偏差大 = 模型与市场分歧大，用于检验模型，不构成任何投注建议。")
    return payload


if __name__ == "__main__":
    n = 5
    if "--n" in sys.argv:
        try: n = int(sys.argv[sys.argv.index("--n") + 1])
        except Exception: pass
    build(n)
