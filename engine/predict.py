"""预测：用全部数据拟合，对未来赛程输出概率 + 市场对比 + edge，写入预测台账（复盘用）。
休赛期无赛程时，用测试样例演示格式。"""
import sys, datetime as dt, numpy as np, pandas as pd
from config import LEAGUES, PRED
from data import load_matches, load_fixtures
from model import Poisson, elo_run, elo_1x2, implied_1x2
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass


def _row(div, date, home, away, pp, pe, pm):
    # 融合：泊松与 Elo 的胜平负取平均（简单集成）
    blend = {o: (pp[o] + pe[o]) / 2 for o in ("H", "D", "A")}
    pick = max(blend, key=blend.get)
    edge = ""
    if pm:
        e = {o: blend[o] - pm[o] for o in ("H", "D", "A")}
        best = max(e, key=e.get)
        if e[best] > 0.03:  # 只标记正边际 >3%
            edge = f"{best}+{e[best]*100:.0f}%"
    return {
        "联赛": LEAGUES.get(div, div), "日期": date, "主队": home, "客队": away,
        "主胜": round(blend["H"], 3), "平": round(blend["D"], 3), "客胜": round(blend["A"], 3),
        "倾向": pick, "最可能比分": pp["top_score"],
        "大2.5": round(pp["over25"], 3), "小2.5": round(pp["under25"], 3),
        "市场主胜": round(pm["H"], 3) if pm else "", "边际": edge,
    }


def run():
    m = load_matches()
    pois = Poisson().fit(m)
    r, _ = elo_run(m)

    fx = load_fixtures()
    rows = []
    if len(fx):
        for f in fx.itertuples():
            h, a = f.HomeTeam, f.AwayTeam
            pp = pois.predict(h, a)
            pe = elo_1x2(r.get(h, 1500), r.get(a, 1500))
            pm = implied_1x2(getattr(f, "B365H", None), getattr(f, "B365D", None), getattr(f, "B365A", None))
            rows.append(_row(f.Div, f.Date.date() if pd.notna(f.Date) else "", h, a, pp, pe, pm))
        src = "未来赛程"
    else:
        # 休赛期：用最近 8 场真实对阵做“格式演示”
        demo = m.tail(8)
        for m_ in demo.itertuples():
            pp = pois.predict(m_.HomeTeam, m_.AwayTeam)
            pe = elo_1x2(r.get(m_.HomeTeam, 1500), r.get(m_.AwayTeam, 1500))
            rows.append(_row(m_.Div, m_.Date.date(), m_.HomeTeam, m_.AwayTeam, pp, pe, None))
        src = "⚠ 休赛期无赛程 → 用最近 8 场做格式演示（赛季开始后自动切真实赛程）"

    out = pd.DataFrame(rows)
    today = load_matches()["Date"].max().date()  # 用数据最新日期占位（无 Date.now 依赖）
    fp = PRED / f"pred_{today}.csv"
    out.to_csv(fp, index=False, encoding="utf-8-sig")
    print(f"来源：{src}\n")
    print(out.to_string(index=False))
    print(f"\n✅ 预测台账已写入：{fp.name}（复盘时对照实际结果用）")
    return out


if __name__ == "__main__":
    run()
