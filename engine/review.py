"""④ 赛后回填 + 复盘台账
把一份预测台账(predictions/pred_*.csv)与真实结果对照：
  命中率 / 每场 Brier / 校准 / 大小球命中 / 高信心翻车归因候选
产出：
  predictions/review_<date>.csv   逐场复盘
  predictions/review_<date>.md    复盘报告(含归因候选)
  predictions/LEDGER.md           累积成长台账(每复盘一轮追加一行)
用法：
  python review.py                 # 复盘最新一份 pred_*.csv
  python review.py pred_2026-05-24.csv
"""
import sys, glob, numpy as np, pandas as pd
from config import PRED
from data import load_matches
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

OUT = ["H", "D", "A"]
ZH = {"H": "主胜", "D": "平", "A": "客胜"}


def latest_pred():
    fs = sorted(glob.glob(str(PRED / "pred_*.csv")))
    return fs[-1] if fs else None


def run(pred_file=None):
    pred_file = pred_file or latest_pred()
    if not pred_file:
        raise SystemExit("没有预测台账。先 python predict.py")
    pred_file = PRED / pred_file if not str(pred_file).startswith(str(PRED)) else pred_file
    pred = pd.read_csv(pred_file)
    pred["日期"] = pd.to_datetime(pred["日期"], errors="coerce").dt.date

    # 真实结果
    m = load_matches()
    m["d"] = m["Date"].dt.date
    key = m.set_index(["d", "HomeTeam", "AwayTeam"])[["FTHG", "FTAG", "FTR"]]

    recs, matched = [], 0
    for _, p in pred.iterrows():
        try:
            act = key.loc[(p["日期"], p["主队"], p["客队"])]
        except KeyError:
            continue
        if isinstance(act, pd.DataFrame): act = act.iloc[0]
        matched += 1
        probs = {"H": p["主胜"], "D": p["平"], "A": p["客胜"]}
        y = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}[act.FTR]
        br = sum((probs[o] - y[i]) ** 2 for i, o in enumerate(OUT))
        pick = p["倾向"]
        hit = int(pick == act.FTR)
        pick_p = probs[pick]
        tot = int(act.FTHG + act.FTAG)
        ou_pred = "大" if p["大2.5"] > 0.5 else "小"
        ou_hit = int((ou_pred == "大") == (tot > 2))
        recs.append({
            "联赛": p["联赛"], "日期": p["日期"], "主队": p["主队"], "客队": p["客队"],
            "预测倾向": ZH[pick], "信心": round(pick_p, 3),
            "实际": f"{act.FTHG}-{act.FTAG}", "实际结果": ZH[act.FTR],
            "命中": "✓" if hit else "✗", "Brier": round(br, 3),
            "大小球预测": ou_pred, "大小球命中": "✓" if ou_hit else "✗",
            "翻车": "⚠高信心翻车" if (not hit and pick_p >= 0.55) else "",
        })

    if not matched:
        raise SystemExit("这份预测里的比赛还没有结果（未来赛程，等赛后再复盘）。")

    df = pd.DataFrame(recs)
    stamp = df["日期"].max()
    hit_rate = df["命中"].eq("✓").mean()
    brier = df["Brier"].mean()
    ou_rate = df["大小球命中"].eq("✓").mean()
    flops = df[df["翻车"] != ""]

    # 逐场 CSV
    csv_fp = PRED / f"review_{stamp}.csv"
    df.to_csv(csv_fp, index=False, encoding="utf-8-sig")

    # 复盘报告 md
    md = [f"# 复盘报告 · {stamp}\n",
          f"- 复盘场次：**{matched}**",
          f"- 胜平负命中率：**{hit_rate:.0%}**",
          f"- 平均 Brier：**{brier:.3f}**（越低越好）",
          f"- 大小球命中率：**{ou_rate:.0%}**\n",
          "## 逐场\n", df.to_markdown(index=False), "\n"]
    if len(flops):
        md.append("## ⚠ 高信心翻车（归因候选 · 重点看这些）\n")
        for f in flops.itertuples():
            md.append(f"- **{f.主队} vs {f.客队}**：判 {f.预测倾向}({f.信心}) → 实际 {f.实际}（{f.实际结果}）。"
                      f"归因？→ [ ] 数据(伤停/轮换) [ ] 模型(系统性偏差) [ ] 运气(xG占优却输)")
    md.append("\n## 下一步改进（本轮结论 → 下轮改一个假设）\n- [ ] 假设：____\n- [ ] 验证：Brier 是否下降\n")
    md_fp = PRED / f"review_{stamp}.md"
    md_fp.write_text("\n".join(md), encoding="utf-8")

    # 累积成长台账
    ledger = PRED / "LEDGER.md"
    head = "| 复盘日 | 场次 | 命中率 | 平均Brier | 大小球命中 | 高信心翻车 |\n|---|---|---|---|---|---|\n"
    line = f"| {stamp} | {matched} | {hit_rate:.0%} | {brier:.3f} | {ou_rate:.0%} | {len(flops)} |\n"
    if not ledger.exists():
        ledger.write_text("# 复盘成长台账（每轮一行，看趋势）\n\n" + head + line, encoding="utf-8")
    else:
        txt = ledger.read_text(encoding="utf-8")
        if f"| {stamp} |" not in txt:   # 幂等：同一天不重复追加
            ledger.write_text(txt + line, encoding="utf-8")

    print(f"复盘 {matched} 场 · 命中率 {hit_rate:.0%} · Brier {brier:.3f} · 大小球 {ou_rate:.0%} · 翻车 {len(flops)} 场")
    print(f"✅ 逐场：{csv_fp.name} · 报告：{md_fp.name} · 台账：LEDGER.md")
    if len(flops):
        print("⚠ 高信心翻车：", ", ".join(f"{f.主队}({f.信心})" for f in flops.itertuples()))
    return df


if __name__ == "__main__":
    arg = next((a for a in sys.argv[1:] if a.endswith(".csv")), None)
    run(arg)
