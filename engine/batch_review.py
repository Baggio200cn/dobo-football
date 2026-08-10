"""批次复盘 + 双轨短文生成

对照真实结果复盘某一期批次，并生成两个版本的短文：
  · article_internal_<期>.md  —— 内部版（模拟投注叙事，你自己看）
  · article_public_<期>.md    —— 对外版（模型批次叙事，可发头条，无彩票话术）

用法：
  python batch_review.py                      # 复盘最新一期
  python batch_review.py batch_2026-05-24.json
"""
import sys, json, glob
import pandas as pd
from pathlib import Path
from config import BASE
from data import load_matches
from export_web import ZH
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

BATCH = BASE / "batches"
ART = BASE / "articles"; ART.mkdir(exist_ok=True)
LEDGER = BATCH / "BATCH_LEDGER.json"


def latest():
    fs = sorted(glob.glob(str(BATCH / "batch_*.json")))
    return fs[-1] if fs else None


def run(fp=None):
    fp = Path(fp) if fp else (Path(latest()) if latest() else None)
    if not fp: raise SystemExit("没有批次。先 python batch.py")
    if not fp.is_absolute(): fp = BATCH / fp.name
    data = json.loads(fp.read_text(encoding="utf-8"))
    period, picks = data["period"], data["picks"]

    m = load_matches(); m["d"] = m["Date"].dt.date
    key = m.set_index(["d", "HomeTeam", "AwayTeam"])[["FTHG", "FTAG", "FTR"]]

    recs, hit = [], 0
    for p in picks:
        try:
            act = key.loc[(pd.to_datetime(p["日期"]).date(), p["主队EN"], p["客队EN"])]
        except KeyError:
            continue
        if isinstance(act, pd.DataFrame): act = act.iloc[0]
        ok = (p["选择码"] == act.FTR)
        hit += int(ok)
        recs.append({**p, "实际比分": f"{act.FTHG}-{act.FTAG}",
                     "实际": {"H": "主胜", "D": "平", "A": "客胜"}[act.FTR],
                     "结果": "✓" if ok else "✗"})

    if not recs:
        raise SystemExit(f"第 {period} 期比赛还没结果，赛后再复盘。")

    n = len(recs); rate = hit / n
    df = pd.DataFrame(recs)
    avg_edge = df["偏差"].mean()

    # 台账
    led = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []
    led = [x for x in led if x["period"] != period]
    led.append({"period": period, "n": n, "hit": hit, "rate": round(rate, 3),
                "avg_edge": round(float(avg_edge), 3)})
    led.sort(key=lambda x: x["period"])
    LEDGER.write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding="utf-8")

    trend = ""
    if len(led) >= 2:
        prev = led[-2]["rate"]
        d = rate - prev
        trend = f"较上期 {'↑' if d>0 else ('↓' if d<0 else '→')} {abs(d):.0%}"

    # ===== 内部版（你的投注叙事）=====
    lines = [f"# 第 {period} 期 · 模拟方案复盘（内部）\n",
             f"模拟 **{n}** 场 · 命中 **{hit}** 场 · 命中率 **{rate:.0%}** {trend}",
             f"平均偏差（模型−市场）**{avg_edge:+.3f}**\n", "## 逐场\n",
             df[["序号","联赛","主队","客队","选择","模型概率","市场概率","偏差","实际比分","实际","结果"]].to_markdown(index=False), "\n"]
    miss = df[df["结果"] == "✗"]
    if len(miss):
        lines.append("## 未命中归因（重点看）\n")
        for r in miss.itertuples():
            lines.append(f"- **{r.主队} vs {r.客队}**：模型选 {r.选择}({r.模型概率})，市场仅 {r.市场概率}，实际 {r.实际比分}（{r.实际}）。"
                         f"→ [ ] 模型高估 [ ] 数据缺失(伤停/轮换) [ ] 运气")
    lines.append("\n## 下期改进\n- [ ] 假设：____\n- [ ] 验证：命中率/Brier 是否改善\n")
    (ART / f"article_internal_{period}.md").write_text("\n".join(lines), encoding="utf-8")

    # ===== 对外版（安全叙事，可发头条）=====
    top = df.iloc[0]
    pub = [f'# 我让模型和博彩市场「打了一架」，第 {period} 轮结果出来了\n',
           f"这是我用 Elo + 泊松 + Dixon-Coles 搭的足球预测模型的**第 {len(led)} 轮公开验证记录**。",
           "规则很简单：**只挑模型和市场分歧最大的比赛**——因为只有在分歧处，才能看出模型到底是「看到了市场没看到的」，还是「自己错了」。\n",
           f"## 本轮结果：{n} 个高置信预测，命中 {hit} 个（{rate:.0%}）{(' · ' + trend) if trend else ''}\n",
           "| 比赛 | 模型判断 | 模型概率 | 市场概率 | 分歧 | 实际 | 结果 |",
           "|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        pub.append(f"| {r.主队} vs {r.客队} | {r.选择} | {r.模型概率:.0%} | {r.市场概率:.0%} | +{r.偏差:.3f} | {r.实际比分} | {'✅' if r.结果=='✓' else '❌'} |")
    pub.append("")
    if len(miss):
        pub.append("## 输在哪里\n")
        for r in miss.itertuples():
            pub.append(f"**{r.主队} vs {r.客队}**：模型认为{r.选择}有 {r.模型概率:.0%}，市场只给 {r.市场概率:.0%}，"
                       f"结果 {r.实际比分}。模型在这场上明显高估了——这正是下一轮要修的地方。\n")
    pub += ["## 这个记录的意义\n",
            "我不关心某一轮「猜中几个」——单轮全是噪声。我关心的是**这条曲线长期往哪走**：",
            "模型能不能一点点逼近市场的准确度，甚至在某些分歧上稳定地对。",
            "每一轮我都会公开：预测了什么、错在哪、下一轮改什么。\n",
            "---\n",
            "⚠️ **本文仅为机器学习模型的公开验证记录与数据分析**，不构成任何投注建议，不推荐任何形式的下注。\n"]
    (ART / f"article_public_{period}.md").write_text("\n".join(pub), encoding="utf-8")

    print(f"第 {period} 期复盘：{n} 场 · 命中 {hit} · {rate:.0%} {trend}")
    print(f"✅ 内部版：articles/article_internal_{period}.md")
    print(f"✅ 对外版：articles/article_public_{period}.md（可发头条）")
    return {"period": period, "n": n, "hit": hit, "rate": rate, "ledger": led}


if __name__ == "__main__":
    a = next((x for x in sys.argv[1:] if x.endswith(".json")), None)
    run(a)
