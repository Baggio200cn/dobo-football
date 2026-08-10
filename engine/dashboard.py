"""B · 私人看板：把引擎输出渲染成一个本地网页 dashboard.html（DoBo 深色风格）。
包含：模型对战(Brier/LogLoss) · 下一轮预测 · 最近复盘 · 成长台账。
用法：python dashboard.py  →  用浏览器打开 engine/dashboard.html
"""
import sys, glob, re, numpy as np, pandas as pd
from config import PRED, BASE
import evaluate
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

NAMES = {"elo": "Elo 评分", "poisson(进球)": "泊松·进球", "poisson-xG(射正)": "泊松·xG代理",
         "poisson-DC": "泊松·Dixon-Coles", "market": "市场(博彩基准)"}


def table(df, highlight_col=None):
    if df is None or not len(df): return "<p style='color:#5f6b7d'>暂无数据</p>"
    th = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for _, r in df.iterrows():
        tds = "".join(f"<td>{'' if pd.isna(v) else v}</td>" for v in r)
        rows += f"<tr>{tds}</tr>"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table>"


def model_bars(acc):
    rows = []
    stats = {k: (np.array(v)[:, 0].mean(), np.array(v)[:, 1].mean()) for k, v in acc.items() if v}
    best = min(s[0] for s in stats.values())
    worst = max(s[0] for s in stats.values())
    for k, (br, ll) in sorted(stats.items(), key=lambda x: x[1][0]):
        w = 100 * (worst - br) / (worst - best + 1e-9)          # 越低越好 → 条越长
        color = "#27d07a" if k == "market" else ("#f5c451" if k == "poisson-DC" else "#3f7dff")
        rows.append(
            f"<div class='bar'><div class='bl'>{NAMES.get(k,k)}</div>"
            f"<div class='bt'><div class='bf' style='width:{max(w,6):.0f}%;background:{color}'></div></div>"
            f"<div class='bv'>{br:.4f}<span>LL {ll:.3f}</span></div></div>")
    return "".join(rows)


def parse_ledger():
    fp = PRED / "LEDGER.md"
    if not fp.exists(): return None
    lines = [l for l in fp.read_text(encoding="utf-8").splitlines() if l.startswith("|")]
    if len(lines) < 3: return None
    hdr = [c.strip() for c in lines[0].strip("|").split("|")]
    data = [[c.strip() for c in l.strip("|").split("|")] for l in lines[2:]]
    return pd.DataFrame(data, columns=hdr)


def run():
    print("跑回测（几秒）…")
    acc = evaluate.run()

    preds = sorted(glob.glob(str(PRED / "pred_*.csv")))
    pred = pd.read_csv(preds[-1]) if preds else None
    if pred is not None:
        pred = pred[["联赛", "日期", "主队", "客队", "主胜", "平", "客胜", "倾向", "最可能比分", "大2.5"]]

    revs = sorted(glob.glob(str(PRED / "review_*.csv")))
    rev = pd.read_csv(revs[-1]) if revs else None
    rev_sum = ""
    if rev is not None and len(rev):
        hit = (rev["命中"] == "✓").mean(); br = rev["Brier"].mean()
        rev_sum = f"复盘 {len(rev)} 场 · 命中率 <b>{hit:.0%}</b> · 平均 Brier <b>{br:.3f}</b>"
        rev = rev[["联赛", "主队", "客队", "预测倾向", "信心", "实际", "实际结果", "命中", "翻车"]]

    ledger = parse_ledger()

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>足球预测复盘 · 私人看板 | DoBo</title>
<style>
:root{{--bg:#0a0e14;--card:#161c27;--card2:#1b2230;--line:#232c3b;--txt:#eef2f8;--mut:#8b97a8;--mut2:#5f6b7d;--gold:#f5c451;--green:#27d07a;--blue:#3f7dff}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--txt);font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:760px;margin:0 auto;padding:20px 16px 60px;line-height:1.5}}
.top{{display:flex;align-items:center;gap:10px;margin-bottom:4px}}
.top .t{{font-size:19px;font-weight:800}}.top .t b{{color:var(--gold)}}
.sub{{font-size:12px;color:var(--mut2);letter-spacing:1px;margin-bottom:18px}}
.beta{{margin-left:auto;font-size:10px;color:var(--mut2);border:1px solid var(--line);padding:3px 9px;border-radius:20px}}
h2{{font-size:15px;color:var(--gold);margin:26px 0 12px;display:flex;align-items:center;gap:8px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}}
.bar{{display:flex;align-items:center;gap:10px;margin:9px 0;font-size:13px}}
.bl{{width:120px;flex:0 0 auto;color:var(--txt)}}
.bt{{flex:1;height:12px;background:var(--card2);border-radius:6px;overflow:hidden}}
.bf{{height:100%;border-radius:6px}}
.bv{{width:110px;flex:0 0 auto;text-align:right;font-variant-numeric:tabular-nums;color:var(--txt)}}
.bv span{{color:var(--mut2);font-size:11px;margin-left:6px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th,td{{padding:8px 6px;border-bottom:1px solid var(--line);text-align:center;white-space:nowrap}}
th{{color:var(--mut);font-weight:600;font-size:11px}}
td{{color:var(--txt)}}
.scroll{{overflow-x:auto}}
.note{{font-size:11px;color:var(--mut2);line-height:1.7;margin-top:10px}}
.sum{{font-size:13px;color:var(--mut);margin-bottom:10px}}
.foot{{margin-top:30px;padding-top:16px;border-top:1px solid var(--line);font-size:11px;color:var(--mut2);line-height:1.8}}
</style></head><body>
<div class="top"><div class="t">Do<b>Bo</b> · 足球预测复盘看板</div><div class="beta">私测 · 自用</div></div>
<div class="sub">欧洲五大联赛 · Elo + 泊松 + Dixon-Coles · 数据 football-data.co.uk</div>

<h2>⚔️ 模型对战 · Brier（越短越好）</h2>
<div class="card">{model_bars(acc)}
<div class="note">Brier / LogLoss 越低越好。市场=Bet365 去水隐含概率，是最强基准；模型目标是持续逼近它并保持校准。</div></div>

<h2>🎯 下一轮预测</h2>
<div class="card"><div class="scroll">{table(pred)}</div>
<div class="note">⚠️ 仅数据分析与个人复盘，不构成投注建议。休赛期显示最近赛程做格式演示，新赛季自动切真实赛程。</div></div>

<h2>🔁 最近复盘</h2>
<div class="card"><div class="sum">{rev_sum or '暂无复盘（赛后运行 review.py 生成）'}</div>
<div class="scroll">{table(rev)}</div></div>

<h2>📈 成长台账</h2>
<div class="card"><div class="scroll">{table(ledger)}</div>
<div class="note">每复盘一轮追加一行，看命中率/Brier 的长期趋势——这才是真本事，别看单轮。</div></div>

<div class="foot">出品 · DoBo 体育 · @巴老师挨扯淡 · 私人自用看板<br>
仅数据分析与个人复盘，不构成任何投注建议、不推荐下注。数据以官方为准。</div>
</body></html>"""

    out = BASE / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n✅ 看板已生成：{out}")
    print("   用浏览器打开即可预览。")
    return out


if __name__ == "__main__":
    run()
