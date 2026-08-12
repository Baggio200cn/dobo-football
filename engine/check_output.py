"""产出自检：CI 用。缺关键字段就退出码 1，让工作流失败而不是静默通过。"""
import sys, json
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

SITE = Path(__file__).resolve().parent.parent / "site"
fp = SITE / "data.json"

if not fp.exists():
    print(f"❌ 缺 {fp}"); sys.exit(1)

d = json.loads(fp.read_text(encoding="utf-8"))
errs = []
if not d.get("sfc_lock"):  errs.append("缺当期预测 sfc_lock")
if not d.get("articles"):  errs.append("缺文章清单 articles")
if not d.get("lab", {}).get("dataset", {}).get("n"): errs.append("缺训练集档案")

arts = SITE / "articles"
md = list(arts.glob("*.md")) if arts.exists() else []
if not md: errs.append("site/articles 无 .md 文件")

if errs:
    print("❌ 自检不通过：")
    for e in errs: print("  -", e)
    sys.exit(1)

L = d["sfc_lock"]
print("✅ 自检通过")
print(f"   当期预测 {L['期号']} · 胜率 {L['单式']['预测胜率']:.1%} · 锁定 {L.get('锁定时间','')[:16]}")
if d.get("sfc_review"):
    R = d["sfc_review"]
    print(f"   已复盘   {R['期号']} · 命中 {R['复盘']['单式命中']}/14")
print(f"   文章     {len(d['articles'])} 篇 · {len(md)} 个 .md 文件")
print(f"   训练库   {d['lab']['dataset']['n']:,} 场")
