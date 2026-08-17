"""统一构建网站数据：把所有模块合并进 site/data.json

顺序：
  1. export_web.py      → 基础（Elo榜/回测/实验室档案/数据源）
  2. 合并预测锁定记录     → sfc_lock（当期）/ sfc_review（最近已复盘）/ sfc_ledger
  3. 合并胜负彩预测明细   → sfc
  4. 合并文章清单        → articles
  5. 写入 site/data.json

用法：python build_web.py
"""
import sys, json, glob, re
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE

SITE = Path(__file__).resolve().parent.parent / "site"
BAT = BASE / "batches"


def load(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


def main():
    # 1 基础数据
    import export_web
    export_web.main()
    web = load(BASE / "web_data.json", {})

    # 2 期次锁定：最新未复盘的当作"当期"，最新已复盘的当作"复盘"
    locks = sorted(f for f in glob.glob(str(BAT / "sfc_*_LOCK.json")))
    cur, rev = None, None
    for f in locks:
        d = load(f)
        if not d: continue
        if d.get("已复盘"): rev = d
        else: cur = d
    if cur: web["sfc_lock"] = cur
    if rev: web["sfc_review"] = rev
    led = load(BAT / "SFC_LEDGER.json", [])
    if led: web["sfc_ledger"] = led

    # 3 胜负彩预测明细（当期）
    if cur:
        det = load(BAT / f"sfc_{cur['期号']}.json")
        if det: web["sfc"] = det

    # 4 文章清单
    import build_articles
    arts = build_articles.main()
    web["articles"] = arts

    # 4.5 在售期次（实时赛程 + 投注截止时间）
    try:
        import fetch_onsale
        fetch_onsale.main()
    except Exception as e:
        print(f"  ⚠ 在售期次抓取失败（沿用上次结果）：{e}")

    # 4.6 自动预测：在售且尚未出票的期次，够格就出，不够格记入 SKIPPED
    try:
        import predict_any
        predict_any.main()
    except Exception as e:
        print(f"  ⚠ 自动预测失败：{e}")

    ons = load(BAT / "ONSALE.json", {})
    if ons.get("期次"):
        skip = {s["期号"]: s for s in load(BAT / "SKIPPED.json", {}).get("期次", [])}
        for p in ons["期次"]:
            # 预测可能刚刚生成，重新判定
            p["已预测"] = (BAT / f"sfc_{p['期号']}_LOCK.json").exists()
            if p["期号"] in skip and not p["已预测"]:
                s = skip[p["期号"]]
                p["跳过"] = {"原因": s["原因"], "覆盖": s["覆盖"],
                            "闸门": s["闸门"], "缺数据场次": s["缺数据场次"]}
        web["onsale"] = ons

    # 5 写出
    (BASE / "web_data.json").write_text(json.dumps(web, ensure_ascii=False, indent=1), encoding="utf-8")
    SITE.mkdir(exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(web, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 56)
    print("  网站数据构建完成")
    print("=" * 56)
    print(f"  训练库      {web.get('lab',{}).get('dataset',{}).get('n',0):,} 场")
    print(f"  当期预测    {cur['期号'] if cur else '—'}"
          + (f"  胜率 {cur['单式']['预测胜率']:.1%}" if cur else ""))
    print(f"  最近复盘    {rev['期号'] if rev else '—'}"
          + (f"  命中 {rev['复盘']['单式命中']}/14" if rev else ""))
    print(f"  台账        {len(led)} 期")
    print(f"  文章        {len(arts)} 篇")
    print(f"  → {SITE/'data.json'}")
    return web


if __name__ == "__main__":
    main()
