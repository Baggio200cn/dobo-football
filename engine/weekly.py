"""每周一键：自动判断当前该做什么并执行

逻辑：
  1) 有未复盘的旧批次 且 结果已出 → 先复盘（出短文素材）
  2) 有新赛程 → 生成本轮预测批次
  3) 都没有 → 报告赛季倒计时与就绪状态

用法：
  python weekly.py            # 自动判断并执行
  python weekly.py --check    # 只检查状态，不执行
"""
import sys, json, glob, datetime as dt
from pathlib import Path
import pandas as pd
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE, LEAGUES
import data as D

# 2026-27 五大联赛开赛日
OPENING = {
    "SP1": ("西甲", dt.date(2026, 8, 16)),
    "E0":  ("英超", dt.date(2026, 8, 22)),
    "I1":  ("意甲", dt.date(2026, 8, 23)),
    "F1":  ("法甲", dt.date(2026, 8, 23)),
    "D1":  ("德甲", dt.date(2026, 8, 28)),
}
BATCH_DIR = BASE / "batches"


def today():
    return dt.date.today()


def check():
    """返回 (状态字典) 并打印人类可读报告"""
    st = {}
    t = today()
    print("=" * 56)
    print(f"  足球预测引擎 · 状态检查   {t}")
    print("=" * 56)

    # 1 赛季倒计时
    print("\n【赛季日程】")
    started = []
    for div, (zh, d) in sorted(OPENING.items(), key=lambda kv: kv[1][1]):
        days = (d - t).days
        if days > 0:
            print(f"  {zh:<4} {d}  还有 {days:>2} 天")
        else:
            print(f"  {zh:<4} {d}  已开赛 ✅")
            started.append(div)
    st["started"] = started

    # 2 赛程可用性
    print("\n【赛程数据】")
    try:
        D.update(refresh=True)
        fx = D.load_fixtures()
    except Exception as e:
        fx = pd.DataFrame(); print(f"  ⚠ 拉取失败: {e}")
    st["fixtures"] = len(fx)
    if len(fx):
        print(f"  ✅ 五大联赛未来赛程 {len(fx)} 场")
        if "Date" in fx:
            print(f"     日期范围 {pd.to_datetime(fx.Date).min().date()} → {pd.to_datetime(fx.Date).max().date()}")
        for div, n in fx.groupby("Div").size().items():
            print(f"     {LEAGUES.get(div, div)}: {n} 场")
    else:
        print("  ⏳ 暂无五大联赛赛程（通常开赛前 3-5 天发布）")

    # 3 待复盘批次
    print("\n【预测批次】")
    # 注意：Windows glob 大小写不敏感，需排除 BATCH_LEDGER.json
    files = sorted(f for f in glob.glob(str(BATCH_DIR / "batch_*.json"))
                   if Path(f).name.startswith("batch_"))
    pending = None
    if not files:
        print("  尚无批次")
    else:
        m = D.load_matches(); m["d"] = m["Date"].dt.date
        key = set(zip(m["d"], m["HomeTeam"], m["AwayTeam"]))
        for f in files:
            b = json.loads(Path(f).read_text(encoding="utf-8"))
            done = sum(1 for p in b["picks"]
                       if (pd.to_datetime(p["日期"]).date(), p["主队EN"], p["客队EN"]) in key)
            tag = "✅ 已可复盘" if done == len(b["picks"]) else f"⏳ {done}/{len(b['picks'])} 有结果"
            print(f"  {Path(f).stem}  {tag}")
            if done == len(b["picks"]) and not b.get("reviewed"):
                pending = f
        st["pending_review"] = pending

    # 4 建议动作
    print("\n【建议动作】")
    if pending:
        print(f"  → 复盘：python batch_review.py")
    elif len(fx):
        print(f"  → 出预测：python batch.py")
    else:
        nearest = min(OPENING.values(), key=lambda x: abs((x[1] - t).days))
        print(f"  → 等待赛程发布（最近：{nearest[0]} {nearest[1]}）")
        print(f"     赛程一出，本脚本会自动切到「出预测」")
    print()
    return st


def main():
    st = check()
    if "--check" in sys.argv:
        return
    if st.get("pending_review"):
        print(">>> 执行复盘\n")
        import batch_review; batch_review.main()
    elif st.get("fixtures"):
        print(">>> 生成预测批次\n")
        import batch; batch.main()
    else:
        print("（无可执行动作，等赛程发布）")


if __name__ == "__main__":
    main()
