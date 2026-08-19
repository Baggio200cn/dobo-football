"""错误率归因诊断：是数据不够、权重没调好，还是题本身就难？

方法：拿同一批已开奖的比赛，做四个对照
  ① 我们的预测           实际命中
  ② 纯市场 argmax        同样这些场次，只看赔率会猜对几场
  ③ 市场理论上限         Σ max(市场概率) —— 任何只用公开信息的预测器的天花板
  ④ 全押主胜             最笨的基准

判读
  · 若 ① ≈ ② ≈ ③ 且都不高 → 题本身难，不是我们的问题，调权重/加数据都没用
  · 若 ① 明显 < ②        → 我们的融合把市场的判断弄坏了 → 权重问题
  · 若 ② 明显 < ③        → 市场概率虽准但 argmax 就是会错 → 不可约误差
"""
import sys, json, glob, re
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
from model import implied_1x2

CODE = {"H": "3", "D": "1", "A": "0"}


def odds_of(period):
    """从开奖 XML 取每场平均欧赔"""
    p = BASE / "cache_xml" / f"sfc_{period}.xml"
    raw = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
    if not raw:
        import build_l500 as B
        b = B._curl(f"https://kaijiang.500.com/static/info/kaijiang/xml/sfc/{period}.xml")
        raw = b.decode("utf-8", "replace") if b else None
    if not raw: return None
    try: r = ET.fromstring(raw)
    except Exception: return None
    out = {}
    for m in r.findall(".//MatchTeam"):
        a = m.attrib
        od = (a.get("AverageOdds") or "").split()
        if len(od) == 3:
            try: out[int(a["OrderNum"])] = tuple(float(x) for x in od)
            except Exception: pass
    return out


def main():
    rows = []
    for f in sorted(glob.glob(str(BASE / "batches" / "sfc_*_LOCK.json"))):
        L = json.loads(Path(f).read_text(encoding="utf-8"))
        per = L["期号"]
        res = L.get("实际开奖") or (L.get("预复盘_非官方") or {}).get("赛果")
        if not res: continue
        od = odds_of(per) or {}
        for i, g in enumerate(L["逐场"]):
            act = res[i]
            if act not in "310": continue          # 跳过未赛
            o = od.get(g["场次"])
            rows.append({"期": per, "场": g["场次"], "实际": act,
                         "我们": g["单选"], "置信": g["置信"],
                         "赔": o, "等级": L.get("数据等级", "A")})
    if not rows:
        print("无可用样本"); return

    print("=" * 78)
    print(f"  错误率归因诊断 · 样本 {len(rows)} 场（{len(set(r['期'] for r in rows))} 期）")
    print("=" * 78)

    with_od = [r for r in rows if r["赔"]]
    ours = np.mean([r["我们"] == r["实际"] for r in rows])

    # 纯市场 argmax + 理论上限
    mk_hit, mk_ceiling, home_hit = [], [], []
    for r in with_od:
        p = implied_1x2(*r["赔"])
        best = max(p, key=p.get)
        mk_hit.append(CODE[best] == r["实际"])
        mk_ceiling.append(max(p.values()))
        home_hit.append(r["实际"] == "3")
    ours_sub = np.mean([r["我们"] == r["实际"] for r in with_od])

    print(f"\n  {'对照':<28}{'命中率':>10}{'场次':>8}")
    print("  " + "-" * 48)
    print(f"  {'① 我们的预测':<28}{ours_sub:>10.1%}{len(with_od):>8}")
    print(f"  {'② 纯市场 argmax':<28}{np.mean(mk_hit):>10.1%}{len(with_od):>8}")
    print(f"  {'③ 市场理论上限 Σmax(p)':<28}{np.mean(mk_ceiling):>10.1%}{len(with_od):>8}")
    print(f"  {'④ 全押主胜':<28}{np.mean(home_hit):>10.1%}{len(with_od):>8}")
    print("  " + "-" * 48)

    d12 = ours_sub - np.mean(mk_hit)
    d23 = np.mean(mk_hit) - np.mean(mk_ceiling)
    print(f"\n  ① vs ②（我们 vs 市场）      {d12:+.1%}")
    print(f"  ② vs ③（市场实际 vs 上限）  {d23:+.1%}")

    from scipy import stats
    a = np.array([r["我们"] == r["实际"] for r in with_od], float)
    b = np.array(mk_hit, float)
    t, p = stats.ttest_rel(a, b)
    print(f"  ①②配对检验：t={t:.2f}  p={p:.3f}")

    print("\n" + "=" * 78)
    print("  判读")
    print("=" * 78)
    if p > 0.05:
        print(f"  · 我们与纯市场**无显著差异**（p={p:.3f}）。")
        print("    → 不是权重问题：把权重调成任何值，最好也就是市场那个水平。")
    elif d12 < 0:
        print(f"  · 我们**显著差于**纯市场（p={p:.3f}）→ 融合把市场判断弄坏了，是权重问题。")
    else:
        print(f"  · 我们**显著优于**纯市场（p={p:.3f}）。")
    print(f"  · 市场自己也只有 {np.mean(mk_hit):.1%}，理论上限 {np.mean(mk_ceiling):.1%}。")
    print(f"    → 即便拿到完美的公开信息概率，argmax 也只能到 {np.mean(mk_ceiling):.1%} 左右。")
    print(f"    这 {1-np.mean(mk_ceiling):.0%} 是**不可约误差**：足球本身的随机性。")

    # 按置信度分层：看我们的概率准不准（校准）
    print("\n" + "=" * 78)
    print("  我们的概率校准（说 X% 把握，实际中多少）")
    print("=" * 78)
    d = np.array([r["置信"] for r in rows]); y = np.array([r["我们"] == r["实际"] for r in rows], float)
    bins = [(0, .40), (.40, .45), (.45, .50), (.50, .55), (.55, .65), (.65, 1.0)]
    print(f"  {'置信区间':<14}{'场次':>6}{'我们说':>9}{'实际中':>9}{'偏差':>9}")
    print("  " + "-" * 48)
    for lo, hi in bins:
        m = (d >= lo) & (d < hi)
        if m.sum() < 3: continue
        print(f"  {f'{lo:.0%}–{hi:.0%}':<14}{m.sum():>6}{d[m].mean():>9.1%}{y[m].mean():>9.1%}"
              f"{y[m].mean()-d[m].mean():>+9.1%}")
    print("  " + "-" * 48)
    print(f"  {'合计':<14}{len(d):>6}{d.mean():>9.1%}{y.mean():>9.1%}{y.mean()-d.mean():>+9.1%}")

    # 平局专项
    nd = sum(1 for r in rows if r["实际"] == "1")
    lost_d = sum(1 for r in rows if r["实际"] == "1" and r["我们"] != "1")
    print(f"\n  平局：实际 {nd}/{len(rows)} = {nd/len(rows):.1%} · 我们全数错过 {lost_d} 场")
    print(f"  若把这 {lost_d} 场也算上，理论命中率可到 {(y.sum()+lost_d)/len(rows):.1%}"
          f"（但需要事先知道哪几场会平）")


if __name__ == "__main__":
    main()
