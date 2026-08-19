"""逆向策略的最优强度：反几场最划算？

背景（已实证，n=61 期，p<0.0001）
    热门命中数 vs 一等奖注数   ρ = +0.78
    热门命中数 vs 单注奖金     ρ = -0.75
  → 人群跟随热门。赛果越冷门，中奖人越少，单注奖金越高。

但「跟人群反着来」不能直接执行：反向同时压低中奖概率。
真正的问题是最优强度 —— 反几场时 期望赔付 = P(中奖) × 奖金 最大。

模型
  · P(全中)：把 k 场从首选换成次选，P 乘以 ∏(p2/p1)
  · 奖金：对 log(单注奖金) ~ 热门命中数 做回归，用于外推
  · 500 万封顶必须计入，否则会高估逆向收益（冷门期大量顶格）

⚠ 这是研究性测算，不是投注建议。足彩返奖率 65%，长期期望为负。
"""
import sys, re
import numpy as np
import xml.etree.ElementTree as ET
from scipy import stats
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
from model import implied_1x2

CODE = {"H": "3", "D": "1", "A": "0"}
CAP = 5_000_000


def load():
    out = []
    for f in sorted((BASE / "cache_xml").glob("sfc_*.xml")):
        try: r = ET.fromstring(f.read_text(encoding="utf-8", errors="replace"))
        except Exception: continue
        g = lambda t: (r.findtext(t) or "").strip()
        res = "".join(g("Result").split(","))
        if len(res) != 14: continue
        num = lambda t: int(re.sub(r"[^\d]", "", g(t) or "0") or 0)
        if not num("Num1"): continue
        P, ok = [], True
        for m in sorted(r.findall(".//MatchTeam"), key=lambda x: int(x.attrib.get("OrderNum", 0))):
            od = (m.attrib.get("AverageOdds") or "").split()
            if len(od) != 3: ok = False; break
            P.append(implied_1x2(*[float(x) for x in od]))
        if not ok or len(P) != 14: continue
        fav = sum(1 for i, p in enumerate(P) if CODE[max(p, key=p.get)] == res[i])
        out.append({"期": g("PeriodicalNO"), "P": P, "res": res, "热门命中": fav,
                    "注数": num("Num1"), "单注": num("Money1"), "销量": num("TotalMoney")})
    return out


def main():
    D = load()
    print("=" * 80)
    print(f"  逆向策略最优强度测算 · 样本 {len(D)} 期")
    print("=" * 80)

    # 1) 奖金 ~ 热门命中数 回归（对数尺度）
    x = np.array([d["热门命中"] for d in D], float)
    y = np.log(np.array([d["单注"] for d in D], float))
    sl, ic, r, p, se = stats.linregress(x, y)
    print(f"\n  log(单注奖金) = {ic:.3f} {sl:+.3f} × 热门命中数     R²={r**2:.2f}  p={p:.1e}")
    print(f"  → 热门命中每少 1 场，单注奖金约 ×{np.exp(-sl):.2f}")
    print(f"  {'热门命中':>9}{'拟合单注奖金':>15}{'封顶后':>13}")
    for k in range(4, 13):
        v = np.exp(ic + sl * k)
        print(f"  {k:>9}{v:>15,.0f}{min(v, CAP):>13,.0f}")

    # 2) 逐期计算：反 k 场时的 P(全中) 与期望赔付
    print("\n" + "=" * 80)
    print("  反 k 场的代价与收益（每期取次选概率最高的 k 场做反向）")
    print("=" * 80)
    print(f"  {'反几场':>7}{'P(14全中)':>14}{'相对全押热门':>14}{'预期热门命中':>14}"
          f"{'拟合奖金':>13}{'期望赔付/注':>13}")
    print("  " + "-" * 76)
    base_ev = None
    for k in range(0, 6):
        Ps, favs = [], []
        for d in D:
            pr = [sorted(p.values(), reverse=True) for p in d["P"]]
            p1 = np.array([q[0] for q in pr]); p2 = np.array([q[1] for q in pr])
            ratio = p2 / p1
            idx = np.argsort(-ratio)[:k]          # 反向代价最小的 k 场
            pp = p1.copy()
            for i in idx: pp[i] = p2[i]
            Ps.append(float(np.prod(pp)))
            favs.append(14 - k)                    # 反向那 k 场按不中热门计
        P = float(np.mean(Ps)); f = float(np.mean(favs))
        prize = min(np.exp(ic + sl * f), CAP)
        ev = P * prize
        if k == 0: base_ev = ev
        print(f"  {k:>7}{P:>14.2e}{P/np.mean([np.prod([max(p.values()) for p in d['P']]) for d in D]):>14.2f}"
              f"{f:>14.1f}{prize:>13,.0f}{ev:>13,.2f}")
    print("  " + "-" * 76)
    print(f"  （每注成本 2 元；上表为一等奖期望赔付，未计二等奖）")

    # 3) 现实检验：热门命中数的实际分布
    print("\n" + "=" * 80)
    print("  现实检验：赛果本身有多冷门？")
    print("=" * 80)
    print(f"  热门命中数分布：均值 {x.mean():.1f} · 中位 {np.median(x):.0f} · 范围 {x.min():.0f}–{x.max():.0f}")
    print(f"  ≥10 场（很热）{(x>=10).mean():.0%} · 7–9 场 {((x>=7)&(x<10)).mean():.0%} · ≤6 场（很冷）{(x<=6).mean():.0%}")
    print(f"\n  ⚠ 关键：赛果冷不冷由比赛决定，不由我们决定。")
    print(f"     我们能选的只是「押多冷」，押冷了但赛果不冷 → 全盘皆输。")
    print(f"     上表 P(全中) 已含这一代价。")

    print("\n" + "=" * 80)
    print("  ⚠ 本测算为研究用途。足彩返奖率 65%，长期期望为负；")
    print("     且一等奖 500 万封顶会截断逆向策略的上行空间。不构成投注建议。")
    print("=" * 80)


if __name__ == "__main__":
    main()
