"""方法论第二条：任选九的逆向策略（实测版）

已证实的机制（61 期，p<0.0001）
    热门命中数 vs 一等奖注数   ρ = +0.78
    热门命中数 vs 单注奖金     ρ = -0.75
  → 人群跟随热门。赛果越冷，中奖人越少，单注奖金越高。

但「反着来」不能直接执行：赛果冷不冷由比赛决定（≤6 场热门的期次只占 21%），
押冷而赛果不冷则全盘皆输。正确表述是「在人群偏离真实概率最多处反向」。

⚠ 本脚本全部使用**实测频率**，不使用理论 P(全中)。
  上一版把理论概率与实测奖金混用，期望赔付被高估约 10 倍（1/4,795 vs 1/46,870）。
  硬约束：返奖率 65%，全体平均恒为 65%，任何算出长期数倍回报的模型必然有错。

策略族
    k = 0   全押热门（基线）
    k = 1..3  在入选的 9 场中，挑 p2/p1 比值最高的 k 场翻成次选
              （p2/p1 高 = 次选与首选差距小 = 反向代价最低）

评价（全部实测）
    · 平均命中 /9
    · P(9/9) 实测频率
    · 中奖期次的实际任九单注奖金
    · 期望赔付 = 实测 P(9/9) × 实际平均奖金
"""
import sys, re, json, itertools
import numpy as np
import xml.etree.ElementTree as ET
from scipy import stats
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
from model import implied_1x2

CODE = {"H": "3", "D": "1", "A": "0"}
KMAX = 3


def load():
    out = []
    for f in sorted((BASE / "cache_xml").glob("sfc_*.xml")):
        try: r = ET.fromstring(f.read_text(encoding="utf-8", errors="replace"))
        except Exception: continue
        g = lambda t: (r.findtext(t) or "").strip()
        res = "".join(g("Result").split(","))
        if len(res) != 14: continue
        num = lambda t: int(re.sub(r"[^\d]", "", g(t) or "0") or 0)
        nn, nm = num("NineNum"), num("NineMoney")
        if not nn: continue                      # 无任九奖金数据则跳过
        P, ok = [], True
        for m in sorted(r.findall(".//MatchTeam"), key=lambda x: int(x.attrib.get("OrderNum", 0))):
            od = (m.attrib.get("AverageOdds") or "").split()
            if len(od) != 3: ok = False; break
            P.append(implied_1x2(*[float(x) for x in od]))
        if not ok or len(P) != 14: continue
        out.append({"期": g("PeriodicalNO"), "P": P, "res": res,
                    "任九注数": nn, "任九单注": nm,
                    "任九销量": num("NineTotalMoney")})
    return out


def pick9(P):
    """任选九选场：取市场首选概率最高的 9 场（赛前可得）"""
    conf = [max(p.values()) for p in P]
    return sorted(range(14), key=lambda i: -conf[i])[:9]


def strategy(P, idx9, k):
    """在 9 场中翻转 p2/p1 最高的 k 场为次选，返回 {场序: 选码}"""
    pr = {i: sorted(P[i].items(), key=lambda x: -x[1]) for i in idx9}
    ratio = sorted(idx9, key=lambda i: -(pr[i][1][1] / pr[i][0][1]))
    flip = set(ratio[:k])
    return {i: CODE[(pr[i][1][0] if i in flip else pr[i][0][0])] for i in idx9}, flip


def main():
    D = load()
    print("=" * 84)
    print(f"  任选九逆向策略 · 实测回测 · 样本 {len(D)} 期")
    print("=" * 84)

    # 奖金 ~ 该期任九中奖注数：先看任九奖金与冷门度的关系
    fav_all, ninem, ninen = [], [], []
    for d in D:
        f = sum(1 for i, p in enumerate(d["P"]) if CODE[max(p, key=p.get)] == d["res"][i])
        fav_all.append(f); ninem.append(d["任九单注"]); ninen.append(d["任九注数"])
    fav_all = np.array(fav_all, float); ninem = np.array(ninem, float); ninen = np.array(ninen, float)
    r1, p1 = stats.spearmanr(fav_all, ninen)
    r2, p2 = stats.spearmanr(fav_all, ninem)
    print(f"\n  【任九也遵循同一机制】")
    print(f"    热门命中数 vs 任九中奖注数   ρ = {r1:+.2f}  p = {p1:.1e}")
    print(f"    热门命中数 vs 任九单注奖金   ρ = {r2:+.2f}  p = {p2:.1e}")
    hi = fav_all >= 9; lo = fav_all <= 6
    if hi.sum() and lo.sum():
        print(f"    热门 ≥9 场（{hi.sum()} 期）：任九中位 {np.median(ninen[hi]):>8,.0f} 注 · "
              f"单注 {np.median(ninem[hi]):>8,.0f} 元")
        print(f"    热门 ≤6 场（{lo.sum()} 期）：任九中位 {np.median(ninen[lo]):>8,.0f} 注 · "
              f"单注 {np.median(ninem[lo]):>8,.0f} 元")

    # 逐策略实测
    print("\n" + "=" * 84)
    print("  单式（1 注 2 元）：反 k 场的实测表现")
    print("=" * 84)
    print(f"  {'反几场':>7}{'平均命中/9':>12}{'9/9 期数':>10}{'实测P(9/9)':>13}"
          f"{'中奖时均奖金':>14}{'期望赔付/注':>13}{'相对基线':>10}")
    print("  " + "-" * 79)
    base = None
    detail = {}
    for k in range(0, KMAX + 1):
        hits, wins, prizes = [], 0, []
        for d in D:
            idx9 = pick9(d["P"])
            sel, flip = strategy(d["P"], idx9, k)
            h = sum(1 for i in idx9 if sel[i] == d["res"][i])
            hits.append(h)
            if h == 9:
                wins += 1; prizes.append(d["任九单注"])
        pw = wins / len(D)
        avg = float(np.mean(prizes)) if prizes else 0.0
        ev = pw * avg
        if k == 0: base = ev
        rel = (ev / base) if base else float("nan")
        detail[k] = {"平均命中": float(np.mean(hits)), "中奖期数": wins,
                     "P": pw, "均奖金": avg, "EV": ev}
        print(f"  {k:>7}{np.mean(hits):>12.2f}{wins:>10}{pw:>13.2%}"
              f"{avg:>14,.0f}{ev:>13.2f}{rel:>10.2f}×")
    print("  " + "-" * 79)
    print(f"  每注成本 2 元。期望赔付为实测频率 × 实际奖金，未做任何理论外推。")

    # 复式 16 元
    print("\n" + "=" * 84)
    print("  复式（16 元 8 注）：把预算用于「加保」还是「逆向」？")
    print("=" * 84)
    print(f"  {'方案':<26}{'9/9 期数':>10}{'实测P':>10}{'中奖时均奖金':>14}{'期望赔付/16元':>15}")
    print("  " + "-" * 75)
    plans = [("A 全热门 + 加保最接近 3 场", 0, 3),
             ("B 反 1 场 + 加保 2 场", 1, 2),
             ("C 反 2 场 + 加保 1 场", 2, 1),
             ("D 反 1 场 + 加保 3 场（12注超预算）", 1, 3)]
    for name, k, nh in plans:
        wins, prizes = 0, []
        for d in D:
            idx9 = pick9(d["P"])
            sel, flip = strategy(d["P"], idx9, k)
            pr = {i: sorted(d["P"][i].items(), key=lambda x: -x[1]) for i in idx9}
            cand = [i for i in idx9 if i not in flip]
            cand.sort(key=lambda i: -(pr[i][1][1] / pr[i][0][1]))
            hedged = set(cand[:nh])
            bets = 2 ** nh * (1 if k == 0 else 1)
            ok = True
            for i in idx9:
                allow = {sel[i]}
                if i in hedged: allow.add(CODE[pr[i][1][0]])
                if d["res"][i] not in allow: ok = False; break
            if ok:
                wins += 1; prizes.append(d["任九单注"])
        pw = wins / len(D); avg = float(np.mean(prizes)) if prizes else 0.0
        print(f"  {name:<26}{wins:>10}{pw:>10.2%}{avg:>14,.0f}{pw*avg*(2**nh):>15,.2f}")
    print("  " + "-" * 75)

    print("\n" + "=" * 84)
    print("  ⚠ 免责：返奖率 65%，长期期望为负。本测算为方法论研究，不构成投注建议。")
    print("=" * 84)
    return detail


if __name__ == "__main__":
    main()
