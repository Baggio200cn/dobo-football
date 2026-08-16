"""26106 期 A/B/C 三方案并行锁定

v14 实验发现：实盘用的 w=0.55（市场权重）在 36 联赛全域上劣于纯市场 0.0035 Brier。
但「改成纯市场」等于放弃模型。正确做法不是拍脑袋改，而是**同期并行下三套**，
让真实开奖来裁决。三套都锁定存档，复盘时逐一对账。

  A  w=0.55  现行方案（对照组）
  B  w=0.85  实验组：按回测曲线向市场靠拢
  C  w=1.00  纯市场基准（模型完全不发声）
"""
import sys, json, itertools, datetime as dt
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
from model import elo_1x2, implied_1x2
import sfc_lock as SL
from sfc26106 import FIX
import data as D
from model import elo_run

CODE = {"H": "3", "D": "1", "A": "0"}
ZH = {"H": "胜", "D": "平", "A": "负"}
ARMS = {"A": 0.55, "B": 0.85, "C": 1.00}


def find(r, name):
    if name in r: return name, True
    low = name.lower()
    for t in r:
        if t.lower() == low: return t, True
    c = [t for t in r if t.lower().startswith(low[:6])] or [t for t in r if low[:6] in t.lower()]
    return (max(c, key=lambda t: r[t]), True) if c else (None, False)


def probs(r, w):
    out = []
    for no, lg, hz, he, az, ae, ko, odds, neutral in FIX:
        th, _ = find(r, he); ta, _ = find(r, ae)
        pe = elo_1x2(r.get(th, 1500.0), r.get(ta, 1500.0), hfa=0 if neutral else 60)
        mk = implied_1x2(*odds) if odds else None
        if mk:
            p = {k: w*mk[k] + (1-w)*pe[k] for k in "HDA"}
            s = sum(p.values()); p = {k: v/s for k, v in p.items()}
        else:
            p = pe                      # 第 5 场无赔率，三套都只能用 Elo
        out.append({"no": no, "nm": f"{hz}vs{az}", "p": p})
    return out


def plan16(rows):
    """14 场 16 元复式（≤8 注）"""
    M = [{"pr": sorted([(x["p"]["H"], "H"), (x["p"]["D"], "D"), (x["p"]["A"], "A")],
                       reverse=True)} for x in rows]
    n = len(M)
    cover = lambda ks: float(np.prod([sum(M[i]["pr"][t][0] for t in range(ks[i])) for i in range(n)]))
    best = None
    for nd in range(0, 4):
        for c in itertools.combinations(range(n), nd):
            ks = [1]*n
            for i in c: ks[i] = 2
            pr = cover(ks)
            if best is None or pr > best[0]: best = (pr, 2**nd, list(ks))
    for i3 in range(n):
        for nd in (0, 1):
            for c in itertools.combinations([x for x in range(n) if x != i3], nd):
                ks = [1]*n; ks[i3] = 3
                for i in c: ks[i] = 2
                pr = cover(ks)
                if best is None or pr > best[0]: best = (pr, 3*(2**nd), list(ks))
    pr, bets, ks = best
    det = []
    for i in range(n):
        sel = "".join(sorted(CODE[M[i]["pr"][t][1]] for t in range(ks[i])))
        det.append({"场次": rows[i]["no"], "对阵": rows[i]["nm"], "选项": sel,
                    "中文": "+".join(ZH[M[i]["pr"][t][1]] for t in range(ks[i])),
                    "选项数": ks[i],
                    "覆盖概率": round(sum(M[i]["pr"][t][0] for t in range(ks[i])), 4)})
    return pr, bets, det


def main():
    m = D.load_matches()
    r, _ = elo_run(m)
    out = {}
    for arm, w in ARMS.items():
        rows = probs(r, w)
        conf = np.array([max(x["p"].values()) for x in rows])
        single = "".join(CODE[max(x["p"], key=x["p"].get)] for x in rows)
        pr, bets, det = plan16(rows)
        cov = np.array([d["覆盖概率"] for d in det])
        out[arm] = {"w市场": w, "单式": single,
                    "预测胜率": round(float(conf.mean()), 4),
                    "期望命中": round(float(conf.sum()), 2),
                    "全中概率": float(np.prod(conf)),
                    "复式": {"注数": bets, "金额": bets*2,
                            "加保场": [d["场次"] for d in det if d["选项数"] > 1],
                            "覆盖胜率": round(float(cov.mean()), 4),
                            "期望命中": round(float(cov.sum()), 2),
                            "全中概率": pr, "明细": det},
                    "逐场概率": [{"场次": x["no"], "对阵": x["nm"],
                              "胜": round(x["p"]["H"], 3), "平": round(x["p"]["D"], 3),
                              "负": round(x["p"]["A"], 3)} for x in rows]}

    print("=" * 76)
    print("  26106 期 A/B/C 三方案并行锁定")
    print("=" * 76)
    print(f"\n{'方案':<4}{'w市场':>7}{'单式串':>18}{'预测胜率':>10}{'期望命中':>10}"
          f"{'复式覆盖':>10}{'复式期望':>10}")
    print("-" * 76)
    for a in "ABC":
        o = out[a]
        print(f"{a:<4}{o['w市场']:>7.2f}{o['单式']:>18}{o['预测胜率']:>10.1%}"
              f"{o['期望命中']:>10.2f}{o['复式']['覆盖胜率']:>10.1%}{o['复式']['期望命中']:>10.2f}")
    print("-" * 76)

    # 差异定位
    print("\n【三套选号的分歧】")
    diff = 0
    for i in range(14):
        codes = {a: out[a]["单式"][i] for a in "ABC"}
        if len(set(codes.values())) > 1:
            diff += 1
            nm = out["A"]["逐场概率"][i]["对阵"]
            det = "  ".join(f"{a}={ZH[{'3':'H','1':'D','0':'A'}[codes[a]]]}" for a in "ABC")
            print(f"  第{i+1:>2}场 {nm:<24}{det}")
    if diff == 0:
        print("  三套单式选号完全相同 —— 权重只改变置信度，不改变判断。")
    print(f"\n【加保位置】")
    for a in "ABC":
        print(f"  {a}（w={ARMS[a]:.2f}）加保第 {out[a]['复式']['加保场']} 场")

    obj = {"期号": "26106", "截止": "2026-08-16 21:30",
           "锁定时间": str(dt.datetime.now())[:19],
           "实验": "v14 融合权重 A/B/C 并行检验",
           "说明": "同期并行三套方案，赛后用真实开奖裁决权重是否该改",
           "方案": out, "已复盘": False}
    fp = BASE / "batches" / "sfc_26106_AB.json"
    fp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ {fp.name}")


if __name__ == "__main__":
    main()
