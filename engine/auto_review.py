"""自动复盘：扫描所有未复盘的锁定期次，开奖后自动结算。

数据源：https://kaijiang.500.com/static/info/kaijiang/xml/sfc/{期号}.xml
        纯 XML，无 JS 渲染，GitHub Actions 里 requests 直接可读。
        关键字段：
          AllMatchFinished  1=全部踢完   0=未完
          Result            "0,3,0,..." 14 个赛果，逗号分隔
          MatchTeam@MatchScore / @Result / @HomeTeamName  逐场明细
          Num1/Money1 一等奖 · Num2/Money2 二等奖 · NineNum/NineMoney 任九

幂等：已复盘的跳过；未开奖的跳过并原样退出。可以随便多跑几次。

用法：
  python auto_review.py            # 扫描全部未复盘期次
  python auto_review.py 26106      # 只结算指定期
"""
import sys, json, re, glob
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import requests
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE

XML = "https://kaijiang.500.com/static/info/kaijiang/xml/sfc/{}.xml"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
ZH = {"3": "胜", "1": "平", "0": "负"}


# ---------------- 抓开奖 ----------------
def fetch(period):
    """返回 dict 或 None（未开奖）。"""
    try:
        r = requests.get(XML.format(period), headers=UA, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content.decode("utf-8", "replace"))
    except Exception as e:
        print(f"  ⚠ 抓取 {period} 失败：{e}")
        return None

    def g(t, d=""):
        e = root.find(t)
        return (e.text or d) if e is not None else d

    if g("AllMatchFinished") != "1" or not g("Result").strip():
        return None

    res = "".join(g("Result").split(","))
    if len(res) != 14 or any(c not in "310" for c in res):
        print(f"  ⚠ {period} 赛果格式异常：{g('Result')!r}")
        return None

    games = []
    for mt in root.findall(".//MatchTeam"):
        a = mt.attrib
        games.append({"序": int(a.get("OrderNum", 0)),
                      "主": a.get("HomeTeamName", ""), "客": a.get("GuestTeamName", ""),
                      "比分": a.get("MatchScore", ""), "果": a.get("Result", ""),
                      "赛事": a.get("SimpleGBName", "")})
    games.sort(key=lambda x: x["序"])

    def num(t):
        return int(re.sub(r"[^\d]", "", g(t, "0")) or 0)

    return {"期号": period, "开奖": res, "开奖时间": g("ResultTime"),
            "逐场": games,
            "销量": num("TotalMoney"), "任九销量": num("NineTotalMoney"),
            "奖金": {"一等奖": {"注数": num("Num1"), "单注": num("Money1")},
                    "二等奖": {"注数": num("Num2"), "单注": num("Money2")},
                    "任九":   {"注数": num("NineNum"), "单注": num("NineMoney")}}}


# ---------------- 结算一注 ----------------
def settle(sel_list, res, idx):
    """sel_list: 每场允许的选项字符串（如 '03'）；idx: 对应场次号 1-based"""
    return sum(1 for k, i in zip(sel_list, idx) if res[i - 1] in k)


def review_one(fp, kj):
    L = json.loads(Path(fp).read_text(encoding="utf-8"))
    res = kj["开奖"]
    S = L["单式"]

    # 队名核对（防止场次错位）
    warn = []
    for g, k in zip(L["逐场"], kj["逐场"]):
        ours = g["对阵"].split("vs")
        if ours[0][:2] not in k["主"] and k["主"][:2] not in ours[0]:
            warn.append(f"第{g['场次']}场 我们「{g['对阵']}」 vs 官方「{k['主']}vs{k['客']}」")

    rows, s_hit = [], 0
    for i, g in enumerate(L["逐场"]):
        act = res[i]
        ok = (g["单选"] == act)
        s_hit += ok
        g["实际"] = act; g["单式命中"] = bool(ok)
        rows.append({"场次": g["场次"], "对阵": g["对阵"], "预测": ZH[g["单选"]],
                     "置信": g["置信"], "实际": ZH[act], "命中": bool(ok),
                     "比分": kj["逐场"][i]["比分"]})

    R = {"单式命中": s_hit, "单式胜率": round(s_hit / 14, 4),
         "期望14": S["期望命中"], "偏差14": round(s_hit - S["期望命中"], 2),
         "实际平局数": res.count("1"), "逐场": rows,
         "奖金": kj["奖金"], "销量": kj["销量"], "任九销量": kj["任九销量"]}

    # 14 场复式（若有）
    F = L.get("复式16元")
    if F and F.get("明细"):
        idx = [d["场次"] for d in F["明细"]]
        sels = [d["选项"] for d in F["明细"]]
        fh = settle(sels, res, idx)
        R["复式命中"] = fh
        R["期望复式"] = F["期望命中"]
        R["偏差复式"] = round(fh - F["期望命中"], 2)
        R["复式加保收益"] = fh - s_hit
        R["复式中奖"] = bool(fh == 14)

    # 任选九（两种结构：主推 or 对照）
    N = L.get("任选九") or L.get("任选九对照")
    if N:
        n_idx = N["场次"]
        n_hit = sum(1 for i in n_idx if res[i - 1] == L["逐场"][i - 1]["单选"])
        R["任九命中"] = n_hit
        R["期望9"] = N["期望命中"]
        R["偏差9"] = round(n_hit - N["期望命中"], 2)
        R["任九场次"] = n_idx
        NF = N.get("复式")
        if NF and NF.get("明细"):
            idx = [d["场次"] for d in NF["明细"]]
            sels = [d["选项"] for d in NF["明细"]]
            nfh = settle(sels, res, idx)
            R["任九复式命中"] = nfh
            R["期望任九复式"] = NF["期望命中"]
            R["任九复式中奖"] = bool(nfh == 9)
            R["任九加保收益"] = nfh - n_hit
        R["任九中奖"] = bool(n_hit == 9)

    if warn:
        R["队名核对警告"] = warn

    L["实际开奖"] = res
    L["开奖时间"] = kj["开奖时间"]
    L["复盘"] = R
    L["已复盘"] = True
    Path(fp).write_text(json.dumps(L, ensure_ascii=False, indent=1), encoding="utf-8")
    return L, R, warn


def show(L, R, warn):
    p = L["期号"]
    print("=" * 78)
    print(f"  第 {p} 期复盘 · 锁定 {L['锁定时间']}（截止 {L['截止']}）")
    print("=" * 78)
    if warn:
        print("  ⚠ 队名核对警告：")
        for w in warn: print("    " + w)
    print(f"\n  开奖  {L['实际开奖']}")
    print(f"  预测  {L['单式']['串']}\n")
    print(f"  {'场':>3} {'对阵':<24}{'预测':>5}{'置信':>7}{'实际':>5}{'比分':>8}{'命中':>5}")
    print("  " + "-" * 60)
    for r in R["逐场"]:
        mark = "  ← 高置信翻车" if (not r["命中"] and r["置信"] >= 0.55) else ""
        print(f"  {r['场次']:>3} {r['对阵']:<24}{r['预测']:>5}{r['置信']:>7.0%}"
              f"{r['实际']:>5}{r['比分']:>8}{'✓' if r['命中'] else '✗':>5}{mark}")
    print("  " + "-" * 60)
    print(f"\n  【核心结果】")
    print(f"    单式 14 场   {R['单式命中']}/14 = {R['单式胜率']:.1%}   "
          f"期望 {R['期望14']}   偏差 {R['偏差14']:+.2f}")
    if "复式命中" in R:
        print(f"    16 元复式    {R['复式命中']}/14   期望 {R['期望复式']}   "
              f"偏差 {R['偏差复式']:+.2f}   加保救回 {R['复式加保收益']} 场")
    if "任九命中" in R:
        print(f"    任选九单式   {R['任九命中']}/9    期望 {R['期望9']}   偏差 {R['偏差9']:+.2f}"
              f"   {'🎉 中奖' if R['任九中奖'] else ''}")
    if "任九复式命中" in R:
        print(f"    任九复式     {R['任九复式命中']}/9    期望 {R['期望任九复式']}"
              f"   {'🎉 中奖' if R['任九复式中奖'] else ''}")
    print(f"\n  【平局】实际开出 {R['实际平局数']} 场（{R['实际平局数']/14:.0%}）")
    dm = [r for r in R["逐场"] if not r["命中"] and r["实际"] == "平"]
    if dm:
        print(f"    被平局打掉 {len(dm)} 场：" + "、".join(
            f"第{r['场次']}场({r['置信']:.0%})" for r in dm))
    a = R["奖金"]
    print(f"\n  【奖金】销量 {R['销量']:,} 元 · 任九销量 {R['任九销量']:,} 元")
    for k in ("一等奖", "二等奖", "任九"):
        print(f"    {k:<5} {a[k]['注数']:>7,} 注 × {a[k]['单注']:>9,} 元")


def ledger():
    lp = BASE / "batches" / "SFC_LEDGER.json"
    led = []
    for f in sorted(glob.glob(str(BASE / "batches" / "sfc_*_LOCK.json"))):
        L = json.loads(Path(f).read_text(encoding="utf-8"))
        if not L.get("已复盘"): continue
        R = L["复盘"]
        # 兼容旧版 sfc_review.py 写的字段名（期望单式/偏差单式，且无 实际平局数）
        exp = R.get("期望14", R.get("期望单式", L["单式"]["期望命中"]))
        hit = R["单式命中"]
        dev = R.get("偏差14", R.get("偏差单式", round(hit - exp, 2)))
        nd = R.get("实际平局数")
        if nd is None:
            nd = L.get("实际开奖", "").count("1") or None
        e = {"期号": L["期号"], "单式命中": hit,
             "预测胜率": L["单式"]["预测胜率"],
             "实际胜率": R.get("单式胜率", round(hit / 14, 4)),
             "期望": exp, "偏差": dev, "实际平局数": nd}
        for k in ("复式命中", "任九命中", "任九复式命中"):
            if k in R: e[k] = R[k]
        led.append(e)
    led.sort(key=lambda x: x["期号"])
    lp.write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding="utf-8")
    if led:
        h = sum(x["单式命中"] for x in led); n = len(led) * 14
        d = sum(x["实际平局数"] or 0 for x in led)
        print("\n" + "=" * 78)
        print(f"  累积台账：{len(led)} 期 · {n} 场 · 命中 {h} = {h/n:.1%} · 平局 {d} 场 = {d/n:.1%}")
        print("=" * 78)
        print(f"  {'期号':<8}{'预测':>8}{'实际':>8}{'偏差':>8}{'平局':>6}")
        for x in led:
            nd = x["实际平局数"]
            print(f"  {x['期号']:<8}{x['预测胜率']:>8.1%}{x['实际胜率']:>8.1%}"
                  f"{x['偏差']:>+8.2f}{(str(nd) if nd is not None else '—'):>6}")
    return led


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    files = sorted(glob.glob(str(BASE / "batches" / "sfc_*_LOCK.json")))
    pend = []
    for f in files:
        L = json.loads(Path(f).read_text(encoding="utf-8"))
        if L.get("已复盘"): continue
        if only and L["期号"] != only: continue
        pend.append((f, L["期号"]))

    if not pend:
        print("没有待复盘的期次。"); ledger(); return 0

    print(f"待复盘 {len(pend)} 期：{', '.join(p for _, p in pend)}\n")
    done = 0
    for f, p in pend:
        print(f"→ 查询 {p} …")
        kj = fetch(p)
        if kj is None:
            print(f"  ⏳ {p} 尚未开奖（或全部比赛未结束），跳过\n")
            continue
        L, R, warn = review_one(f, kj)
        show(L, R, warn)
        print()
        done += 1

    ledger()
    print(f"\n{'✅ 本次结算 %d 期' % done if done else 'ℹ 本次无新结算'}")
    return done


if __name__ == "__main__":
    main()
