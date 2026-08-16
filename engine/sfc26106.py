"""胜负彩第 26106 期预测（截止 2026-08-16 21:30）

本期特点：
  - 西甲 3 场 / 葡超 4 场 / 挪超·瑞典超 4 场 / 荷甲 1 场 / 英冠 1 场 / 社区盾 1 场
  - 第 2 场社区盾在温布利，中立场地 → 取消主场优势
  - 第 5 场（塞尔塔vs奥萨苏纳）500 站暂无平均赔率 → 仅用 Elo
  - 无跨联赛比赛，不需要 UEFA 校准
赔率取自 500.com「平均赔率」2026-08-16 抓取时点。
"""
import sys, json, datetime as dt
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
import data as D
from model import elo_run, elo_1x2, implied_1x2, goal_tendency

PERIOD, DEADLINE = "26106", "2026-08-16 21:30"
W_MKT = 0.55

# (场次, 联赛, 主中, 主英, 客中, 客英, 开赛, 平均赔率 或 None, 中立场)
FIX = [
 (1, "英冠",  "伯恩利","Burnley",     "西汉姆联","West Ham",    "08-16 23:00",(2.98,3.38,2.01),False),
 (2, "社区盾","阿森纳","Arsenal",     "曼彻斯特城","Man City",  "08-16 22:00",(2.47,3.12,2.48),True),
 (3, "西甲",  "桑坦德","Santander",   "比利亚雷亚尔","Villarreal","08-16 23:00",(3.25,3.48,1.87),False),
 (4, "西甲",  "西班牙人","Espanol",   "莱万特","Levante",       "08-17 01:00",(1.90,3.12,3.55),False),
 (5, "西甲",  "塞尔塔","Celta",       "奥萨苏纳","Osasuna",     "08-17 03:30",None,             False),
 (6, "荷甲",  "阿贾克斯","Ajax",      "海伦芬","Heerenveen",    "08-16 22:45",(1.32,4.78,6.20),False),
 (7, "葡超",  "葡国民","Nacional",    "埃斯托里尔","Estoril",   "08-16 22:30",(2.33,3.23,2.57),False),
 (8, "葡超",  "阿罗卡","Arouca",      "摩雷伦斯","Moreirense",  "08-17 01:00",(1.96,3.12,3.35),False),
 (9, "葡超",  "法马利康","Famalicao", "马里迪莫","Maritimo",    "08-17 03:30",(1.59,3.46,4.75),False),
 (10,"葡超",  "布拉加","Sp Braga",    "吉尔维森特","Gil Vicente","08-17 03:30",(1.44,3.80,5.90),False),
 (11,"瑞典超","盖斯","Gais",          "马尔默","Malmo",         "08-16 22:30",(2.04,3.45,2.86),False),
 (12,"瑞典超","卡尔马","Kalmar",      "哈马比","Hammarby",      "08-16 22:30",(4.00,3.76,1.63),False),
 (13,"挪超",  "布兰","Brann",         "汉姆卡姆","HamKam",      "08-16 23:00",(1.40,4.40,5.35),False),
 (14,"挪超",  "莫尔德","Molde",       "特罗姆瑟","Tromso",      "08-16 23:00",(1.93,3.60,3.00),False),
]


def find(r, name):
    if name in r: return name, True
    low = name.lower()
    for t in r:
        if t.lower() == low: return t, True
    cands = [t for t in r if t.lower().startswith(low[:6])] or \
            [t for t in r if low[:6] in t.lower()]
    if cands: return max(cands, key=lambda t: r[t]), True
    return None, False


def main():
    m = D.load_matches()
    r, _ = elo_run(m)
    GT = goal_tendency(m)          # v15：每队近 20 场场均总进球

    picks = []
    for no, lg, hz, he, az, ae, ko, odds, neutral in FIX:
        th, okh = find(r, he); ta, oka = find(r, ae)
        eh = r.get(th, 1500.0); ea = r.get(ta, 1500.0)
        gts = [GT[x] for x in (th, ta) if x in GT]
        gt = float(np.mean(gts)) if gts else None
        pe = elo_1x2(eh, ea, hfa=0 if neutral else 60, goal_tend=gt)
        mk = implied_1x2(*odds) if odds else None
        if mk:
            p = {k: W_MKT*mk[k] + (1-W_MKT)*pe[k] for k in "HDA"}
            s = sum(p.values()); p = {k: v/s for k, v in p.items()}
            src = "Elo+市场"
        else:
            p = pe; src = "仅Elo"
        code = max(p, key=p.get)
        picks.append({
            "场次":no,"赛事":lg,"主队":hz,"客队":az,"主队EN":th or he,"客队EN":ta or ae,
            "开赛":ko,"中立场":neutral,
            "进球倾向":round(gt,2) if gt else None,
            "Elo主":round(eh),"Elo客":round(ea),"Elo差":round(eh-ea),
            "胜":round(p["H"],3),"平":round(p["D"],3),"负":round(p["A"],3),
            "推荐码":{"H":"3","D":"1","A":"0"}[code],
            "推荐":{"H":"胜(3)","D":"平(1)","A":"负(0)"}[code],
            "置信":round(p[code],3),"来源":src,
            "市场胜":round(mk["H"],3) if mk else None,
            "偏差":round(p[code]-mk[code],3) if mk else None,
            "匹配":"实测" if (okh and oka) else ("单方" if (okh or oka) else "先验"),
        })

    conf = np.array([p["置信"] for p in picks])
    single = "".join(p["推荐码"] for p in picks)

    print("="*100)
    print(f"  胜负彩第 {PERIOD} 期预测   截止 {DEADLINE}")
    print("="*100)
    print(f"\n{'场':>3} {'联赛':<6}{'主队':<11}{'客队':<12}{'Elo差':>7}{'胜':>6}{'平':>6}{'负':>6}"
          f"{'荐':>4}{'置信':>7}  {'来源':<9}{'匹配'}")
    print("-"*100)
    for p in picks:
        flag = " ⚑中立" if p["中立场"] else ""
        print(f"{p['场次']:>3} {p['赛事']:<6}{p['主队']:<11}{p['客队']:<12}{p['Elo差']:>+7}"
              f"{p['胜']:>6.2f}{p['平']:>6.2f}{p['负']:>6.2f}{p['推荐码']:>4}{p['置信']:>7.1%}  "
              f"{p['来源']:<9}{p['匹配']}{flag}")
    print("-"*100)
    print(f"单式：{single}")
    print(f"预测胜率 {conf.mean():.1%} · 期望命中 {conf.sum():.2f}/14")
    print(f"推荐分布：胜 {single.count('3')} · 平 {single.count('1')} · 负 {single.count('0')}")
    q={}
    for p in picks: q[p["匹配"]]=q.get(p["匹配"],0)+1
    print(f"球队匹配：{q}")

    obj={"期号":PERIOD,"截止":DEADLINE,"生成时间":str(dt.datetime.now())[:19],
         "单式":single,"预测胜率":round(float(conf.mean()),4),
         "期望命中":round(float(conf.sum()),2),
         "训练库场次":int(len(m)),"训练库联赛":int(m.Div.nunique()),"picks":picks}
    fp=BASE/"batches"/f"sfc_{PERIOD}.json"
    fp.write_text(json.dumps(obj,ensure_ascii=False,indent=1),encoding="utf-8")
    print(f"\n✅ 已存档 {fp.name}")
    return obj


if __name__ == "__main__":
    main()
