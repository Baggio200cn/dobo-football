"""胜负彩第 26104 期预测（截止 2026-08-14 22:00）

14 场全部落在我们数据库覆盖的联赛内（英冠/德乙/法乙/荷甲/葡超/瑞典超/挪超），
无需 UEFA 跨联赛校准 —— 比 26103 期的欧战条件好得多。

赔率取自 500.com「平均赔率」（截图时点），第 6、7 场暂无赔率 → 仅用 Elo。
"""
import sys, json, datetime as dt
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
import data as D
from model import elo_run, elo_1x2, implied_1x2

PERIOD, DEADLINE = "26104", "2026-08-14 22:00"
W_MKT = 0.55   # 市场权重（与前期一致）

# (场次, 联赛, 主队中文, 主队英文, 客队中文, 客队英文, 开赛, 平均赔率[胜,平,负] 或 None)
FIX = [
 (1,"英冠","狼队","Wolves","布莱克本","Blackburn","08-15 03:00",(1.53,4.07,5.69)),
 (2,"德乙","布伦瑞克","Braunschweig","波鸿","Bochum","08-15 00:30",(2.43,3.29,2.73)),
 (3,"德乙","荷尔斯泰因","Holstein Kiel","圣保利","St Pauli","08-15 00:30",(2.33,3.44,2.78)),
 (4,"法乙","昂纳西","Annecy","罗德兹","Rodez","08-15 02:45",(2.09,3.36,3.23)),
 (5,"法乙","南锡","Nancy","蒙彼利埃","Montpellier","08-15 02:45",(2.63,3.35,2.45)),
 (6,"法乙","第戎","Dijon","波城","Pau","08-15 02:45",None),
 (7,"法乙","格勒诺布尔","Grenoble","梅斯","Metz","08-15 02:45",None),
 (8,"法乙","甘冈","Guingamp","布洛涅","Boulogne","08-15 02:45",(1.72,3.57,4.50)),
 (9,"法乙","兰斯","Reims","敦刻尔克","Dunkerque","08-15 02:45",(1.70,3.69,4.46)),
 (10,"法乙","圣埃蒂安","St Etienne","克莱蒙","Clermont","08-15 02:45",(1.54,4.09,5.19)),
 (11,"荷甲","特尔斯达","Telstar","鹿特丹斯巴达","Sparta Rotterdam","08-15 02:00",(2.27,3.56,2.82)),
 (12,"葡超","葡萄牙体育","Sp Lisbon","吉马良斯","Guimaraes","08-15 03:15",(1.20,6.50,11.48)),
 (13,"瑞典超","埃尔夫斯堡","Elfsborg","瓦斯特拉斯","Vasteras","08-15 01:00",(1.81,3.69,3.86)),
 (14,"挪超","罗森博格","Rosenborg","维京","Viking","08-15 01:00",(2.55,3.80,2.33)),
]


def find(r, name):
    """在 Elo 表里找球队（精确 → 前缀 → 包含）"""
    if name in r: return name, True
    low = name.lower()
    for t in r:
        if t.lower() == low: return t, True
    cands = [t for t in r if t.lower().startswith(low[:6])] or \
            [t for t in r if low[:6] in t.lower()]
    if cands:
        return max(cands, key=lambda t: r[t]), True
    return None, False


def main():
    m = D.load_matches()
    r, _ = elo_run(m)
    last = {}
    for x in m.itertuples(): last[x.HomeTeam]=x.Div; last[x.AwayTeam]=x.Div

    picks = []
    for no, lg, hz, he, az, ae, ko, odds in FIX:
        th, okh = find(r, he); ta, oka = find(r, ae)
        eh = r.get(th, 1500.0); ea = r.get(ta, 1500.0)
        pe = elo_1x2(eh, ea)
        mk = implied_1x2(*odds) if odds else None
        if mk:
            p = {k: W_MKT*mk[k] + (1-W_MKT)*pe[k] for k in "HDA"}
            s = sum(p.values()); p = {k: v/s for k, v in p.items()}
            src = "Elo+市场"
        else:
            p = pe; src = "仅Elo"
        code = max(p, key=p.get)
        edge = (p[code] - mk[code]) if mk else None
        picks.append({
            "场次":no,"赛事":lg,"主队":hz,"客队":az,"主队EN":th or he,"客队EN":ta or ae,
            "开赛":ko,"Elo主":round(eh),"Elo客":round(ea),"Elo差":round(eh-ea),
            "胜":round(p["H"],3),"平":round(p["D"],3),"负":round(p["A"],3),
            "推荐码":{"H":"3","D":"1","A":"0"}[code],
            "推荐":{"H":"胜(3)","D":"平(1)","A":"负(0)"}[code],
            "置信":round(p[code],3),"来源":src,
            "市场胜":round(mk["H"],3) if mk else None,
            "偏差":round(edge,3) if edge is not None else None,
            "匹配":"实测" if (okh and oka) else ("单方" if (okh or oka) else "先验"),
        })

    conf = np.array([p["置信"] for p in picks])
    single = "".join(p["推荐码"] for p in picks)

    print("="*96)
    print(f"  胜负彩第 {PERIOD} 期预测   截止 {DEADLINE}")
    print("="*96)
    print(f"\n{'场':>3} {'联赛':<5}{'主队':<12}{'客队':<14}{'Elo差':>7}{'胜':>6}{'平':>6}{'负':>6}"
          f"{'荐':>4}{'置信':>7}  {'来源':<9}{'匹配'}")
    for p in picks:
        print(f"{p['场次']:>3} {p['赛事']:<5}{p['主队']:<12}{p['客队']:<14}{p['Elo差']:>+7}"
              f"{p['胜']:>6.2f}{p['平']:>6.2f}{p['负']:>6.2f}{p['推荐码']:>4}{p['置信']:>7.1%}  "
              f"{p['来源']:<9}{p['匹配']}")
    print("-"*96)
    print(f"单式：{single}")
    print(f"预测胜率 {conf.mean():.1%} · 期望命中 {conf.sum():.2f}/14")
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
