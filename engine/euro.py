"""欧战预测（跨联赛）· 用 UEFA 国家系数校准 Elo

核心问题：我们的 Elo 按联赛独立演化（数据里没有跨联赛比赛），
         「挪超 1772」与「比甲 1765」不可直接比较。
解决：用 UEFA 国家系数（5 年累计）给每个联赛一个强度偏移
     adj_elo = 联赛内 Elo + 170 × ln(该国系数)
     缺失球队 → 用该国联赛均值(1500) + 偏移 作为先验

⚠️ 局限：欧战有双回合、旅行、轮换等联赛没有的因素；本模型不建模这些。
"""
import sys, math, json, datetime as dt
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE
import data as D
from model import elo_run, elo_1x2, draw_prob

# UEFA 国家系数（2027 榜 · 2026-08-06 更新 · kassiesa.net）
UEFA = {
    "England":101.852,"Italy":87.660,"Spain":82.368,"Germany":80.116,"France":67.653,
    "Portugal":63.650,"Belgium":57.850,"Netherlands":51.562,"Turkey":47.575,"Czechia":44.025,
    "Poland":43.525,"Greece":41.312,"Denmark":36.181,"Norway":34.212,"Cyprus":33.193,
    "Switzerland":29.075,"Hungary":26.562,"Sweden":25.625,"Austria":25.250,"Scotland":25.050,
    "Croatia":24.531,"Romania":24.500,"Ukraine":24.087,"Israel":22.625,"Slovenia":22.468,
    "Azerbaijan":21.187,"Bulgaria":20.187,"Slovakia":20.125,"Serbia":17.500,"Russia":17.332,
    "Iceland":16.770,"Ireland":16.093,"Finland":12.875,"Kazakhstan":12.375,
}
# 我们的联赛代码 → 国家
DIV2CTY = {
    "E0":"England","E1":"England","E2":"England","E3":"England","EC":"England",
    "I1":"Italy","I2":"Italy","SP1":"Spain","SP2":"Spain","D1":"Germany","D2":"Germany",
    "F1":"France","F2":"France","P1":"Portugal","B1":"Belgium","N1":"Netherlands",
    "T1":"Turkey","G1":"Greece","SC0":"Scotland","SC1":"Scotland",
    "DNK":"Denmark","NOR":"Norway","SWZ":"Switzerland","SWE":"Sweden","AUT":"Austria",
    "POL":"Poland","ROU":"Romania","RUS":"Russia","IRL":"Ireland","FIN":"Finland",
}
K_LEAGUE = 170.0     # 系数→Elo 的换算尺度


def league_offset(country):
    c = UEFA.get(country)
    return K_LEAGUE * math.log(c) if c else K_LEAGUE * math.log(15.0)


# 26103 期 14 场：(场次, 赛事, 主队中文, 主队英文/None, 主队国, 客队中文, 客队英文/None, 客队国, 中立场?)
FIX = [
 (1,"欧冠","阿拉木图",None,"Kazakhstan","凯尔特人","Celtic","Scotland",False),
 (2,"欧冠","博德","Bodo/Glimt","Norway","圣吉尔联合","St. Gilloise","Belgium",False),
 (3,"欧冠","萨巴赫",None,"Azerbaijan","奥胡斯","Aarhus","Denmark",False),
 (4,"欧冠","奈梅亨","Nijmegen","Netherlands","奥林匹亚科斯","Olympiakos","Greece",False),
 (5,"欧冠","布拉迪斯拉发",None,"Slovakia","米亚尔比","Mjallby","Sweden",False),
 (6,"欧冠","格拉茨风暴","Sturm Graz","Austria","费内巴切","Fenerbahce","Turkey",False),
 (7,"欧冠","里昂","Lyon","France","布拉格斯巴达",None,"Czechia",False),
 (8,"欧罗巴","帕福斯",None,"Cyprus","萨尔茨堡","Salzburg","Austria",False),
 (9,"欧罗巴","克拉约瓦","Univ. Craiova","Romania","库奥皮奥","KuPS","Finland",False),
 (10,"欧罗巴","雷克雅未克维京人",None,"Iceland","图恩","Thun","Switzerland",False),
 (11,"欧罗巴","安德莱赫特","Anderlecht","Belgium","塞萨洛尼基","PAOK","Greece",False),
 (12,"欧罗巴","流浪者","Rangers","Scotland","比亚韦","Jagiellonia","Poland",False),
 (13,"欧罗巴","哈茨","Hearts","Scotland","本菲卡","Benfica","Portugal",False),
 (14,"欧超杯","巴黎圣日耳曼","Paris SG","France","阿斯顿维拉","Aston Villa","England",True),
]
PERIOD, DEADLINE = "26103", "2026-08-11 22:00"


def main():
    m = D.load_matches()
    r, _ = elo_run(m)
    last = {}
    for x in m.itertuples(): last[x.HomeTeam]=x.Div; last[x.AwayTeam]=x.Div

    def team_elo(en, cty):
        """返回 (联赛内Elo, 是否真实数据)"""
        if en and en in r: return r[en], True
        return 1500.0, False        # 缺失 → 该国联赛均值先验

    picks = []
    for no, comp, hz, he, hc, az, ae, ac, neutral in FIX:
        eh, ok_h = team_elo(he, hc)
        ea, ok_a = team_elo(ae, ac)
        adj_h = eh + league_offset(hc)
        adj_a = ea + league_offset(ac)
        hfa = 0 if neutral else 60          # 中立场无主场优势
        p = elo_1x2(adj_h, adj_a, hfa=hfa)
        code = max(p, key=p.get)
        conf = p[code]
        quality = "双方实测" if (ok_h and ok_a) else ("单方实测" if (ok_h or ok_a) else "双方先验")
        picks.append({
            "场次":no,"赛事":comp,"主队":hz,"客队":az,
            "主国":hc,"客国":ac,
            "主Elo":round(eh),"客Elo":round(ea),
            "主校准":round(adj_h),"客校准":round(adj_a),
            "校准差":round(adj_h+hfa-adj_a),
            "胜":round(p["H"],3),"平":round(p["D"],3),"负":round(p["A"],3),
            "推荐码":{"H":"3","D":"1","A":"0"}[code],
            "推荐":{"H":"胜(3)","D":"平(1)","A":"负(0)"}[code],
            "置信":round(conf,3),"数据质量":quality,"中立场":neutral,
        })

    conf = np.array([p["置信"] for p in picks])
    single = "".join(p["推荐码"] for p in picks)
    print("="*94)
    print(f"  胜负彩第 {PERIOD} 期 · 欧战专版预测（UEFA 系数校准）  截止 {DEADLINE}")
    print("="*94)
    print(f"\n{'场':>3} {'赛事':<5}{'主队':<14}{'客队':<14}{'主Elo':>6}{'客Elo':>6}{'校准差':>7}"
          f"{'胜':>6}{'平':>6}{'负':>6}{'荐':>4}{'置信':>7}  数据质量")
    for p in picks:
        print(f"{p['场次']:>3} {p['赛事']:<5}{p['主队']:<14}{p['客队']:<14}"
              f"{p['主Elo']:>6}{p['客Elo']:>6}{p['校准差']:>+7}"
              f"{p['胜']:>6.2f}{p['平']:>6.2f}{p['负']:>6.2f}{p['推荐码']:>4}{p['置信']:>7.1%}  {p['数据质量']}"
              + ("  ⚑中立场" if p["中立场"] else ""))
    print("\n" + "-"*94)
    print(f"单式：{single}")
    print(f"预测胜率 {conf.mean():.1%} · 期望命中 {conf.sum():.2f}/14")
    q = {}
    for p in picks: q[p["数据质量"]] = q.get(p["数据质量"],0)+1
    print(f"数据质量分布：{q}")

    obj = {"期号":PERIOD,"截止":DEADLINE,"生成时间":str(dt.datetime.now())[:19],
           "方法":"UEFA国家系数校准跨联赛Elo","单式":single,
           "预测胜率":round(float(conf.mean()),4),"期望命中":round(float(conf.sum()),2),
           "picks":picks}
    fp = BASE/"batches"/f"sfc_{PERIOD}.json"
    fp.write_text(json.dumps(obj,ensure_ascii=False,indent=1),encoding="utf-8")
    print(f"\n✅ 已存档 {fp.name}")
    return obj


if __name__ == "__main__":
    main()
