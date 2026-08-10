"""导出网站所需数据：Elo 实力榜（含中文队名）+ 回测指标 → web_data.json
用法：python export_web.py
"""
import sys, json
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import data
from model import elo_run
from config import LEAGUES, BASE

ZH = {
 # 英超
 "Arsenal":"阿森纳","Man City":"曼城","Man United":"曼联","Liverpool":"利物浦","Aston Villa":"阿斯顿维拉",
 "Bournemouth":"伯恩茅斯","Brighton":"布莱顿","Chelsea":"切尔西","Brentford":"布伦特福德","Sunderland":"桑德兰",
 "Newcastle":"纽卡斯尔","Leeds":"利兹联","Fulham":"富勒姆","Nott'm Forest":"诺丁汉森林","Everton":"埃弗顿",
 "Crystal Palace":"水晶宫","West Ham":"西汉姆联","Tottenham":"热刺","Leicester":"莱斯特城","Wolves":"狼队",
 "Burnley":"伯恩利","Ipswich":"伊普斯维奇","Southampton":"南安普顿",
 # 西甲
 "Barcelona":"巴萨","Real Madrid":"皇马","Villarreal":"比利亚雷亚尔","Ath Madrid":"马竞","Betis":"贝蒂斯",
 "Celta":"塞尔塔","Vallecano":"巴列卡诺","Valencia":"瓦伦西亚","Sociedad":"皇家社会","Ath Bilbao":"毕尔巴鄂",
 "Sevilla":"塞维利亚","Girona":"赫罗纳","Osasuna":"奥萨苏纳","Mallorca":"马洛卡","Getafe":"赫塔菲",
 "Espanol":"西班牙人","Alaves":"阿拉维斯","Levante":"莱万特","Elche":"埃尔切","Oviedo":"奥维耶多",
 "Las Palmas":"拉斯帕尔马斯","Valladolid":"巴拉多利德","Leganes":"莱加内斯",
 # 意甲
 "Inter":"国米","Napoli":"那不勒斯","Atalanta":"亚特兰大","Juventus":"尤文","Milan":"AC米兰","Roma":"罗马",
 "Lazio":"拉齐奥","Fiorentina":"佛罗伦萨","Bologna":"博洛尼亚","Como":"科莫","Torino":"都灵","Udinese":"乌迪内斯",
 "Genoa":"热那亚","Cagliari":"卡利亚里","Parma":"帕尔马","Lecce":"莱切","Verona":"维罗纳","Sassuolo":"萨索洛",
 "Pisa":"比萨","Cremonese":"克雷莫纳","Empoli":"恩波利","Monza":"蒙扎","Venezia":"威尼斯",
 # 德甲
 "Bayern Munich":"拜仁","Leverkusen":"勒沃库森","Dortmund":"多特蒙德","Ein Frankfurt":"法兰克福","RB Leipzig":"莱比锡",
 "Stuttgart":"斯图加特","Freiburg":"弗赖堡","Werder Bremen":"云达不莱梅","Wolfsburg":"沃尔夫斯堡","M'gladbach":"门兴",
 "Mainz":"美因茨","Augsburg":"奥格斯堡","Hoffenheim":"霍芬海姆","Union Berlin":"柏林联合","St Pauli":"圣保利",
 "Heidenheim":"海登海姆","Hamburg":"汉堡","FC Koln":"科隆","Holstein Kiel":"基尔","Bochum":"波鸿",
 # 法甲
 "Paris SG":"巴黎圣日耳曼","Marseille":"马赛","Monaco":"摩纳哥","Lille":"里尔","Nice":"尼斯","Lyon":"里昂",
 "Lens":"朗斯","Strasbourg":"斯特拉斯堡","Brest":"布雷斯特","Rennes":"雷恩","Toulouse":"图卢兹","Auxerre":"欧塞尔",
 "Nantes":"南特","Angers":"昂热","Le Havre":"勒阿弗尔","Metz":"梅斯","Lorient":"洛里昂","Paris FC":"巴黎FC",
 "Reims":"兰斯","Montpellier":"蒙彼利埃","St Etienne":"圣埃蒂安",
}

def main():
    m = data.load_matches()
    r, _ = elo_run(m)
    last = {}
    for x in m.itertuples():
        last[x.HomeTeam] = x.Div; last[x.AwayTeam] = x.Div

    tables = {}
    for div, zh in LEAGUES.items():
        teams = [{"name": ZH.get(t, t), "en": t, "elo": round(v)}
                 for t, v in r.items() if last.get(t) == div]
        teams.sort(key=lambda x: -x["elo"])
        tables[zh] = teams[:20]

    out = {
        "updated": str(m["Date"].max().date()),
        "matches": int(len(m)),
        "span": f'{m["Date"].min().date()} → {m["Date"].max().date()}',
        "elo": tables,
        "backtest": [
            {"name": "市场(博彩基准)", "brier": 0.5827, "ll": 0.9790, "kind": "market"},
            {"name": "Elo 评分", "brier": 0.6015, "ll": 1.0070, "kind": "ours"},
            {"name": "泊松·Dixon-Coles", "brier": 0.6099, "ll": 1.0201, "kind": "ours"},
            {"name": "泊松·进球", "brier": 0.6105, "ll": 1.0208, "kind": "ours"},
            {"name": "泊松·xG 代理", "brier": 0.6106, "ll": 1.0205, "kind": "ours"},
        ],
        "test_n": 1752,
        # ===== 实验室档案（模型能力板块）=====
        "lab": {
            "dataset": {
                "n": int(len(m)),
                "span": f'{m["Date"].min().date()} → {m["Date"].max().date()}',
                "leagues": "英超 / 西甲 / 意甲 / 德甲 / 法甲",
                "fields": "比分 · 射门 · 射正 · 角球 · 红黄牌 · Bet365 赔率",
                "split": "训练 3,504 场 / 测试 1,752 场（按时间切分，防未来信息泄漏）",
                "source": "football-data.co.uk（公开免费）",
                "where": "本机 engine/data/raw/ · 网站仅展示结果快照",
            },
            "models": [
                {"name": "Elo 评分", "algo": "滚动评分更新", "feat": "历史胜负 + 主场优势"},
                {"name": "泊松·进球", "algo": "GLM 泊松回归", "feat": "球队攻防强度 + 主场"},
                {"name": "泊松·xG代理", "algo": "GLM 泊松回归", "feat": "射正 × 转化率"},
                {"name": "泊松·DC", "algo": "泊松 + Dixon-Coles", "feat": "攻防 + 低比分相关性"},
                {"name": "市场基准", "algo": "赔率去水", "feat": "Bet365 赔率"},
            ],
            "iterations": [
                {"v": "v1", "change": "Elo 基线", "brier": 0.6015, "delta": None, "verdict": "起点"},
                {"v": "v2", "change": "加泊松进球模型", "brier": 0.6105, "delta": 0.0090, "verdict": "单独用不如 Elo"},
                {"v": "v3", "change": "泊松改用 xG 代理（射正）", "brier": 0.6106, "delta": 0.0001, "verdict": "无效 · 粗糙代理丢信息"},
                {"v": "v4", "change": "泊松加 Dixon-Coles 修正", "brier": 0.6099, "delta": -0.0006, "verdict": "小幅有效"},
                {"v": "v5", "change": "26 特征 + LightGBM（原版）", "brier": 0.6136, "delta": 0.0037, "verdict": "过拟合 · 比 Elo 还差"},
                {"v": "v6", "change": "LightGBM 强正则简化（树400→120）", "brier": 0.5839, "delta": -0.0297, "verdict": "★ 大幅有效 · 首超 Elo"},
                {"v": "v7", "change": "0.6×市场 + 0.4×简化模型", "brier": 0.5801, "delta": -0.0038, "verdict": "达到市场水平"},
                {"v": "v8", "change": "温度缩放修校准", "brier": 0.5808, "delta": 0.0003, "verdict": "无效 · v6 已自愈过度自信"},
                {"v": "v9", "change": "加联赛特征（Div）", "brier": 0.5808, "delta": 0.0003, "verdict": "无效 · 模型已间接学到"},
                {"v": "v10", "change": "真 xG（3 个数据源）", "brier": None, "delta": None, "verdict": "数据不可用 · 覆盖有系统性偏差"},
                {"v": "v11", "change": "赔率轨迹（开盘→收盘 11 个特征）", "brier": 0.5791, "delta": -0.0010, "verdict": "✅ 有效 · 但价值来自收盘线而非漂移量"},
                {"v": "v12", "change": "主力缺阵指数（FPL 出场数据反推）", "brier": 0.5792, "delta": 0.0001, "verdict": "无效 · 市场已把伤停定价"},
            ],
            "queue": [
                "多家博彩分歧结构（Pinnacle 锐庄 vs 软庄的系统性差异）",
                "分角色建模（大小球 / 让球独立建模，可能比胜平负更有空间）",
                "时间加权（近期比赛权重更高）",
                "小联赛 / 低关注度赛事（市场定价可能更松）",
                "付费数据源评估（Opta / StatsBomb）",
            ],
            "failed": [
                {"item": "真 xG · Understat 直连", "why": "页面 JS 渲染，HTTP 只返回空壳"},
                {"item": "真 xG · FBref (soccerdata)", "why": "需浏览器驱动；直连 403"},
                {"item": "真 xG · GitHub FPL 镜像", "why": "仅英超且球员覆盖 4→19 人逐年漂移，缺失非随机"},
                {"item": "温度缩放", "why": "混合模型本就不过度自信（T=0.90）"},
                {"item": "联赛特征", "why": "增噪声，Brier 反升"},
                {"item": "亚盘线移动 ah_move / 大小球线移动 ou_move", "why": "特征重要性排 #35/#37，ou_move 为 0，无预测力"},
                {"item": "「漂移量=资金流向」假设", "why": "drift 仅排 #11-15；真正有用的是收盘线本身更锐利"},
                {"item": "主力缺阵指数（伤停代理）", "why": "关联成功率 94%，但增益 −0.0001，重要性排 #32-41/43 —— 市场已把伤停定价"},
            ],
            "insight": {
                "title": "市场效率假说：连伤停数据都加不出增量",
                "body": "我们用 FPL 出场记录反推出「主力缺阵指数」，94% 成功关联到英超比赛，数据质量没问题（场均 1.16 人缺阵）。"
                        "但加进模型后增益为 0，特征重要性垫底。原因：<b>庄家早已把伤停信息定价进赔率</b>，"
                        "而我们的模型本就在用市场概率做特征 —— 等于同一份信息喂了两遍。"
                        "这也解释了为什么公开数据很难战胜市场：<b>公开的信息，市场都已经知道了</b>。",
            },
            "insight2": {
                "title": "收盘线比开盘线准 0.0022 Brier",
                "body": "金融学的「收盘价更有效」在足球赔率上被验证。预测时点越晚、信息越充分。"
                        "但代价是：临场才能拿到收盘赔率，而那时模型基本等于复述市场共识（0.5780 vs 市场 0.5779）。",
            },
            "cv": {
                "method": "时间序列滚动 CV（4 折 · 严禁随机切分）",
                "note": "随机 CV 会把未来比赛放进训练集 → 回测虚高、实战崩盘",
                "top_features": ["odds_spread 赔率分歧", "sot_diff 射正差", "mkt_d 市场平局概率",
                                 "form_sot 近期射正", "h2h_goals 交战总进球"],
                "calib_warn": "78-90% 置信档：模型说 80.8%，实际 66.7% → 系统性过度自信，已列入队列",
            },
        },
        "season": [
            {"lg": "西甲", "d": "8月16日", "first": True},
            {"lg": "英超", "d": "8月22日"},
            {"lg": "意甲", "d": "8月23日"},
            {"lg": "法甲", "d": "8月23日"},
            {"lg": "德甲", "d": "8月28日"},
        ],
    }
    fp = BASE / "web_data.json"
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✅ 导出 {fp.name} · {out['matches']} 场 · 更新至 {out['updated']}")
    for k, v in tables.items():
        print(f"  {k}: " + " / ".join(f"{t['name']}{t['elo']}" for t in v[:3]))
    return out

if __name__ == "__main__":
    main()
