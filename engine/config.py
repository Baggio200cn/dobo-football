"""全局配置：五大联赛、赛季、路径。"""
from pathlib import Path

BASE = Path(__file__).parent.resolve()
DATA = BASE / "data"
RAW = DATA / "raw"
PRED = BASE / "predictions"
for d in (DATA, RAW, PRED):
    d.mkdir(parents=True, exist_ok=True)

# football-data.co.uk 联赛代码 → 中文名
LEAGUES = {
    # 五大联赛
    "E0": "英超", "SP1": "西甲", "I1": "意甲", "D1": "德甲", "F1": "法甲",
    # 胜负彩常用：荷葡比 + 英格兰低级别（英联杯球队来源）
    "N1": "荷甲", "P1": "葡超", "B1": "比甲", "T1": "土超", "G1": "希腊超",
    "E1": "英冠", "E2": "英甲", "E3": "英乙", "EC": "英足总",
    "SP2": "西乙", "I2": "意乙", "D2": "德乙", "F2": "法乙",
    "SC0": "苏super", "SC1": "苏冠",
}

# 「new」格式联赛（单文件含多赛季，字段名不同，仅收盘赔率）
NEW_LEAGUES = {
    "SWE": "瑞典超", "NOR": "挪超", "FIN": "芬超", "DNK": "丹麦超",
    "AUT": "奥地利", "SWZ": "瑞士超", "POL": "波兰", "IRL": "爱尔兰",
    "RUS": "俄超", "ROU": "罗马尼亚", "USA": "美职联", "BRA": "巴西",
    "ARG": "阿根廷", "JPN": "日职", "CHN": "中超", "MEX": "墨西哥",
}
ALL_LEAGUES = {**LEAGUES, **NEW_LEAGUES}

# 赛季代码（football-data 格式）：近三个赛季。off-season 时 2526 为最近完赛季。
SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526"]

FD_BASE = "https://www.football-data.co.uk"

# 用到的字段（结果 + 过程指标 + 裁判 + 多家赔率）
COLS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HST", "AST", "HS", "AS",              # 射门 / 射正
        "HC", "AC", "HY", "AY", "HR", "AR",    # 角球 / 红黄牌
        "Referee",                              # 裁判（100% 覆盖）
        # ---- 开盘赔率 ----
        "B365H", "B365D", "B365A",             # Bet365（市场基准）
        "MaxH", "MaxD", "MaxA",                # 最高赔（算分歧度）
        "AvgH", "AvgD", "AvgA",                # 平均赔
        "B365>2.5", "B365<2.5",                # 大小球盘口
        "AHh", "B365AHH", "B365AHA",           # 亚盘线 + 让球赔率
        # ---- 收盘赔率（开→收漂移 = 资金流向）----
        "B365CH", "B365CD", "B365CA",
        "MaxCH", "MaxCD", "MaxCA",
        "AvgCH", "AvgCD", "AvgCA",
        "B365C>2.5", "B365C<2.5",
        "AHCh", "B365CAHH", "B365CAHA"]

MAXG = 10  # 比分矩阵封顶进球数
