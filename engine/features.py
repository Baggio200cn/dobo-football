"""特征工程 · 严格防未来信息泄漏

铁律：每场比赛的特征，只能用「该场开赛前」已知的信息。
实现方式：按时间顺序遍历，先取当前状态算特征，再用本场结果更新状态。
（绝不使用 pandas 的整列 rolling —— 那样极易把未来信息带进来）

特征组：
  A 近期状态 form     —— 近5/10场胜率、场均进球/失球/射正
  B 交战记录 H2H      —— 近N次交手主队胜率、场均总进球
  C 裁判倾向 referee   —— 该裁判场均牌数、主队胜率偏向
  D 市场信号 market    —— 隐含概率、博彩公司分歧度、大小球盘口
  E 赛程密度 rest      —— 距上一场天数
  F 结构化基础概率      —— Elo / 泊松-DC 给出的先验（第一层模型）

用法：
  python features.py            # 构建并保存 features.parquet/csv
"""
import sys, math
from collections import defaultdict, deque
import numpy as np, pandas as pd
from config import BASE
from data import load_matches
from model import Poisson, elo_1x2, implied_1x2
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

FEAT_PATH = BASE / "features.csv"
K_ELO, HFA, BASE_ELO = 20, 60, 1500


def _safe(x, d=np.nan):
    try:
        v = float(x)
        return v if not math.isnan(v) else d
    except Exception:
        return d


def build(min_history=200):
    """构建特征表。min_history: 前 N 场只用于养状态，不产出样本。"""
    m = load_matches().reset_index(drop=True)

    # ---- 状态容器（全部只含"过去"）----
    elo = defaultdict(lambda: BASE_ELO)
    recent = defaultdict(lambda: deque(maxlen=10))   # 每队近10场记录
    h2h = defaultdict(lambda: deque(maxlen=6))       # 对阵组合近6次
    ref = defaultdict(lambda: {"n": 0, "cards": 0.0, "homewin": 0})
    lastdate = {}

    rows = []
    for i, x in enumerate(m.itertuples()):
        h, a = x.HomeTeam, x.AwayTeam
        eh, ea = elo[h], elo[a]

        # ===== A 近期状态（只看已入队的过去比赛）=====
        def form(team):
            r = recent[team]
            if not r: return dict(n=0, pts=np.nan, gf=np.nan, ga=np.nan, sot=np.nan)
            n = len(r)
            return dict(n=n,
                        pts=sum(z["pts"] for z in r) / n,
                        gf=sum(z["gf"] for z in r) / n,
                        ga=sum(z["ga"] for z in r) / n,
                        sot=sum(z["sot"] for z in r) / n)
        fh, fa = form(h), form(a)

        # ===== B 交战记录 =====
        pair = tuple(sorted([h, a]))
        hh = h2h[pair]
        if hh:
            h2h_n = len(hh)
            h2h_home_pts = np.mean([z["home_pts"] if z["home"] == h else (3 - z["home_pts"] if z["home_pts"] != 1 else 1) for z in hh])
            h2h_goals = np.mean([z["goals"] for z in hh])
        else:
            h2h_n, h2h_home_pts, h2h_goals = 0, np.nan, np.nan

        # ===== C 裁判倾向 =====
        rname = getattr(x, "Referee", None)
        rr = ref.get(rname) if rname and isinstance(rname, str) else None
        if rr and rr["n"] >= 5:
            ref_cards, ref_homewin = rr["cards"] / rr["n"], rr["homewin"] / rr["n"]
        else:
            ref_cards, ref_homewin = np.nan, np.nan

        # ===== D 市场信号（开盘）=====
        mk = implied_1x2(getattr(x, "B365H", None), getattr(x, "B365D", None), getattr(x, "B365A", None))
        mk_h, mk_d, mk_a = (mk["H"], mk["D"], mk["A"]) if mk else (np.nan,) * 3
        # 博彩公司分歧度：最高赔 vs 平均赔（分歧大 = 市场不确定）
        maxh, avgh = _safe(getattr(x, "MaxH", None)), _safe(getattr(x, "AvgH", None))
        odds_spread = (maxh - avgh) / avgh if (maxh and avgh and avgh > 0) else np.nan

        # ===== D2 赔率轨迹（开盘 → 收盘 = 资金流向）=====
        mkc = implied_1x2(getattr(x, "AvgCH", None), getattr(x, "AvgCD", None), getattr(x, "AvgCA", None))
        mko = implied_1x2(getattr(x, "AvgH", None), getattr(x, "AvgD", None), getattr(x, "AvgA", None))
        if mkc and mko:
            drift_h = mkc["H"] - mko["H"]      # >0 = 资金流向主队
            drift_d = mkc["D"] - mko["D"]
            drift_a = mkc["A"] - mko["A"]
            drift_abs = abs(drift_h) + abs(drift_d) + abs(drift_a)   # 总移动量
        else:
            drift_h = drift_d = drift_a = drift_abs = np.nan
        mkc_h, mkc_d, mkc_a = (mkc["H"], mkc["D"], mkc["A"]) if mkc else (np.nan,) * 3
        # 收盘分歧度
        maxch, avgch = _safe(getattr(x, "MaxCH", None)), _safe(getattr(x, "AvgCH", None))
        close_spread = (maxch - avgch) / avgch if (maxch and avgch and avgch > 0) else np.nan
        # 亚盘线移动（让球盘变化，强信号）
        ah_o, ah_c = _safe(getattr(x, "AHh", None)), _safe(getattr(x, "AHCh", None))
        ah_move = (ah_c - ah_o) if (ah_o == ah_o and ah_c == ah_c) else np.nan
        # 大小球盘移动
        ou_o, ou_c = _safe(getattr(x, "B365>2.5", None)), _safe(getattr(x, "B365C>2.5", None))
        ou_move = (1 / ou_c - 1 / ou_o) if (ou_o and ou_c and ou_o > 0 and ou_c > 0) else np.nan

        # ===== E 赛程密度 =====
        d = x.Date
        rest_h = (d - lastdate[h]).days if h in lastdate else np.nan
        rest_a = (d - lastdate[a]).days if a in lastdate else np.nan

        # ===== F 结构化先验（Elo）=====
        pe = elo_1x2(eh, ea, HFA)

        if i >= min_history:
            rows.append({
                "Date": d, "Div": x.Div, "HomeTeam": h, "AwayTeam": a, "FTR": x.FTR,
                "FTHG": x.FTHG, "FTAG": x.FTAG,
                # A
                "elo_h": eh, "elo_a": ea, "elo_diff": eh - ea + HFA,
                "form_pts_h": fh["pts"], "form_pts_a": fa["pts"], "form_pts_diff": fh["pts"] - fa["pts"] if fh["n"] and fa["n"] else np.nan,
                "form_gf_h": fh["gf"], "form_ga_h": fh["ga"],
                "form_gf_a": fa["gf"], "form_ga_a": fa["ga"],
                "form_sot_h": fh["sot"], "form_sot_a": fa["sot"],
                "sot_diff": fh["sot"] - fa["sot"] if fh["n"] and fa["n"] else np.nan,
                # B
                "h2h_n": h2h_n, "h2h_home_pts": h2h_home_pts, "h2h_goals": h2h_goals,
                # C
                "ref_cards": ref_cards, "ref_homewin": ref_homewin,
                # D 开盘市场
                "mkt_h": mk_h, "mkt_d": mk_d, "mkt_a": mk_a, "odds_spread": odds_spread,
                # D2 赔率轨迹
                "mkc_h": mkc_h, "mkc_d": mkc_d, "mkc_a": mkc_a,
                "drift_h": drift_h, "drift_d": drift_d, "drift_a": drift_a, "drift_abs": drift_abs,
                "close_spread": close_spread, "ah_move": ah_move, "ou_move": ou_move,
                "ah_close": ah_c,
                # E
                "rest_h": rest_h, "rest_a": rest_a,
                "rest_diff": (rest_h - rest_a) if (rest_h == rest_h and rest_a == rest_a) else np.nan,
                # F
                "elo_p_h": pe["H"], "elo_p_d": pe["D"], "elo_p_a": pe["A"],
            })

        # ========== 用本场结果更新状态（务必在产出特征之后）==========
        hg, ag = int(x.FTHG), int(x.FTAG)
        hp = 3 if hg > ag else (1 if hg == ag else 0)
        ap = 3 - hp if hp != 1 else 1
        hst = _safe(getattr(x, "HST", None), 0.0); ast = _safe(getattr(x, "AST", None), 0.0)
        recent[h].append({"pts": hp, "gf": hg, "ga": ag, "sot": hst})
        recent[a].append({"pts": ap, "gf": ag, "ga": hg, "sot": ast})
        h2h[pair].append({"home": h, "home_pts": hp, "goals": hg + ag})
        if rname and isinstance(rname, str):
            cards = _safe(getattr(x, "HY", None), 0) + _safe(getattr(x, "AY", None), 0) \
                    + 2 * (_safe(getattr(x, "HR", None), 0) + _safe(getattr(x, "AR", None), 0))
            ref[rname]["n"] += 1; ref[rname]["cards"] += cards
            ref[rname]["homewin"] += 1 if hg > ag else 0
        # Elo 更新
        exp_h = 1 / (1 + 10 ** ((ea - eh - HFA) / 400))
        s_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        elo[h] = eh + K_ELO * (s_h - exp_h)
        elo[a] = ea + K_ELO * ((1 - s_h) - (1 - exp_h))
        lastdate[h] = d; lastdate[a] = d

    df = pd.DataFrame(rows)
    df.to_csv(FEAT_PATH, index=False, encoding="utf-8-sig")
    return df


# 基础特征（不含赔率轨迹）——赛前早期即可获得
FEATURE_BASE = [
    "elo_diff", "elo_p_h", "elo_p_d", "elo_p_a",
    "form_pts_h", "form_pts_a", "form_pts_diff",
    "form_gf_h", "form_ga_h", "form_gf_a", "form_ga_a",
    "form_sot_h", "form_sot_a", "sot_diff",
    "h2h_n", "h2h_home_pts", "h2h_goals",
    "ref_cards", "ref_homewin",
    "mkt_h", "mkt_d", "mkt_a", "odds_spread",
    "rest_h", "rest_a", "rest_diff",
]

# 赔率轨迹特征 —— ⚠️ 只有临近开赛才完全可得（收盘赔率在开赛时才定）
FEATURE_ODDS_TRAJ = [
    "mkc_h", "mkc_d", "mkc_a",
    "drift_h", "drift_d", "drift_a", "drift_abs",
    "close_spread", "ah_move", "ou_move", "ah_close",
]

FEATURE_COLS = FEATURE_BASE + FEATURE_ODDS_TRAJ


if __name__ == "__main__":
    df = build()
    print(f"✅ 特征表：{len(df)} 场 × {len(FEATURE_COLS)} 特征")
    print(f"   时间跨度：{df.Date.min().date()} → {df.Date.max().date()}")
    print(f"   保存：{FEAT_PATH.name}\n")
    print("特征缺失率（高缺失需注意）：")
    miss = df[FEATURE_COLS].isna().mean().sort_values(ascending=False)
    for k, v in miss.items():
        flag = "⚠" if v > 0.2 else " "
        print(f"  {flag} {k:<16} {v:6.1%}")
