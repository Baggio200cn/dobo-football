"""模型层：两个基线
  1) Elo 评分（walk-forward，天然可在线更新）→ 球队实力排名 + 胜平负
  2) 泊松强度回归（statsmodels GLM）→ 比分分布 → 胜平负 / 大小球 / 最可能比分

诚实说明：泊松假设主客进球独立；Dixon-Coles 低比分相关性修正留作下一步改进。
"""
import numpy as np, pandas as pd
from scipy.stats import poisson
import statsmodels.formula.api as smf
import statsmodels.api as sm
from config import MAXG

# ---------- Elo ----------
def elo_run(matches, k=20, hfa=60, base=1500):
    """按时间顺序更新 Elo；返回 (最终评分 dict, 每场赛前评分列表用于回测)."""
    r = {}
    pre = []  # (idx, home_r, away_r)
    for idx, m in matches.iterrows():
        h, a = m.HomeTeam, m.AwayTeam
        rh, ra = r.get(h, base), r.get(a, base)
        pre.append((rh, ra))
        eh = 1 / (1 + 10 ** ((ra - rh - hfa) / 400))  # 主队期望
        sh = 1.0 if m.FTHG > m.FTAG else (0.5 if m.FTHG == m.FTAG else 0.0)
        r[h] = rh + k * (sh - eh)
        r[a] = ra + k * ((1 - sh) - (1 - eh))
    return r, pre


# 平局率随实力差变化的经验系数（10.6 万场二次拟合）
# 平局率 ≈ a·d² + b·d + c，d = |Elo差|（含主场修正）
DRAW_FIT = (-6.848e-07, -2.552e-04, 0.2896)


def draw_prob(elo_gap):
    """按实力差估平局率：势均力敌 ~29%，实力悬殊 ~8%。
    旧版固定 0.26 会系统性低估均势对决、高估悬殊对决的平局。"""
    d = abs(float(elo_gap))
    a, b, c = DRAW_FIT
    return float(np.clip(a * d * d + b * d + c, 0.05, 0.32))


def elo_1x2(rh, ra, hfa=60, draw=None):
    """Elo 差 → 胜平负。draw=None 时按实力差动态估平局率（推荐）。"""
    gap = rh + hfa - ra                      # 含主场优势的有效实力差
    if draw is None:
        draw = draw_prob(gap)
    eh = 1 / (1 + 10 ** (-gap / 400))        # 主队期望胜率（含平局）
    ph = eh * (1 - draw); pa = (1 - eh) * (1 - draw)
    return {"H": ph, "D": draw, "A": pa}


# ---------- 泊松强度 ----------
def sot_conversion(matches):
    """射正→进球的联赛整体转化率（xG 代理的核心系数）。"""
    d = matches.dropna(subset=["HST", "AST"])
    sot = (d["HST"].sum() + d["AST"].sum())
    goals = (d["FTHG"].sum() + d["FTAG"].sum())
    return float(goals / sot) if sot else 0.32


def dc_tau(i, j, lh, la, rho):
    """Dixon-Coles 低比分相关性修正因子（修正泊松独立假设）。"""
    if i == 0 and j == 0: return 1 - lh * la * rho
    if i == 0 and j == 1: return 1 + lh * rho
    if i == 1 and j == 0: return 1 + la * rho
    if i == 1 and j == 1: return 1 - rho
    return 1.0


class Poisson:
    """泊松强度回归。
    use_xg=True：用『射正×转化率』作 xG 代理目标（过程替代结果，降噪）。
    rho≠0：叠加 Dixon-Coles 低比分修正。"""
    def __init__(self, use_xg=False, rho=0.0):
        self.res = {}; self.teams = set(); self.mu = 1.35
        self.div_of = {}; self.mu_of = {}
        self.use_xg = use_xg; self.conv = 0.32; self.rho = rho

    def fit(self, matches):
        """按联赛分别拟合。

        本库无跨联赛比赛，故『整体拟合』与『分联赛拟合』在数学上等价，
        但整体拟合的设计矩阵是 2N × 2T（36 联赛下约 21 万 × 2600 = 4.4 GB），
        分联赛后降到 36 个 6000 × 80 的小矩阵。这是内存可行性的必要改动。
        """
        for div, sub in matches.groupby("Div", observed=True):
            self._fit_one(div, sub)
        return self

    def _fit_one(self, div, matches):
        rows = []
        if self.use_xg:
            self.conv = sot_conversion(matches)
            for m in matches.itertuples():
                if pd.isna(getattr(m, "HST", np.nan)) or pd.isna(getattr(m, "AST", np.nan)):
                    xh, xa = m.FTHG, m.FTAG          # 缺射正的老数据回退到真实进球
                else:
                    xh, xa = m.HST * self.conv, m.AST * self.conv
                rows.append((m.HomeTeam, m.AwayTeam, xh, 1))
                rows.append((m.AwayTeam, m.HomeTeam, xa, 0))
        else:
            for m in matches.itertuples():
                rows.append((m.HomeTeam, m.AwayTeam, m.FTHG, 1))
                rows.append((m.AwayTeam, m.HomeTeam, m.FTAG, 0))
        long = pd.DataFrame(rows, columns=["team", "opp", "goals", "home"])
        if long.team.nunique() < 3:                # 样本太少的联赛跳过
            return self
        self.teams |= set(long.team)
        for t in long.team.unique(): self.div_of[t] = div
        self.mu_of[div] = float(long.goals.mean())
        self.mu = float(np.mean(list(self.mu_of.values())))
        self.res[div] = smf.glm("goals ~ C(team) + C(opp) + home",
                                data=long, family=sm.families.Poisson()).fit()
        return self

    def lambdas(self, home, away):
        dh, da = self.div_of.get(home), self.div_of.get(away)
        if dh is None or dh != da or dh not in self.res:
            mu = self.mu_of.get(dh or da, self.mu)   # 未知队/跨联赛 → 联赛均值兜底
            return mu * 1.1, mu * 0.95
        res = self.res[dh]
        lh = res.predict(pd.DataFrame([{"team": home, "opp": away, "home": 1}]))[0]
        la = res.predict(pd.DataFrame([{"team": away, "opp": home, "home": 0}]))[0]
        return float(lh), float(la)

    def fit_rho(self, matches, grid=None):
        """网格搜索 Dixon-Coles 的 rho（最大化训练集比分对数似然）。"""
        grid = grid if grid is not None else np.linspace(-0.25, 0.02, 28)
        lam = [(self.lambdas(m.HomeTeam, m.AwayTeam), int(m.FTHG), int(m.FTAG))
               for m in matches.itertuples()]
        best_r, best_ll = 0.0, -1e18
        for r in grid:
            ll = 0.0
            for (lh, la), i, j in lam:
                p = poisson.pmf(i, lh) * poisson.pmf(j, la) * dc_tau(i, j, lh, la, r)
                ll += np.log(max(p, 1e-12))
            if ll > best_ll: best_ll, best_r = ll, r
        self.rho = float(best_r)
        return self

    def predict(self, home, away):
        lh, la = self.lambdas(home, away)
        gh = poisson.pmf(np.arange(MAXG + 1), lh)
        ga = poisson.pmf(np.arange(MAXG + 1), la)
        M = np.outer(gh, ga)                       # P(主 i : 客 j)
        if self.rho:                               # Dixon-Coles 修正（仅低比分格）
            for i in (0, 1):
                for j in (0, 1):
                    M[i, j] *= dc_tau(i, j, lh, la, self.rho)
            M /= M.sum()
        ph = np.tril(M, -1).sum()                  # 主胜 i>j
        pd_ = np.trace(M)                          # 平 i==j
        pa = np.triu(M, 1).sum()                   # 客胜 i<j
        s = ph + pd_ + pa
        i, j = np.unravel_index(M.argmax(), M.shape)
        over = sum(M[x, y] for x in range(MAXG + 1) for y in range(MAXG + 1) if x + y > 2)
        return {
            "H": ph / s, "D": pd_ / s, "A": pa / s,
            "lambda_home": lh, "lambda_away": la,
            "top_score": f"{i}-{j}", "over25": over, "under25": 1 - over,
        }


def implied_1x2(oh, od, oa):
    """Bet365 赔率 → 去水后的市场隐含概率（作为诚实基准）。"""
    if any(pd.isna(x) or not x for x in (oh, od, oa)): return None
    inv = np.array([1 / oh, 1 / od, 1 / oa]); inv /= inv.sum()
    return {"H": inv[0], "D": inv[1], "A": inv[2]}
