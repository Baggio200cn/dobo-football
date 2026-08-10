"""数据层：下载五大联赛历史 CSV + 未来赛程，合并成干净 DataFrame。"""
import sys, io
import numpy as np
import pandas as pd, requests
from config import LEAGUES, SEASONS, FD_BASE, RAW, DATA, COLS
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

UA = {"User-Agent": "Mozilla/5.0"}


def _download(url, dest, refresh=False):
    if dest.exists() and not refresh:
        return dest
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def update_new_format(refresh=False):
    """下载「new」格式联赛（北欧等，单文件含多赛季）。"""
    from config import NEW_LEAGUES
    files = []
    for lg in NEW_LEAGUES:
        dest = RAW / f"NEW_{lg}.csv"
        try:
            _download(f"{FD_BASE}/new/{lg}.csv", dest, refresh)
            files.append(dest)
        except Exception as e:
            print(f"  ⚠ 跳过 new/{lg}: {e}")
    return files


def _read_new(path):
    """把 new 格式规范化成主格式的列名。
    new: Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH...AvgCH...B365CH
    """
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    lg = path.stem.replace("NEW_", "")
    ren = {"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG", "Res": "FTR",
           # new 格式只有收盘赔率 → 同时充当开盘（缺开盘时的合理退化）
           "B365CH": "B365CH", "B365CD": "B365CD", "B365CA": "B365CA",
           "AvgCH": "AvgCH", "AvgCD": "AvgCD", "AvgCA": "AvgCA",
           "MaxCH": "MaxCH", "MaxCD": "MaxCD", "MaxCA": "MaxCA"}
    df = df.rename(columns=ren)
    if not {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(df.columns):
        return None
    df["Div"] = lg
    # 开盘赔率缺失 → 用收盘填充（这些联赛只提供收盘）
    for o, c in [("B365H", "B365CH"), ("B365D", "B365CD"), ("B365A", "B365CA"),
                 ("AvgH", "AvgCH"), ("AvgD", "AvgCD"), ("AvgA", "AvgCA"),
                 ("MaxH", "MaxCH"), ("MaxD", "MaxCD"), ("MaxA", "MaxCA")]:
        if c in df.columns:
            df[o] = df[c]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for c in ("FTHG", "FTAG", "B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA",
              "MaxH", "MaxD", "MaxA", "B365CH", "B365CD", "B365CA",
              "AvgCH", "AvgCD", "AvgCA", "MaxCH", "MaxCD", "MaxCA"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int); df["FTAG"] = df["FTAG"].astype(int)
    if "FTR" not in df.columns or df["FTR"].isna().all():
        df["FTR"] = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG == df.FTAG, "D", "A"))
    keep = [c for c in COLS if c in df.columns]
    return df[keep]


def update(refresh=False):
    """下载全部 联赛×赛季 CSV 到 data/raw，返回本地文件清单。"""
    files = []
    for season in SEASONS:
        for div in LEAGUES:
            url = f"{FD_BASE}/mmz4281/{season}/{div}.csv"
            dest = RAW / f"{season}_{div}.csv"
            try:
                _download(url, dest, refresh)
                files.append(dest)
            except Exception as e:
                print(f"  ⚠ 跳过 {season}/{div}: {e}")
    # new 格式联赛（北欧等）
    try:
        update_new_format(refresh)
    except Exception as e:
        print(f"  ⚠ new 格式下载异常: {e}")
    # 未来赛程
    try:
        _download(f"{FD_BASE}/fixtures.csv", DATA / "fixtures.csv", refresh=True)
    except Exception as e:
        print(f"  ⚠ fixtures.csv 下载失败: {e}")
    return files


def _read_one(path):
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    df = df.rename(columns={df.columns[0]: "Div"})  # 修 BOM 污染的首列名
    keep = [c for c in COLS if c in df.columns]
    df = df[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    numcols = ("FTHG", "FTAG", "HST", "AST", "HS", "AS", "HC", "AC",
               "HY", "AY", "HR", "AR",
               "B365H", "B365D", "B365A", "MaxH", "MaxD", "MaxA",
               "AvgH", "AvgD", "AvgA", "B365>2.5", "B365<2.5",
               "AHh", "B365AHH", "B365AHA",
               "B365CH", "B365CD", "B365CA", "MaxCH", "MaxCD", "MaxCA",
               "AvgCH", "AvgCD", "AvgCA", "B365C>2.5", "B365C<2.5",
               "AHCh", "B365CAHH", "B365CAHA")
    for c in numcols:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["FTHG"] = df["FTHG"].astype(int); df["FTAG"] = df["FTAG"].astype(int)
    return df


def load_matches():
    """合并所有已下载赛季的比赛（按日期排序）。"""
    frames = []
    for p in sorted(RAW.glob("*.csv")):
        try:
            d = _read_new(p) if p.name.startswith("NEW_") else _read_one(p)
            if d is not None and len(d): frames.append(d)
        except Exception as e: print(f"  ⚠ 读取失败 {p.name}: {e}")
    if not frames:
        raise SystemExit("没有数据。先运行 data.update()。")
    m = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return m


def load_fixtures():
    """未来赛程（仅五大联赛）。off-season 可能为空。"""
    fp = DATA / "fixtures.csv"
    if not fp.exists(): return pd.DataFrame()
    df = pd.read_csv(fp, encoding="latin-1", on_bad_lines="skip")
    df = df[df["Div"].isin(LEAGUES)].copy() if "Div" in df else pd.DataFrame()
    if len(df):
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        for c in ("B365H", "B365D", "B365A"):
            if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    print("下载五大联赛数据…")
    update(refresh=refresh)
    m = load_matches()
    print(f"✅ 比赛样本：{len(m)} 场 · {m['Date'].min().date()} → {m['Date'].max().date()}")
    print(m.groupby("Div").size().to_string())
    fx = load_fixtures()
    print(f"未来赛程：{len(fx)} 场" + ("（休赛期为空属正常）" if not len(fx) else ""))
