"""文章封面生成器 · DoBo 品牌风格

从 .md 文章的 frontmatter + 正文自动提取标题、副标、关键数据，
生成 1200×675（16:9，头条推荐尺寸）封面图。

用法：
  python make_cover.py                     # 为 articles/ 下所有 .md 生成封面
  python make_cover.py 02_第一次真实检验.md   # 单篇
"""
import sys, re, glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE

ART = BASE.parent / "articles"
W, H = 1200, 675
BG      = (13, 17, 23)        # #0d1117 与网站一致
CARD    = (22, 27, 36)
LINE    = (36, 44, 58)
GOLD    = (245, 196, 81)
GREEN   = (34, 197, 94)
RED     = (226, 59, 59)
TXT     = (233, 238, 245)
MUT     = (141, 153, 171)
MUT2    = (93, 104, 121)

# 跨平台字体：Windows 用微软雅黑/华文中宋，Linux(CI) 用 Noto CJK
FONT_DIRS = ["C:/Windows/Fonts/", "/usr/share/fonts/opentype/noto/",
             "/usr/share/fonts/truetype/noto/", "/usr/share/fonts/truetype/dejavu/"]
FALLBACK = {"msyhbd.ttc": ["NotoSansCJK-Bold.ttc", "NotoSansCJKsc-Bold.otf", "NotoSerifCJK-Bold.ttc"],
            "msyh.ttc":   ["NotoSansCJK-Regular.ttc", "NotoSansCJKsc-Regular.otf"],
            "STZHONGS.TTF":["NotoSerifCJK-Bold.ttc", "NotoSerifCJKsc-Bold.otf", "NotoSansCJK-Bold.ttc"]}

def font(name, size):
    for cand in [name] + FALLBACK.get(name, []):
        for d in FONT_DIRS:
            try: return ImageFont.truetype(d + cand, size)
            except Exception: continue
    return ImageFont.load_default()

FB   = lambda s: font("msyhbd.ttc", s)      # 粗
FR   = lambda s: font("msyh.ttc", s)        # 常规
FSER = lambda s: font("STZHONGS.TTF", s)    # 衬线（大标题）


def wrap(draw, text, fnt, maxw):
    lines, cur = [], ""
    for ch in text:
        t = cur + ch
        if draw.textlength(t, font=fnt) > maxw and cur:
            lines.append(cur); cur = ch
        else:
            cur = t
    if cur: lines.append(cur)
    return lines


def parse(md_path):
    s = Path(md_path).read_text(encoding="utf-8")
    meta = {}
    m = re.match(r"^---\n(.*?)\n---\n", s, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"')
        body = s[m.end():]
    else:
        body = s
    # 标题
    title = meta.get("title") or ""
    h1 = re.search(r"^#\s+(.+)$", body, re.M)
    if h1: title = h1.group(1).strip()
    # 主副标（用中文冒号或英文冒号切）
    main, sub = title, ""
    for sep in ("：", ": "):
        if sep in title:
            main, sub = title.split(sep, 1); break
    # 关键数据：优先用 frontmatter 的 cover_stats（格式 "标签=值 | 标签=值"）
    stats = []
    cs = meta.get("cover_stats", "")
    if cs:
        for item in cs.split("|"):
            if "=" in item:
                k, v = item.split("=", 1)
                stats.append((k.strip(), v.strip()))
    else:
        # 兜底：抓「实际中 N 场」「胜率 N%」这类明确表述，且去重
        seen = set()
        for pat, lab in [(r"实际中?\s*(\d+)\s*场", "实际命中"),
                         (r"命中率?\s*(\d{2}\.\d)%", "命中率"),
                         (r"(\d+)\s*轮迭代", "迭代轮次")]:
            for v in re.findall(pat, body):
                if (lab, v) not in seen:
                    seen.add((lab, v)); stats.append((lab, v)); break
    return {"title": main.strip(), "sub": sub.strip(),
            "date": meta.get("date", ""), "style": meta.get("style", ""),
            "period": meta.get("period", ""), "stats": stats[:3]}


def make(md_path, out=None):
    d = parse(md_path)
    img = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(img)

    # 背景网格
    for x in range(0, W, 40): dr.line([(x,0),(x,H)], fill=(18,23,31), width=1)
    for y in range(0, H, 40): dr.line([(0,y),(W,y)], fill=(18,23,31), width=1)
    # 右上光晕
    for i in range(120, 0, -3):
        a = int(18 * (i/120))
        dr.ellipse([W-160-i, -60-i, W-160+i, -60+i], fill=(BG[0]+a//3, BG[1]+a//3, BG[2]+a//8))
    # 顶部红条
    dr.rectangle([0, 0, W, 5], fill=RED)

    # 品牌
    try:
        logo = Image.open(BASE.parent.parent.parent / "10.projects/JOJO-Director/assets/JOJO-Director-logo.png")
    except Exception:
        logo = None
    x0 = 62
    dr.text((x0, 46), "DoBo", font=FB(34), fill=TXT)
    lw = dr.textlength("DoBo", font=FB(34))
    dr.text((x0+lw+10, 52), "足球数据", font=FB(26), fill=GOLD)
    dr.text((x0, 92), "FIVE  LEAGUES  DATA  ·  MODEL  REVIEW", font=FR(14), fill=MUT2)

    # 期号标签
    if d["period"]:
        tag = d["period"].replace("复盘", "复盘 ").replace("预测", " 预测")
        tw = dr.textlength(tag, font=FR(18))
        dr.rounded_rectangle([W-62-tw-26, 48, W-62, 84], 6, outline=GOLD, width=2)
        dr.text((W-62-tw-13, 55), tag, font=FR(18), fill=GOLD)

    # 主标题（衬线大字）
    y = 168
    fs = 62 if len(d["title"]) <= 16 else (52 if len(d["title"]) <= 24 else 44)
    for ln in wrap(dr, d["title"], FSER(fs), W-124)[:3]:
        dr.text((x0, y), ln, font=FSER(fs), fill=TXT); y += fs + 16

    # 副标
    if d["sub"]:
        y += 6
        for ln in wrap(dr, d["sub"], FR(26), W-124)[:2]:
            dr.text((x0, y), ln, font=FR(26), fill=MUT); y += 38

    # 关键数据条
    if d["stats"]:
        by = H - 172
        dr.line([(x0, by-24), (W-62, by-24)], fill=LINE, width=1)
        bx = x0
        for lab, v in d["stats"]:
            dr.text((bx, by), v, font=FB(46), fill=GOLD)
            vw = dr.textlength(v, font=FB(46))
            dr.text((bx, by+56), lab, font=FR(17), fill=MUT2)
            bx += max(vw, 60) + 72

    # 底部
    dr.line([(x0, H-72), (W-62, H-72)], fill=LINE, width=1)
    dr.text((x0, H-56), "@巴老师挨扯淡  ·  数据分析与模型复盘，不构成投注建议",
            font=FR(17), fill=MUT2)
    if d["date"]:
        dw = dr.textlength(d["date"], font=FR(17))
        dr.text((W-62-dw, H-56), d["date"], font=FR(17), fill=MUT2)

    out = out or Path(md_path).with_suffix(".cover.png")
    img.save(out, quality=95)
    return out, d


def main():
    ART.mkdir(exist_ok=True)
    targets = sys.argv[1:] or sorted(glob.glob(str(ART / "*.md")))
    targets = [ART / t if not str(t).startswith(str(ART)) else Path(t) for t in targets]
    for t in targets:
        if not Path(t).exists(): print(f"  ⚠ 跳过 {t}"); continue
        out, d = make(t)
        print(f"  ✅ {Path(out).name}")
        print(f"     标题：{d['title']}")
        if d["sub"]: print(f"     副标：{d['sub'][:40]}")
        if d["stats"]: print(f"     数据：{d['stats']}")


if __name__ == "__main__":
    main()
