"""文章模块构建：扫描 articles/*.md → 生成清单 + 复制到网站

产出：
  site/articles/<slug>.md        原文（可下载）
  site/articles/<slug>.png       封面
  articles.json（并入 web_data）  清单：标题/日期/摘要/字数/封面/GitHub编辑链接
"""
import sys, re, json, shutil, glob
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from config import BASE

ART = BASE.parent / "articles"
SITE = Path(__file__).resolve().parent.parent / "site"   # 跨平台：仓库内 site/
OUT = SITE / "articles"
GH_REPO = "Baggio200cn/dobo-football"          # 建仓后用于「在 GitHub 编辑」


def parse(p):
    s = Path(p).read_text(encoding="utf-8")
    meta = {}
    m = re.match(r"^---\n(.*?)\n---\n", s, re.S)
    body = s
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"')
        body = s[m.end():]
    title = meta.get("title", Path(p).stem)
    # 摘要：正文第一个非标题非空段
    lead = ""
    for para in body.split("\n\n"):
        t = para.strip()
        if t and not t.startswith("#") and not t.startswith("---") and not t.startswith("|"):
            lead = re.sub(r"[*`>]", "", t).strip()
            break
    words = len(re.sub(r"\s", "", body))
    # 小标题目录
    heads = re.findall(r"^##\s+(.+)$", body, re.M)
    # slug 用纯 ASCII（中文文件名在部分环境需 URL 编码，易出问题）
    num = re.match(r"^(\d+)", Path(p).stem)
    slug = f"post{num.group(1)}" if num else re.sub(r"[^\w-]", "", Path(p).stem) or "post"
    return {
        "slug": slug,
        "file": Path(p).name,
        "title": title,
        "date": meta.get("date", ""),
        "style": meta.get("style", ""),
        "period": meta.get("period", ""),
        "status": meta.get("status", ""),
        "tags": [t.strip() for t in meta.get("tags", "").strip("[]").split(",") if t.strip()],
        "lead": lead[:160],
        "words": words,
        "heads": heads[:8],
        "cover": f"/articles/{Path(p).stem}.png",
        "md": f"/articles/{Path(p).stem}.md",
        "edit": f"https://github.com/{GH_REPO}/edit/main/articles/{Path(p).name}",  # 编辑链接用原名
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(glob.glob(str(ART / "*.md"))):
        d = parse(p)
        shutil.copy(p, OUT / f"{d['slug']}.md")
        cov = Path(p).with_suffix(".cover.png")
        if cov.exists():
            shutil.copy(cov, OUT / f"{d['slug']}.png")
        else:
            d["cover"] = None
        items.append(d)
    items.sort(key=lambda x: x["date"], reverse=True)

    # 并入 web_data.json
    wp = BASE / "web_data.json"
    web = json.loads(wp.read_text(encoding="utf-8")) if wp.exists() else {}
    web["articles"] = items
    wp.write_text(json.dumps(web, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy(wp, SITE / "data.json")

    print(f"✅ 文章模块：{len(items)} 篇")
    for d in items:
        print(f"   {d['date']}  {d['title'][:34]:<36} {d['words']:>5} 字"
              f"  封面{'✓' if d['cover'] else '✗'}  {len(d['heads'])} 节")
    return items


if __name__ == "__main__":
    main()
