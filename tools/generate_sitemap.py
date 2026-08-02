from pathlib import Path
from datetime import date
from urllib.parse import quote
BASE="https://imonikheayeni.com"
ROOT=Path(__file__).resolve().parents[1]
exclude={"404.html"}
files=[]
for p in ROOT.rglob("*.html"):
    if any(part.startswith(".") for part in p.parts) or p.name in exclude: continue
    rel=p.relative_to(ROOT).as_posix()
    url="/" if rel=="index.html" else ("/"+rel[:-10] if rel.endswith("/index.html") else "/"+rel)
    files.append(BASE+quote(url, safe="/:"))
today=date.today().isoformat()
lines=["<?xml version=\"1.0\" encoding=\"UTF-8\"?>","<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">"]
for url in sorted(set(files)):
    priority="1.0" if url==BASE+"/" else ("0.9" if "/learning-library" in url else "0.7")
    lines.append(f"  <url><loc>{url}</loc><lastmod>{today}</lastmod><changefreq>monthly</changefreq><priority>{priority}</priority></url>")
lines.append("</urlset>")
(ROOT/"sitemap.xml").write_text("\n".join(lines),encoding="utf-8")
print(f"Wrote {len(files)} URLs")
