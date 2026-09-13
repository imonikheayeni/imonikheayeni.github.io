from pathlib import Path
from datetime import datetime, timezone
from subprocess import run
from urllib.parse import quote

BASE = "https://imonikheayeni.com"
ROOT = Path(__file__).resolve().parents[1]

# Public HTML files that should NOT be submitted to search engines.
# Keep the files in the repository if you still need them, but do not
# advertise backups, verification files, or error pages in sitemap.xml.
EXCLUDED_EXACT = {
    "404.html",
    "index_original.html",
}

EXCLUDED_SUFFIXES = (
    "_delete.html",
    "_original.html",
)


def should_exclude(path: Path) -> bool:
    rel = path.relative_to(ROOT)

    # Ignore hidden folders such as .git and .github.
    if any(part.startswith(".") for part in rel.parts):
        return True

    name = path.name.lower()

    if name in EXCLUDED_EXACT:
        return True

    if name.endswith(EXCLUDED_SUFFIXES):
        return True

    # Google verification HTML files need to stay publicly accessible,
    # but they do not belong in the sitemap.
    if name.startswith("google") and name.endswith(".html"):
        return True

    return False


def page_url(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()

    if rel == "index.html":
        url_path = "/"
    elif rel.endswith("/index.html"):
        # learning-library/index.html -> /learning-library/
        url_path = "/" + rel[:-10]
    else:
        url_path = "/" + rel

    return BASE + quote(url_path, safe="/:")


def last_modified(path: Path) -> str:
    """Return the date of the latest Git commit that changed this file."""
    rel = path.relative_to(ROOT).as_posix()

    result = run(
        ["git", "log", "-1", "--format=%cs", "--", rel],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    commit_date = result.stdout.strip()
    if result.returncode == 0 and commit_date:
        return commit_date

    # Fallback for local runs outside a full Git checkout.
    return datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    ).date().isoformat()


pages = []

for path in ROOT.rglob("*.html"):
    if should_exclude(path):
        continue

    pages.append((page_url(path), last_modified(path)))

# Remove any accidental duplicate URLs and keep the sitemap deterministic.
pages = sorted(dict(pages).items())

lines = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
]

for url, lastmod in pages:
    lines.extend(
        [
            "  <url>",
            f"    <loc>{url}</loc>",
            f"    <lastmod>{lastmod}</lastmod>",
            "  </url>",
        ]
    )

lines.append("</urlset>")

(ROOT / "sitemap.xml").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print(f"Wrote {len(pages)} URLs to sitemap.xml")
