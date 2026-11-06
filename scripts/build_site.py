#!/usr/bin/env python3
"""Build the public AI news site from English digest markdown files.

Usage (from the repository root):
    python3 scripts/build_site.py \
        --digests site/digests \
        --fetcher src/news/fetcher.py \
        --out site/public \
        --site-url https://linfubin5667.github.io/ai-news-bot/

Only the Python standard library is used, so the script runs on a bare
GitHub Actions runner with no extra installs.

Input digest format (what config.yaml stage 2 produces, in English):
    ## Category Name
    ### Headline
    <4-6 sentence summary>
    Source: [Source Name](https://example.com/article)

    ---
    (repeated per item)

Output (all static, no server, no JavaScript required):
    index.html                  latest digest (homepage)
    <YYYY-MM-DD>/index.html     one page per digest date (archive)
    archive/index.html          list of all dates, newest first
    sources/index.html          every RSS source, parsed from fetcher.py
    about/index.html            methodology, attribution, maintainer
    sitemap.xml / robots.txt    for search engines

Rules the builder enforces:
- Each item shows headline + summary + a link out to the original.
  Full article text is never copied onto the site.
- All user-controlled text is HTML-escaped; only the Source URL is
  used as a link href (and only when it starts with http/https).
- An empty digests folder still builds a valid homepage that says
  no digest has been published yet, so a first deploy never 404s.
"""

import argparse
import html
import re
import sys
from datetime import date
from pathlib import Path

SITE_NAME = "Daily AI Brief"
SITE_TAGLINE = "A daily, human-readable selection of AI news, summarised in English."
REPO_URL = "https://github.com/linfubin5667/ai-news-bot"

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
margin:0;color:#1a1a1a;background:#fff;line-height:1.6}
header{border-bottom:3px solid #111;padding:20px 16px 12px;max-width:760px;margin:0 auto}
header h1{font-size:28px;margin:0;line-height:1.2}
header h1 a{color:#111;text-decoration:none}
header p{color:#555;margin:6px 0 0}
nav{max-width:760px;margin:0 auto;padding:10px 16px;font-size:15px}
nav a{margin-right:16px;color:#0b5fff}
main{max-width:760px;margin:0 auto;padding:8px 16px 40px}
h2{font-size:21px;border-bottom:1px solid #ddd;padding-bottom:6px;margin-top:32px}
h3{font-size:18px;margin:22px 0 6px;line-height:1.35}
article p{margin:6px 0}
.source{font-size:14px;color:#555}
.source a{color:#0b5fff}
.meta{color:#666;font-size:14px}
footer{border-top:1px solid #ddd;max-width:760px;margin:0 auto;padding:16px;font-size:13px;color:#666}
ul.dates{list-style:none;padding:0}
ul.dates li{padding:8px 0;border-bottom:1px solid #eee}
.notice{background:#fff8e1;border:1px solid #f0d48a;padding:12px 14px;border-radius:6px;font-size:14px}
@media (max-width:600px){header h1{font-size:23px}h2{font-size:19px}}
"""

SOURCE_RE = re.compile(
    r"^Source:\s*(?:\*\*)?\s*\[(?P<name>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)",
    re.IGNORECASE,
)
CATEGORY_RE = re.compile(r"^##\s+(?P<name>.+?)\s*$")
TITLE_RE = re.compile(r"^###\s+(?P<title>.+?)\s*$")
DATE_FILE_RE = re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\.md$")
DATE_STAMP_RE = re.compile(r"digest_(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})_")


def parse_digest(text):
    """Parse one digest markdown file into [(category, {title, summary, source, url})]."""
    items = []
    category = "AI News"
    title = None
    summary_lines = []
    source_name = ""
    source_url = ""

    def flush():
        nonlocal title, summary_lines, source_name, source_url
        if title:
            items.append({
                "category": category,
                "title": title.strip(),
                "summary": " ".join(s.strip() for s in summary_lines if s.strip()),
                "source": source_name.strip(),
                "url": source_url.strip(),
            })
        title = None
        summary_lines = []
        source_name = ""
        source_url = ""

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == "---":
            continue
        m = CATEGORY_RE.match(line)
        if m and not line.startswith("###"):
            flush()
            category = m.group("name").strip()
            continue
        m = TITLE_RE.match(line)
        if m:
            flush()
            title = m.group("title")
            continue
        m = SOURCE_RE.match(line)
        if m:
            source_name = m.group("name")
            source_url = m.group("url")
            continue
        if title is not None:
            summary_lines.append(line)
    flush()
    return items


def digest_date(path):
    """Return the digest date from the filename, or None if it has no date."""
    m = DATE_FILE_RE.match(path.name)
    if m:
        return date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
    m = DATE_STAMP_RE.search(path.name)
    if m:
        return date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
    return None


def load_digests(digests_dir):
    """Return {date: [items]} sorted newest first; undated files are skipped."""
    digests = {}
    if not digests_dir.is_dir():
        return digests
    for path in sorted(digests_dir.glob("*.md")):
        d = digest_date(path)
        if d is None:
            print(f"warning: skipping {path.name}: no YYYY-MM-DD date in filename",
                  file=sys.stderr)
            continue
        items = parse_digest(path.read_text(encoding="utf-8"))
        if items:
            digests[d] = items
        else:
            print(f"warning: {path.name}: no items parsed, skipped", file=sys.stderr)
    return dict(sorted(digests.items(), reverse=True))


def parse_sources(fetcher_path):
    """Parse {group: [(name, url)]} from src/news/fetcher.py.

    The fetcher is the single source of truth for the source list, so the
    Sources page always matches what the bot actually reads. Returns {} if
    the file is missing or unparsable.
    """
    if not fetcher_path or not fetcher_path.is_file():
        return {}
    text = fetcher_path.read_text(encoding="utf-8")
    groups = {}
    dict_re = re.compile(r"self\.(\w+_feeds)\s*=\s*\{(.*?)\n        \}", re.DOTALL)
    entry_re = re.compile(r'"([^"]+)"\s*:\s*"(https?://[^"]+)"')
    for m in dict_re.finditer(text):
        entries = [(n, u) for n, u in entry_re.findall(m.group(2))]
        if entries:
            groups[m.group(1)] = entries
    return groups


def page(site_url, title, body, description=""):
    esc_title = html.escape(title)
    desc = html.escape(description or SITE_TAGLINE)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc_title} | {SITE_NAME}</title>
<meta name="description" content="{desc}">
<style>{CSS}</style>
</head>
<body>
<header>
<h1><a href="{site_url}/">{SITE_NAME}</a></h1>
<p>{SITE_TAGLINE}</p>
</header>
<nav><a href="{site_url}/">Today</a><a href="{site_url}/archive/">Archive</a><a href="{site_url}/sources/">Sources</a><a href="{site_url}/about/">About</a></nav>
<main>
{body}
</main>
<footer>
<p>{SITE_NAME} is maintained by <a href="https://github.com/linfubin5667">Fubin Lin</a>.
Summaries are AI-generated from the sources linked under each item; read the
original article for the full story. The bot's code is open source at
<a href="{REPO_URL}">{REPO_URL}</a>.</p>
</footer>
</body>
</html>
"""


def item_html(item):
    title = html.escape(item["title"])
    summary = html.escape(item["summary"]) if item["summary"] else ""
    if item["url"]:
        src = html.escape(item["source"] or "Original source")
        link = html.escape(item["url"], quote=True)
        source_html = (f'<p class="source">Source: '
                       f'<a href="{link}" rel="noopener" target="_blank">{src}</a></p>')
    else:
        source_html = '<p class="source">Source: not linked</p>'
    return f"<article>\n<h3>{title}</h3>\n<p>{summary}</p>\n{source_html}\n</article>\n"


def digest_body(digest_date, items, is_latest):
    grouped = {}
    for item in items:
        grouped.setdefault(item["category"], []).append(item)
    parts = []
    if is_latest:
        parts.append(f'<p class="meta">Latest digest: {digest_date.isoformat()} '
                     f"&middot; {len(items)} stories</p>")
    else:
        parts.append(f'<p class="meta">Digest for {digest_date.isoformat()} '
                     f"&middot; {len(items)} stories</p>")
    parts.append('<p class="notice">Each story below is a short summary. '
                 "Click the source link to read the full article on the "
                 "publisher's own site.</p>")
    for category, cat_items in grouped.items():
        parts.append(f"<h2>{html.escape(category)}</h2>")
        parts.extend(item_html(i) for i in cat_items)
    return "\n".join(parts)


def build(digests_dir, fetcher_path, out_dir, site_url):
    digests = load_digests(digests_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = site_url.rstrip("/")

    if digests:
        latest_date, latest_items = next(iter(digests.items()))
        index_body = digest_body(latest_date, latest_items, is_latest=True)
    else:
        latest_date, latest_items = None, []
        index_body = ("<h2>No digest published yet</h2>"
                      "<p>The first English digest will appear here after the "
                      "next scheduled run. In the meantime, see "
                      '<a href="./about/">About</a> for how this site works.</p>')
    (out_dir / "index.html").write_text(
        page(base, "Today's AI News" if latest_date else SITE_NAME, index_body),
        encoding="utf-8")

    for d, items in digests.items():
        day_dir = out_dir / d.isoformat()
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "index.html").write_text(
            page(base, f"AI News — {d.isoformat()}",
                 digest_body(d, items, is_latest=False)),
            encoding="utf-8")

    archive_parts = ["<h2>Archive</h2>"]
    if digests:
        archive_parts.append('<ul class="dates">')
        for d, items in digests.items():
            archive_parts.append(
                f'<li><a href="{base}/{d.isoformat()}/">{d.isoformat()}</a>'
                f" &middot; {len(items)} stories</li>")
        archive_parts.append("</ul>")
    else:
        archive_parts.append("<p>No digests yet.</p>")
    archive_dir = out_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "index.html").write_text(
        page(base, "Archive", "\n".join(archive_parts)), encoding="utf-8")

    groups = parse_sources(fetcher_path)
    total = sum(len(v) for v in groups.values())
    source_parts = ["<h2>Sources</h2>",
                    f"<p>This site summarises stories from {total} RSS sources "
                    "read by the bot every day. Each story links back to its "
                    "original publisher; nothing is republished in full.</p>"]
    labels = {
        "rss_feeds": "International",
        "chinese_feeds": "Chinese",
        "japanese_feeds": "Japanese",
        "french_feeds": "French",
        "spanish_feeds": "Spanish",
        "german_feeds": "German",
        "korean_feeds": "Korean",
        "portuguese_feeds": "Portuguese",
        "italian_feeds": "Italian",
        "russian_feeds": "Russian",
        "dutch_feeds": "Dutch",
        "arabic_feeds": "Arabic",
        "hindi_feeds": "Hindi",
    }
    if groups:
        for key, entries in groups.items():
            source_parts.append(f"<h2>{html.escape(labels.get(key, key))} "
                                f"({len(entries)})</h2><ul>")
            for name, url in entries:
                source_parts.append(
                    f'<li>{html.escape(name)} — '
                    f'<a href="{html.escape(url, quote=True)}" rel="noopener" '
                    f'target="_blank">{html.escape(url)}</a></li>')
            source_parts.append("</ul>")
    else:
        source_parts.append("<p>Source list unavailable: the fetcher file "
                            "could not be read when this page was built.</p>")
    sources_dir = out_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    (sources_dir / "index.html").write_text(
        page(base, "Sources", "\n".join(source_parts)), encoding="utf-8")

    about_body = """<h2>About</h2>
<p>Daily AI Brief is a public, English-language digest of AI news. Every
morning (09:00 Asia/Shanghai) a bot reads dozens of RSS feeds, an AI model
selects 15–20 of the most important stories, and writes a short summary of
each one. The result is published here as a dated page, so the archive grows
by one page a day.</p>
<h2>How it works</h2>
<ol>
<li>The bot fetches the latest items from the RSS feeds listed on the
<a href="../sources/">Sources</a> page.</li>
<li>It removes duplicates, scores items for quality, and selects 15–20 stories.</li>
<li>An AI model (DeepSeek by default) writes a 4–6 sentence summary of each
story in English, grouped by topic.</li>
<li>The summary is saved as a dated file and this site is rebuilt and published.</li>
</ol>
<h2>Attribution and copyright</h2>
<p>Only headlines, short summaries and links are published here. Full article
text stays on the publisher's own site — click the source link under any story
to read it there. If you are a publisher and want a feed removed, open an
issue on the GitHub repository.</p>
<h2>Separate from the Chinese email digest</h2>
<p>The same bot also sends a Chinese daily email digest. That email is a
separate product with its own language and delivery; this site only publishes
the English version, and changing the site never changes the email.</p>
<h2>Maintainer</h2>
<p>Built and maintained by <a href="https://github.com/linfubin5667">Fubin Lin</a>.
Code: <a href="https://github.com/linfubin5667/ai-news-bot">github.com/linfubin5667/ai-news-bot</a>.</p>"""
    about_dir = out_dir / "about"
    about_dir.mkdir(parents=True, exist_ok=True)
    (about_dir / "index.html").write_text(
        page(base, "About", about_body), encoding="utf-8")

    urls = [base + "/"]
    urls += [f"{base}/{d.isoformat()}/" for d in digests]
    urls += [f"{base}/archive/", f"{base}/sources/", f"{base}/about/"]
    sitemap = ('<?xml version="1.0" encoding="UTF-8"?>\n'
               '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
               + "".join(f"  <url><loc>{html.escape(u)}</loc></url>\n" for u in urls)
               + "</urlset>\n")
    (out_dir / "sitemap.xml").write_text(sitemap, encoding="utf-8")
    (out_dir / "robots.txt").write_text(
        "User-agent: *\nAllow: /\n"
        f"Sitemap: {base}/sitemap.xml\n", encoding="utf-8")

    print(f"built {len(digests)} digest(s), {total} sources, "
          f"{sum(len(v) for v in digests.values())} stories -> {out_dir}")
    if latest_date:
        print(f"latest: {latest_date.isoformat()} ({len(latest_items)} stories)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digests", type=Path, default=Path("site/digests"))
    parser.add_argument("--fetcher", type=Path, default=Path("src/news/fetcher.py"))
    parser.add_argument("--out", type=Path, default=Path("site/public"))
    parser.add_argument("--site-url", default="https://linfubin5667.github.io/ai-news-bot/")
    args = parser.parse_args(argv)
    return build(args.digests, args.fetcher, args.out, args.site_url)


if __name__ == "__main__":
    raise SystemExit(main())
