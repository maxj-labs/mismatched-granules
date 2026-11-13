"""Wrap each page's content into a full document under docs/, which GitHub Pages serves
from the main branch, and put the same navigation at the top of every page.

    python brief/build_pages.py
"""

import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PAGES = {
    "index.html": (HERE / "site" / "index.html", "Overview"),
    "storybook.html": (HERE / "storybook" / "index.html", "Storybook"),
    "evaluation.html": (HERE / "evaluation" / "index.html", "Every option, tested"),
    "brief.html": (HERE / "brief" / "index.html", "Real collections"),
}
HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}
.sitenav{display:flex;flex-wrap:wrap;gap:6px 22px;align-items:baseline;font-size:14px;padding-bottom:14px;border-bottom:1px solid var(--rule)}
.sitenav a{color:var(--ink-2);text-decoration:none}
.sitenav a:hover{color:var(--ink);text-decoration:underline}
.sitenav a[aria-current="page"]{color:var(--ink);font-weight:600}
.sitenav .repo{margin-left:auto}
h2,h3{scroll-margin-top:16px}
.anchor{margin-left:.4em;color:var(--ink-3);text-decoration:none;font-weight:400;opacity:0}
h2:hover .anchor,h3:hover .anchor,.anchor:focus{opacity:1}
@media (hover:none){.anchor{opacity:.5}}</style>
</head>
<body>
"""


def nav(current: str) -> str:
    links = ""
    for name, (_, label) in PAGES.items():
        href = "./" if name == "index.html" else name
        here = ' aria-current="page"' if name == current else ""
        links += f'<a href="{href}"{here}>{label}</a>'
    repo = '<a class="repo" href="https://github.com/maxj-labs/mismatched-granules">Repository</a>'
    return f'<nav class="sitenav" aria-label="Pages">{links}{repo}</nav>'


def anchors(body: str) -> str:
    """Give every h2 and h3 an id made from its text, and a link to itself, so sections can be linked to."""
    used: set[str] = set(re.findall(r'id="([^"]+)"', body))

    def add(m: re.Match) -> str:
        level, attrs, inner = m.groups()
        if "id=" in attrs:
            return m.group(0)
        text = html.unescape(re.sub("<[^>]+>", "", inner))
        words = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-").split("-")
        slug = ""
        for w in words:  # whole words, up to about sixty characters
            if slug and len(slug) + len(w) + 1 > 60:
                break
            slug = f"{slug}-{w}" if slug else w
        base, n = slug or "section", 2
        slug = base
        while slug in used:
            slug, n = f"{base}-{n}", n + 1
        used.add(slug)
        return f'<h{level}{attrs} id="{slug}">{inner}<a class="anchor" href="#{slug}" aria-label="Link to this section">#</a></h{level}>'

    return re.sub(r"<h([23])([^>]*)>(.*?)</h\1>", add, body, flags=re.S)


(HERE / "docs").mkdir(exist_ok=True)
for name, (source, _) in PAGES.items():
    content = source.read_text()
    # the title, font link and styles at the top of the page belong in the head
    split = content.index('<div class="wrap"')
    head = HEAD.replace("</head>", content[:split] + "</head>")
    body = anchors(content[split:])
    opening = body[: body.index(">") + 1]
    (HERE / "docs" / name).write_text(head + opening + nav(name) + body[len(opening):] + "\n</body>\n</html>\n")
    print("wrote docs/" + name)
(HERE / "docs" / ".nojekyll").write_text("")
