"""Wrap each page's content (brief/index.html, evaluation/index.html) into a full document under docs/,
which GitHub Pages serves from the main branch's /docs folder.

    python brief/build_pages.py
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PAGES = {"index.html": HERE / "brief" / "index.html", "evaluation.html": HERE / "evaluation" / "index.html", "storybook.html": HERE / "storybook" / "index.html"}
HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>
</head>
<body>
"""
(HERE / "docs").mkdir(exist_ok=True)
for name, source in PAGES.items():
    content = source.read_text()
    # the title, font link and styles at the top of the page belong in the head
    split = content.index('<div class="wrap"')
    head = HEAD.replace("</head>", content[:split] + "</head>")
    (HERE / "docs" / name).write_text(head + content[split:] + "\n</body>\n</html>\n")
    print("wrote docs/" + name)
(HERE / "docs" / ".nojekyll").write_text("")
