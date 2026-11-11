"""Wrap brief/index.html (page content only) into a full document at docs/index.html,
which GitHub Pages serves from the main branch's /docs folder.

    python brief/build_pages.py
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
content = (HERE / "brief" / "index.html").read_text()
head = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>
</head>
<body>
"""
(HERE / "docs").mkdir(exist_ok=True)
# the title, font link and styles at the top of the page belong in the head
split = content.index('<div class="wrap">')
head = head.replace("</head>", content[:split] + "</head>")
(HERE / "docs" / "index.html").write_text(head + content[split:] + "\n</body>\n</html>\n")
(HERE / "docs" / ".nojekyll").write_text("")
print("wrote docs/index.html")
