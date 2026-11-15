"""Download the corpus listed in data/sources.toml into data/raw/ (HTML, PDF via Chrome, DOCX via Markdown)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import httpx
from docx import Document

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
MD_DIR = RAW / "_md"
MANIFEST = DATA / "manifest.json"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
]


def load_sources() -> list[dict]:
    docs = tomllib.loads((DATA / "sources.toml").read_text())["doc"]
    for d in docs:
        d.setdefault("format", "html")
        d["doc_set"] = "booking_engine" if "booking-api-docs" in d["url"] else "open_api"
    return docs


def find_chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    return None


def parse_md(md: str) -> tuple[dict, str]:
    """Split the ReadMe Markdown export into front matter fields and body."""
    meta = {}
    m = re.match(r"^---\n(.*?)\n---\n", md, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line and not line.startswith(" "):
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        md = md[m.end():]
    title = re.search(r"^# (.+)$", md, re.M)
    meta["title"] = title.group(1).strip() if title else ""
    return meta, md


# --- Markdown export -> DOCX ------------------------------------------------------------------

_TAG = re.compile(
    r"</?(Anchor|Callout|Image|Embed|Table|Cards?|Accordion|HTMLBlock|thead|tbody|tr|th|td|table|ul|ol|li|p|div|span"
    r"|strong|em|b|i|code|a|br|summary|details|h[1-6])(\s[^>]*)?/?>"
)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _inline(text: str) -> str:
    text = _TAG.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = re.sub(r"<br\s*/?>", "", text)
    text = text.replace("&#x20;", " ").replace("\\[", "[").replace("\\]", "]")
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"(?<!\w)\*(\S[^*]*?)\*(?!\w)", r"\1", text)
    return text.strip()


def _text(el) -> str:
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True))


def _soup_to_md(node) -> list[str]:
    """Render embedded HTML (tables, collapsed <details>, lists) as Markdown lines."""
    lines: list[str] = []
    for el in node.find_all(["table", "details", "p", "li", "h1", "h2", "h3", "h4", "pre"]):
        container = el.find_parent(["table", "details", "li"])
        if container is not None and container is not node:
            continue  # rendered as part of its container
        if el.name == "table":
            rows = el.find_all("tr")
            if rows:
                cells = [[_text(c) for c in r.find_all(["th", "td"])] for r in rows]
            else:
                # ReadMe tables written as <p> headers followed by one <ul> per cell.
                heads = [_text(p) for p in el.find_all("p", recursive=False)]
                lists = [", ".join(_text(li) for li in ul.find_all("li")) for ul in el.find_all("ul", recursive=False)]
                if not heads or len(lists) % len(heads):
                    lines += heads + lists
                    continue
                cells = [heads] + [lists[i : i + len(heads)] for i in range(0, len(lists), len(heads))]
            lines += ["| " + " | ".join(cells[0]) + " |", "|" + "---|" * len(cells[0])]
            lines += ["| " + " | ".join(r) + " |" for r in cells[1:]]
            lines.append("")
        elif el.name == "details":
            summary = el.find("summary")
            if summary:
                lines += [f"#### {_text(summary)}", ""]
                summary.extract()
            lines += _soup_to_md(el) or [_text(el)]
            lines.append("")
        elif el.name == "li":
            lines.append(f"- {_text(el)}")
        elif el.name == "pre":
            lines += ["```", el.get_text(), "```"]
        elif el.name.startswith("h"):
            lines += [f"{'#' * int(el.name[1])} {_text(el)}", ""]
        elif _text(el):
            lines += [_text(el), ""]
    return lines


def _expand_html(body: str) -> str:
    from bs4 import BeautifulSoup

    def render(m: re.Match) -> str:
        inner = m.group(1).strip().strip("{}").strip().strip("`")
        return "\n" + "\n".join(_soup_to_md(BeautifulSoup(inner, "html.parser"))) + "\n"

    body = re.sub(r"<HTMLBlock>(.*?)</HTMLBlock>", render, body, flags=re.S)
    return re.sub(r"(<details>.*?</details>)", render, body, flags=re.S)


def md_to_docx(md: str, out: Path) -> None:
    """Headings become Heading styles, lists become List styles, pipe tables become Word tables."""
    _, body = parse_md(md)
    body = _expand_html(body)
    # ReadMe repeats title/description/tags/url as a pseudo front matter block after the H1.
    body = re.sub(r"\n\*\*\*\n\ntitle:.*?\n-{3,}\n", "\n", body, count=1, flags=re.S)
    doc = Document()
    lines = body.splitlines()
    i = 0
    in_code = False
    code: list[str] = []
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            if in_code:
                doc.add_paragraph("\n".join(code), style="No Spacing")
                code = []
            in_code = not in_code
            i += 1
            continue
        if in_code:
            code.append(line)
            i += 1
            continue
        s = re.sub(r"^>\s?", "", line.strip())  # callouts arrive as blockquotes
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i]):
                    rows.append([_inline(c) for c in lines[i].strip().strip("|").split("|")])
                i += 1
            width = max(len(r) for r in rows)
            table = doc.add_table(rows=len(rows), cols=width)
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    table.cell(r, c).text = cell
            continue
        h = re.match(r"^(#{1,4})\s+(.*)$", s)
        if h:
            doc.add_heading(_inline(h.group(2)), level=len(h.group(1)))
        elif re.match(r"^[*-]\s+", s):
            doc.add_paragraph(_inline(re.sub(r"^[*-]\s+", "", s)), style="List Bullet")
        elif re.match(r"^\d+\.\s+", s):
            doc.add_paragraph(_inline(re.sub(r"^\d+\.\s+", "", s)), style="List Number")
        elif s and not s.startswith("<") and s != "***":
            doc.add_paragraph(_inline(s))
        elif s.startswith("<") and not _TAG.fullmatch(s) and _inline(s):
            doc.add_paragraph(_inline(s))
        i += 1
    doc.save(out)


# --- fetching ---------------------------------------------------------------------------------

def print_pdf(chrome: str, url: str, out: Path) -> None:
    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-sandbox", f"--print-to-pdf={out}", url],
        check=True,
        capture_output=True,
        timeout=120,
    )


def fetch(force: bool = False) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    MD_DIR.mkdir(parents=True, exist_ok=True)
    chrome = find_chrome()
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    client = httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": "partner-support-rag (study project)"})
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stats = {"downloaded": 0, "skipped": 0, "pdf_fallback_to_html": 0}

    for d in load_sources():
        md_path = MD_DIR / f"{d['id']}.md"
        fmt = d["format"]
        if fmt == "pdf" and not chrome:
            fmt = "html"
            stats["pdf_fallback_to_html"] += 1
        out = RAW / f"{d['id']}.{fmt}"
        if out.exists() and md_path.exists() and not force:
            stats["skipped"] += 1
        else:
            md = client.get(d["url"] + ".md")
            md.raise_for_status()
            md_path.write_text(md.text, encoding="utf-8")
            if fmt == "html":
                r = client.get(d["url"])
                r.raise_for_status()
                out.write_bytes(r.content)
            elif fmt == "pdf":
                print_pdf(chrome, d["url"], out)
            elif fmt == "docx":
                md_to_docx(md.text, out)
            stats["downloaded"] += 1
            print(f"  fetched {out.name}")

        meta, _ = parse_md(md_path.read_text(encoding="utf-8"))
        entry = manifest.get(d["id"], {})
        entry.update(
            {
                "url": d["url"],
                "doc_set": d["doc_set"],
                "format": fmt,
                "title": meta.get("title", ""),
                "updated_at": meta.get("updatedAt", "")[:10],
                "raw_file": str(out.relative_to(ROOT)),
                "raw_sha256": hashlib.sha256(out.read_bytes()).hexdigest()[:16],
            }
        )
        entry.setdefault("first_fetched_at", now)
        manifest[d["id"]] = entry

    MANIFEST.write_text(json.dumps(dict(sorted(manifest.items())), indent=2, ensure_ascii=False) + "\n")
    stats["documents"] = len(manifest)
    return stats
