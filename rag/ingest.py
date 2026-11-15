"""Raw documents -> clean structured text -> deduplicated -> chunks with metadata (data/processed/chunks.jsonl)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# tiktoken reads its cache location at import time; keep it in the project so CI can cache it.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(__file__).resolve().parent.parent / ".cache" / "tiktoken"))

import tiktoken  # noqa: E402
from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document as DocxDocument
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
MANIFEST = DATA / "manifest.json"
PIPELINE_VERSION = "ingest-1"

_enc = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(_enc.encode(text))


@dataclass
class Doc:
    doc_id: str
    title: str
    url: str
    doc_set: str
    fmt: str
    updated_at: str
    text: str = ""
    content_hash: str = ""
    duplicate_of: str | None = None
    near_duplicates: list[tuple[str, float]] = field(default_factory=list)


# --- extraction: every format ends up as Markdown-like text with "#" headings -----------------

_HTML_DROP = "script, style, svg, button, nav, .heading-anchor-icon, .PageThumbs, .rm-Pagination, .rm-Aside"


def _inline_text(node: Tag) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _html_blocks(node: Tag, out: list[str]) -> None:
    for child in node.children:
        if isinstance(child, NavigableString):
            if child.strip():
                out.append(child.strip())
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name in {"h1", "h2", "h3", "h4"}:
            text = _inline_text(child)
            if text:
                out.append("#" * int(name[1]) + " " + text)
        elif name == "p":
            text = _inline_text(child)
            if text:
                out.append(text)
        elif name in {"ul", "ol"}:
            items = []
            for i, li in enumerate(child.find_all("li", recursive=False), 1):
                text = _inline_text(li)
                if text:
                    items.append(("- " if name == "ul" else f"{i}. ") + text)
            if items:
                out.append("\n".join(items))
        elif name == "table":
            rows = []
            for tr in child.find_all("tr"):
                cells = [_inline_text(c).replace("|", "/") for c in tr.find_all(["th", "td"])]
                if cells:
                    rows.append("| " + " | ".join(cells) + " |")
            if rows:
                header_sep = "|" + "---|" * (rows[0].count("|") - 1)
                out.append("\n".join([rows[0], header_sep, *rows[1:]]))
        elif name == "pre":
            code = child.get_text().strip("\n")
            if code.strip():
                out.append("```\n" + code + "\n```")
        else:
            _html_blocks(child, out)


def extract_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    article = soup.select_one(".rm-Article") or soup.find("article") or soup.body
    for bad in article.select(_HTML_DROP):
        bad.decompose()
    blocks: list[str] = []
    header = article.select_one(".rm-PageHeader")
    if header:
        h1 = header.find("h1")
        if h1:
            blocks.append("# " + _inline_text(h1))
        desc = header.select_one(".rm-Markdown") or header.find("p")
        if desc:
            blocks.append(_inline_text(desc))
    # The page header has its own small .rm-Markdown (the subtitle); the article body is the largest one.
    bodies = [m for m in article.select(".rm-Markdown") if not (header and header in m.parents)]
    body = max(bodies, key=lambda m: len(m.get_text())) if bodies else article
    _html_blocks(body, blocks)
    return "\n\n".join(blocks)


_PDF_HEADER = re.compile(r"^\d{1,2}/\d{1,2}/\d{2},\s+\d{1,2}:\d{2}\s*(AM|PM)\b.*$")
_PDF_FOOTER = re.compile(r"^https?://\S+\s+\d+/\d+$")
_PDF_CODE_LINE = re.compile(r'^("[\w$.-]+"\s*:|[{}\[\]]{1,2},?$|"[^"]*",?$|curl\s|--header|--url)')
_PDF_BOILERPLATE = re.compile(r"^(Did this page help you\?.*|Updated\s*\d+\s+\w+\s+ago|Copy Page|Yes|No|JSON|cURL)$", re.I)


def _looks_like_heading(line: str, next_line: str) -> bool:
    if len(line) > 70 or len(line.split()) > 9 or line.endswith((".", ",", ":", ";", ")")):
        return False
    if not line[:1].isupper() or line.startswith(("•", "-")) or "{" in line or '"' in line:
        return False
    words = [w for w in re.findall(r"[A-Za-z][\w'-]*", line)]
    capitalised = sum(w[0].isupper() for w in words)
    return bool(words) and capitalised / len(words) >= 0.5 and (not next_line or next_line[:1].isupper())


def extract_pdf(path: Path, known_titles: set[str]) -> str:
    pages = [p.extract_text() or "" for p in PdfReader(path).pages]
    page_lines = [[l.strip() for l in p.splitlines() if l.strip()] for p in pages]
    # Lines that repeat on most pages are browser headers/footers, whatever their exact wording.
    counts = Counter(l for lines in page_lines for l in set(lines))
    repeated = {l for l, c in counts.items() if len(pages) > 2 and c >= 0.6 * len(pages)}
    lines: list[str] = []
    for page in page_lines:
        for l in page:
            if l in repeated or _PDF_HEADER.match(l) or _PDF_FOOTER.match(l) or _PDF_BOILERPLATE.match(l):
                continue
            # Previous/next navigation: other documents' titles, alone or side by side on one line.
            if l in known_titles or (sum(t in l for t in known_titles) >= 2 and len(l) < 160):
                continue
            lines.append(l)

    blocks: list[str] = []
    para: list[str] = []
    code: list[str] = []

    def flush() -> None:
        if para:
            blocks.append(" ".join(para))
            para.clear()

    def flush_code() -> None:
        if code:
            blocks.append("```\n" + "\n".join(code) + "\n```")
            code.clear()

    for i, l in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        # JSON examples lose their indentation in PDF text; keep them together as a code block.
        if _PDF_CODE_LINE.match(l) or (code and (l.endswith((",", "{", "[")) or l in {"}", "]", "},", "],"})):
            flush()
            code.append(l)
            continue
        flush_code()
        if l.startswith(("•", "●", "◦")):
            flush()
            blocks.append("- " + l.lstrip("•●◦ ").strip())
        elif i == 0:
            blocks.append("# " + l)
        elif _looks_like_heading(l, nxt):
            flush()
            blocks.append("## " + l)
        else:
            if para and para[-1].endswith("-") and l[:1].islower():
                para[-1] = para[-1][:-1] + l
                continue
            para.append(l)
            if l.endswith((".", "!", "?", ":")) and (not nxt or nxt[:1].isupper()):
                flush()
    flush()
    flush_code()
    return "\n\n".join(blocks)


def extract_docx(path: Path) -> str:
    doc = DocxDocument(path)
    blocks: list[str] = []
    numbered = 0
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            p = DocxParagraph(child, doc)
            text = p.text.strip()
            style = p.style.name if p.style is not None else ""
            if not text:
                continue
            if style.startswith("Heading"):
                level = int(re.sub(r"\D", "", style) or 2)
                blocks.append("#" * min(level, 4) + " " + text)
                numbered = 0
            elif style == "List Bullet":
                blocks.append("- " + text)
            elif style == "List Number":
                numbered += 1
                blocks.append(f"{numbered}. " + text)
            elif style == "No Spacing":
                blocks.append("```\n" + text + "\n```")
            else:
                blocks.append(text)
                numbered = 0
        elif tag == "tbl":
            t = DocxTable(child, doc)
            rows = ["| " + " | ".join(c.text.strip().replace("|", "/") for c in r.cells) + " |" for r in t.rows]
            if rows:
                rows.insert(1, "|" + "---|" * len(t.rows[0].cells))
                blocks.append("\n".join(rows))
    # Consecutive list items belong to one block.
    return re.sub(r"\n\n(?=(- |\d+\. ))", "\n", "\n\n".join(blocks))


# --- cleaning and deduplication ---------------------------------------------------------------

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    canonical = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + k]) for i in range(max(0, len(words) - k + 1))}


def find_near_duplicates(docs: list[Doc], threshold: float) -> list[tuple[str, str, float]]:
    sh = {d.doc_id: _shingles(d.text) for d in docs if not d.duplicate_of}
    ids = sorted(sh)
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            inter = len(sh[a] & sh[b])
            if inter:
                j = inter / len(sh[a] | sh[b])
                if j >= threshold:
                    pairs.append((a, b, round(j, 3)))
    return pairs


# --- chunking ---------------------------------------------------------------------------------

def split_blocks(text: str) -> list[str]:
    """Paragraphs, lists, tables and fenced code blocks; a code block is never split here."""
    blocks, buf, in_code = [], [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            buf.append(line)
            if in_code:
                blocks.append("\n".join(buf))
                buf = []
            in_code = not in_code
            continue
        if in_code:
            buf.append(line)
        elif not line.strip():
            if buf:
                blocks.append("\n".join(buf))
                buf = []
        elif line.startswith("#"):
            if buf:
                blocks.append("\n".join(buf))
                buf = []
            blocks.append(line)
        else:
            buf.append(line)
    if buf:
        blocks.append("\n".join(buf))
    return blocks


def _split_oversized(block: str, max_tokens: int) -> list[str]:
    if n_tokens(block) <= max_tokens:
        return [block]
    lines = block.split("\n")
    if lines[0].startswith("|") and len(lines) > 2:  # table: keep the header row on every part
        head, rows = lines[:2], lines[2:]
        unit, prefix, suffix = rows, head, []
    elif lines[0].startswith("```"):
        unit, prefix, suffix = lines[1:-1], [lines[0]], ["```"]
    elif len(lines) > 1:
        unit, prefix, suffix = lines, [], []
    else:
        unit, prefix, suffix = re.split(r"(?<=[.!?])\s+", block), [], []
    parts, cur = [], []
    for u in unit:
        if cur and n_tokens("\n".join(prefix + cur + [u] + suffix)) > max_tokens:
            parts.append("\n".join(prefix + cur + suffix))
            cur = []
        cur.append(u)
    if cur:
        parts.append("\n".join(prefix + cur + suffix))
    # Last resort (e.g. one huge unpunctuated line): hard token windows.
    out = []
    for p in parts:
        if n_tokens(p) <= max_tokens:
            out.append(p)
        else:
            ids = _enc.encode(p)
            out.extend(_enc.decode(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens))
    return out


def chunk_structure(text: str, max_tokens: int, min_tokens: int) -> list[dict]:
    """Cut at headings; pack whole blocks (paragraph/list/table/code) up to max_tokens; merge tiny sections forward."""
    sections: list[tuple[list[str], list[str]]] = []
    stack: list[tuple[int, str]] = []
    cur_blocks: list[str] = []

    def path() -> list[str]:
        return [t for _, t in stack]

    for b in split_blocks(text):
        m = re.match(r"^(#{1,4}) (.+)$", b)
        if m:
            if cur_blocks:
                sections.append((path(), cur_blocks))
                cur_blocks = []
            level = len(m.group(1))
            stack = [(l, t) for l, t in stack if l < level] + [(level, m.group(2).strip())]
        else:
            cur_blocks.append(b)
    if cur_blocks:
        sections.append((path(), cur_blocks))

    chunks: list[dict] = []
    carry: dict | None = None
    for sec_path, blocks in sections:
        units = [u for b in blocks for u in _split_oversized(b, max_tokens)]
        packed: list[list[str]] = []
        cur: list[str] = []
        for u in units:
            if cur and n_tokens("\n\n".join(cur + [u])) > max_tokens:
                packed.append(cur)
                cur = []
            cur.append(u)
        if cur:
            packed.append(cur)
        for p in packed:
            chunk = {"section_path": sec_path, "text": "\n\n".join(p)}
            if carry:
                merged = carry["text"] + "\n\n" + (sec_path[-1].rstrip(":") + ":\n" if sec_path else "") + chunk["text"]
                # A tiny section may overflow its neighbour by up to min_tokens rather than stand alone.
                if n_tokens(merged) <= max_tokens + min_tokens:
                    chunk = {"section_path": carry["section_path"], "text": merged, "merged_sections": [carry["section_path"], sec_path]}
                elif chunks and n_tokens(chunks[-1]["text"] + carry["text"]) <= max_tokens + min_tokens:
                    chunks[-1]["text"] += "\n\n" + carry["text"]
                else:
                    chunks.append(carry)
                carry = None
            if n_tokens(chunk["text"]) < min_tokens:
                carry = chunk
            else:
                chunks.append(chunk)
    if carry:
        if chunks and n_tokens(chunks[-1]["text"] + carry["text"]) <= max_tokens + min_tokens:
            chunks[-1]["text"] += "\n\n" + carry["text"]
        else:
            chunks.append(carry)
    return chunks


def chunk_fixed(text: str, size: int, overlap: int) -> list[dict]:
    """Baseline: a sliding window of `size` tokens; the section is whatever heading came last before the window."""
    ids = _enc.encode(text)
    headings = [(m.start(), m.group(2)) for m in re.finditer(r"^(#{1,4}) (.+)$", text, re.M)]
    chunks, start = [], 0
    while start < len(ids):
        window = ids[start : start + size]
        char_start = len(_enc.decode(ids[:start]))
        section = [h for pos, h in headings if pos <= char_start][-1:] or []
        chunks.append({"section_path": section, "text": _enc.decode(window)})
        if start + size >= len(ids):
            break
        start += size - overlap
    return chunks


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def chunk_quality(chunks: list[dict]) -> dict:
    """How often a chunk starts or ends mid-thought, or cuts a table/code block."""
    toks = sorted(n_tokens(c["text"]) for c in chunks)
    clean_start = clean_end = broken = 0
    for c in chunks:
        t = c["text"].strip()
        first = t[:1]
        clean_start += first.isupper() or first.isdigit() or first in "-|`#*\"'(["
        clean_end += t.endswith((".", "!", "?", ":", "```", "|", ")", "}", "]")) or t.split("\n")[-1].startswith(("- ", "|"))
        broken += t.count("```") % 2 == 1 or (t.startswith("|") is False and "\n|" in t and not re.search(r"^\|.*\|\n\|[-|]+\|", t, re.M))
    n = len(chunks) or 1
    return {
        "chunks": len(chunks),
        "tokens_mean": round(sum(toks) / n, 1),
        "tokens_p95": toks[int(0.95 * (len(toks) - 1))] if toks else 0,
        "tokens_min": toks[0] if toks else 0,
        "clean_start_pct": round(100 * clean_start / n, 1),
        "clean_end_pct": round(100 * clean_end / n, 1),
        "cut_table_or_code_pct": round(100 * broken / n, 1),
    }


# --- pipeline ---------------------------------------------------------------------------------

def load_docs(manifest: dict) -> list[Doc]:
    titles = {v["title"] for v in manifest.values() if v.get("title")}
    docs = []
    for doc_id, m in sorted(manifest.items()):
        path = ROOT / m["raw_file"]
        if not path.exists():
            raise SystemExit(f"{path} is missing: run `make fetch` first")
        fmt = m["format"]
        raw = extract_html(path) if fmt == "html" else extract_pdf(path, titles - {m["title"]}) if fmt == "pdf" else extract_docx(path)
        d = Doc(doc_id, m["title"], m["url"], m["doc_set"], fmt, m["updated_at"], normalize(raw))
        d.content_hash = content_hash(d.text)
        docs.append(d)
    return docs


def chunk_doc(d: Doc, cfg: dict, strategy: str) -> list[dict]:
    c = cfg["chunking"]
    raw = chunk_structure(d.text, c["max_tokens"], c["min_tokens"]) if strategy == "structure" else chunk_fixed(d.text, c["fixed_tokens"], c["fixed_overlap"])
    records = []
    for i, ch in enumerate(raw):
        path = [p for p in ch["section_path"] if p != d.title] or [d.title]
        section = path[-1]
        records.append(
            {
                "chunk_id": f"{d.doc_id}#{i:03d}",
                "doc_id": d.doc_id,
                "title": d.title,
                "section": section,
                "heading_path": " > ".join([d.title, *[p for p in path if p != d.title]]),
                "url": d.url + ("" if section == d.title else "#" + slug(section)),
                "doc_set": d.doc_set,
                "format": d.fmt,
                "updated_at": d.updated_at,
                "doc_hash": d.content_hash,
                "chunk_hash": content_hash(ch["text"]),
                "tokens": n_tokens(ch["text"]),
                "strategy": strategy,
                "text": ch["text"],
            }
        )
    return records


def run(cfg: dict, strategies: tuple[str, ...] = ("structure", "fixed")) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    docs = load_docs(manifest)

    seen: dict[str, str] = {}
    changed = []
    for d in docs:
        if d.content_hash in seen:
            d.duplicate_of = seen[d.content_hash]
        else:
            seen[d.content_hash] = d.doc_id
        pinned = manifest[d.doc_id].get("content_hash")
        if pinned and pinned != d.content_hash:
            changed.append(d.doc_id)
    near = find_near_duplicates(docs, cfg["dedup"]["near_duplicate_jaccard"])
    for a, b, j in near:
        next(d for d in docs if d.doc_id == a).near_duplicates.append((b, j))
        next(d for d in docs if d.doc_id == b).near_duplicates.append((a, j))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    kept = [d for d in docs if not d.duplicate_of]
    quality = {}
    for strategy in strategies:
        records = [r for d in kept for r in chunk_doc(d, cfg, strategy)]
        with (PROCESSED / f"chunks_{strategy}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        quality[strategy] = chunk_quality(records)
        quality[strategy]["by_format"] = {
            fmt: chunk_quality([r for r in records if r["format"] == fmt]) for fmt in ("html", "pdf", "docx")
        }
    (PROCESSED / "documents.jsonl").write_text(
        "".join(json.dumps({"doc_id": d.doc_id, "title": d.title, "format": d.fmt, "text": d.text}, ensure_ascii=False) + "\n" for d in docs),
        encoding="utf-8",
    )

    for d in docs:
        m = manifest[d.doc_id]
        m["content_hash"] = d.content_hash
        m["tokens"] = n_tokens(d.text)
        m["duplicate_of"] = d.duplicate_of
        m["near_duplicates"] = [{"doc_id": o, "jaccard": j} for o, j in d.near_duplicates]
    for strategy in strategies:
        counts = Counter(json.loads(l)["doc_id"] for l in (PROCESSED / f"chunks_{strategy}.jsonl").open())
        for d in docs:
            manifest[d.doc_id][f"chunks_{strategy}"] = counts.get(d.doc_id, 0)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    return {
        "pipeline_version": PIPELINE_VERSION,
        "documents": len(docs),
        "by_format": dict(Counter(d.fmt for d in docs)),
        "exact_duplicates": {d.doc_id: d.duplicate_of for d in docs if d.duplicate_of},
        "near_duplicates": [{"a": a, "b": b, "jaccard": j} for a, b, j in near],
        "changed_since_manifest": changed,
        "kept_documents": len(kept),
        "chunk_quality": quality,
    }
