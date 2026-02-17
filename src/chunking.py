import re
from typing import List, Tuple, Optional


# -----------------------------
# Token Estimation
# -----------------------------
def approx_tokens(text: str) -> int:
    """
    Approx token count:
    - 1 token ~ 4 chars
    - extra cost for markdown tables due to pipes '|'
    """
    base = max(1, len(text) // 4)
    base += text.count("|") // 6
    return base


# -----------------------------
# Table Detection Helpers
# -----------------------------
def _is_table_line(line: str) -> bool:
    line = line.strip()
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    """
    Detect markdown table separator:
    |------|------|
    |:-----|-----:|
    """
    line = line.strip()
    if not _is_table_line(line):
        return False

    inner = line.strip("|").replace(" ", "")
    parts = inner.split("|")
    return all(re.fullmatch(r":?-{3,}:?", p) for p in parts)


# -----------------------------
# Page Splitting
# -----------------------------
def split_by_pages(md: str) -> List[Tuple[int, str]]:
    """
    Split markdown into pages.

    Supports:
    - form feed: \f
    - '--- Page 3 ---'
    - 'Page 3'
    - '# Page 3' / '## Page 3'
    """
    md = md.replace("\r\n", "\n")

    # Best: form feed
    if "\f" in md:
        chunks = [c.strip() for c in md.split("\f")]
        return [(i + 1, c) for i, c in enumerate(chunks) if c]

    # Page marker regex
    page_pat = re.compile(r"(?im)^\s*(?:---\s*)?(?:#+\s*)?page\s*(\d+)\s*(?:---\s*)?$")

    pages: List[Tuple[int, str]] = []
    buf: List[str] = []
    current_page = 1

    def flush():
        nonlocal buf, current_page
        content = "\n".join(buf).strip()
        if content:
            pages.append((current_page, content))
        buf = []

    for line in md.splitlines():
        m = page_pat.match(line.strip())
        if m:
            flush()
            current_page = int(m.group(1))
        else:
            buf.append(line)

    flush()
    return pages


# -----------------------------
# Heading Splitting
# -----------------------------
def split_by_headings(md: str) -> List[str]:
    """
    Split markdown into sections based on headings (# ...).
    Keeps heading attached to its content.
    """
    pattern = re.compile(r"(^#{1,6}\s.*$)", re.MULTILINE)
    parts = pattern.split(md)

    if len(parts) == 1:
        return [md.strip()] if md.strip() else []

    sections: List[str] = []
    buf = ""

    for part in parts:
        if not part.strip():
            continue

        if part.startswith("#"):  # heading
            if buf.strip():
                sections.append(buf.strip())
            buf = part.strip() + "\n"
        else:
            buf += part

    if buf.strip():
        sections.append(buf.strip())

    return sections


def _extract_section_title(section_md: str) -> Optional[str]:
    """
    Extract first heading from a section.
    """
    for line in section_md.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None


# -----------------------------
# Block Splitting: ("text" | "table")
# -----------------------------
def split_blocks(md: str) -> List[Tuple[str, str]]:
    """
    Splits section markdown into blocks:
    - ("table", table markdown)
    - ("text", paragraph markdown)
    """
    lines = md.splitlines()
    blocks: List[Tuple[str, str]] = []
    i = 0

    while i < len(lines):
        # Detect table start (header line + separator line)
        if i + 1 < len(lines) and _is_table_line(lines[i]) and _is_table_separator(lines[i + 1]):
            table_lines = [lines[i], lines[i + 1]]
            i += 2

            while i < len(lines) and _is_table_line(lines[i]):
                table_lines.append(lines[i])
                i += 1

            blocks.append(("table", "\n".join(table_lines).strip()))
            continue

        # Otherwise it's text
        buf = [lines[i]]
        i += 1

        while i < len(lines):
            if i + 1 < len(lines) and _is_table_line(lines[i]) and _is_table_separator(lines[i + 1]):
                break
            buf.append(lines[i])
            i += 1

        content = "\n".join(buf).strip()
        if content:
            blocks.append(("text", content))

    return blocks

def chunk_table_rows(
    table_md: str,
    *,
    max_table_tokens: int = 350,
    min_rows: int = 12,
    table_title: Optional[str] = None
) -> List[str]:
    """
    Chunk large markdown tables row-wise.
    Always repeats header+separator.
    Adds [TABLE: title] label.
    """
    lines = table_md.splitlines()
    if len(lines) < 3:
        return [table_md.strip()]

    header, sep = lines[0], lines[1]
    rows = lines[2:]

    out: List[str] = []
    current: List[str] = []

    def flush():
        nonlocal current
        if not current:
            return
        prefix = f"[TABLE: {table_title}]\n" if table_title else ""
        chunk = prefix + "\n".join([header, sep] + current)
        out.append(chunk.strip())
        current = []

    for row in rows:
        prefix = f"[TABLE: {table_title}]\n" if table_title else ""
        candidate = prefix + "\n".join([header, sep] + current + [row])

        if approx_tokens(candidate) > max_table_tokens and len(current) >= min_rows:
            flush()
            current = [row]
        else:
            current.append(row)

    flush()
    return out


# -----------------------------
# Chunking: Text
# -----------------------------
def chunk_text(
    text: str,
    *,
    max_text_tokens: int = 520,
    overlap_tokens: int = 80
) -> List[str]:
    """
    Chunk text by paragraphs.
    If paragraph too big -> sentence chunk.
    Adds overlap only to text chunks.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for p in paras:
        candidate = (buf + "\n\n" + p).strip() if buf else p

        if approx_tokens(candidate) <= max_text_tokens:
            buf = candidate
            continue

        if buf:
            flush()

        # if paragraph itself fits
        if approx_tokens(p) <= max_text_tokens:
            buf = p
            continue

        # paragraph too large -> sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", p)
        tmp = ""

        for s in sentences:
            cand2 = (tmp + " " + s).strip() if tmp else s
            if approx_tokens(cand2) <= max_text_tokens:
                tmp = cand2
            else:
                if tmp.strip():
                    chunks.append(tmp.strip())
                tmp = s

        if tmp.strip():
            chunks.append(tmp.strip())

    flush()

    # overlap
    if overlap_tokens > 0 and len(chunks) > 1:
        out = [chunks[0]]
        tail_chars = overlap_tokens * 4

        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            tail = prev[-min(len(prev), tail_chars):]
            out.append((tail + "\n\n" + chunks[i]).strip())

        chunks = out

    return chunks


# -----------------------------
# Final: Page-Level Dynamic Chunker
# -----------------------------
def chunk_markdown_page_level(
    md: str,
    *,
    max_text_tokens: int = 520,
    max_table_tokens: int = 350,
    overlap_text_tokens: int = 80
) -> List[str]:
    """
    Page-level markdown chunking:
    - Split pages
    - Within each page: headings -> blocks -> chunk
    - Prefix each chunk with [PAGE: N]
    """
    final: List[str] = []
    pages = split_by_pages(md)

    for page_no, page_md in pages:
        page_prefix = f"[PAGE: {page_no}]\n"

        sections = split_by_headings(page_md)

        for sec in sections:
            title = _extract_section_title(sec)
            blocks = split_blocks(sec)

            for kind, content in blocks:
                if kind == "table":
                    if approx_tokens(content) <= max_table_tokens:
                        table_prefix = f"[TABLE: {title}]\n" if title else ""
                        final.append((page_prefix + table_prefix + content).strip())
                    else:
                        for table_chunk in chunk_table_rows(
                            content,
                            max_table_tokens=max_table_tokens,
                            min_rows=12,
                            table_title=title
                        ):
                            final.append((page_prefix + table_chunk).strip())

                else:
                    for text_chunk in chunk_text(
                        content,
                        max_text_tokens=max_text_tokens,
                        overlap_tokens=overlap_text_tokens
                    ):
                        final.append((page_prefix + text_chunk).strip())

    return [c.strip() for c in final if c.strip()]
