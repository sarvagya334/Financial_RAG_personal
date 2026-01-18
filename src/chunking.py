# src/chunking.py
from __future__ import annotations
from typing import List


def is_table_line(line: str) -> bool:
    line = line.strip()
    # basic markdown table detector
    return line.startswith("|") and "|" in line[1:]


def split_markdown_blocks(md: str) -> List[str]:
    """
    Split markdown into blocks:
    - headings
    - paragraphs
    - tables (kept intact)
    - code blocks
    """
    lines = md.splitlines()
    blocks: List[str] = []

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # ---- code block ----
        if line.strip().startswith("```"):
            start = i
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                i += 1
            if i < n:
                i += 1  # include closing ```
            blocks.append("\n".join(lines[start:i]).strip())
            continue

        # ---- heading ----
        if line.strip().startswith("#"):
            blocks.append(line.strip())
            i += 1
            continue

        # ---- table ----
        if is_table_line(line):
            start = i
            i += 1
            while i < n and is_table_line(lines[i]):
                i += 1
            blocks.append("\n".join(lines[start:i]).strip())
            continue

        # ---- paragraph / normal text ----
        if line.strip() == "":
            i += 1
            continue

        start = i
        i += 1
        while i < n and lines[i].strip() != "" and not lines[i].strip().startswith("#") and not is_table_line(lines[i]) and not lines[i].strip().startswith("```"):
            i += 1
        blocks.append("\n".join(lines[start:i]).strip())

    return [b for b in blocks if b.strip()]


def split_big_table(table_block: str, max_chars: int) -> List[str]:
    """
    If a markdown table is too large, split by rows while keeping header.
    """
    lines = table_block.splitlines()
    if len(lines) <= 2:
        return [table_block]

    header = lines[0]
    sep = lines[1]
    rows = lines[2:]

    chunks: List[str] = []
    current = [header, sep]

    for row in rows:
        # if adding row exceeds limit -> flush current
        if len("\n".join(current + [row])) > max_chars and len(current) > 2:
            chunks.append("\n".join(current).strip())
            current = [header, sep, row]
        else:
            current.append(row)

    if len(current) > 2:
        chunks.append("\n".join(current).strip())

    return chunks


def chunk_markdown_table_safe(
    md_text: str,
    chunk_size: int = 2500,
    overlap_blocks: int = 1
) -> List[str]:
    """
    Chunk markdown while preserving tables.

    - chunk_size: max characters per chunk
    - overlap_blocks: repeats last N blocks in next chunk for continuity
    """
    blocks = split_markdown_blocks(md_text)
    chunks: List[str] = []

    current_blocks: List[str] = []

    for block in blocks:
        # if table is huge -> split table by rows
        if block.strip().startswith("|") and len(block) > chunk_size:
            table_parts = split_big_table(block, chunk_size)
            for part in table_parts:
                # flush current blocks first
                if current_blocks:
                    chunks.append("\n\n".join(current_blocks).strip())
                    current_blocks = []
                chunks.append(part.strip())
            continue

        # try adding block to current chunk
        candidate = "\n\n".join(current_blocks + [block]).strip()
        if len(candidate) <= chunk_size:
            current_blocks.append(block)
        else:
            # flush chunk
            if current_blocks:
                chunks.append("\n\n".join(current_blocks).strip())

            # overlap last blocks
            if overlap_blocks > 0 and chunks:
                prev_blocks = current_blocks[-overlap_blocks:] if len(current_blocks) >= overlap_blocks else current_blocks
                current_blocks = prev_blocks + [block]
            else:
                current_blocks = [block]

    if current_blocks:
        chunks.append("\n\n".join(current_blocks).strip())

    return chunks
