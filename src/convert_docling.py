import os
from typing import List
from docling.document_converter import DocumentConverter

SUPPORTED_EXTS = {".pdf", ".xlsx", ".xls", ".csv", ".docx", ".pptx"}

def docling_convert_to_markdown(source_file: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)

    converter = DocumentConverter()
    result = converter.convert(source_file)

    md_text = result.document.export_to_markdown()

    base = os.path.basename(source_file)
    name, _ = os.path.splitext(base)
    out_path = os.path.join(out_dir, f"{name}.md")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    return out_path


def convert_all_raw_to_markdown(raw_dir: str, processed_dir: str) -> List[str]:
    """
    Converts supported raw files -> markdown.
    If corresponding .md already exists, skip conversion.
    Returns list of markdown paths (both existing + newly created).
    """
    os.makedirs(processed_dir, exist_ok=True)

    md_paths: List[str] = []

    for filename in os.listdir(raw_dir):
        file_path = os.path.join(raw_dir, filename)

        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTS:
            print(f"[SKIP] Unsupported: {filename}")
            continue

        name, _ = os.path.splitext(filename)
        expected_md = os.path.join(processed_dir, f"{name}.md")

        if os.path.exists(expected_md):
            print(f"[SKIP] Already converted: {filename} → {name}.md")
            md_paths.append(expected_md)
            continue
        try:
            md_path = docling_convert_to_markdown(file_path, processed_dir)
            md_paths.append(md_path)
            print(f"[OK] Converted: {filename} → {os.path.basename(md_path)}")
        except Exception as e:
            print(f"[FAIL] {filename}: {e}")

    return md_paths
