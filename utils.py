import os
from typing import Generator, Tuple

def chunk_text(text: str, max_tokens: int = 300) -> list:
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_length = len(para.split())
        if current_length + para_length <= max_tokens:
            current_chunk.append(para)
            current_length += para_length
        else:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_length = para_length

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    return chunks

def load_all_files(base_dir: str) -> Generator[Tuple[str, str], None, None]:
    valid_extensions = {'.txt', '.md', '.markdown'}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            yield filepath, content
                except UnicodeDecodeError:
                    continue
