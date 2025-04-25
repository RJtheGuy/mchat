import os
from typing import Generator, Tuple, List

def chunk_text(text: str, max_tokens: int = 200) -> List[str]:
    """
    Split text into smaller chunks with improved efficiency.
    Using a smaller max_tokens value for faster retrieval.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        # Simple word count estimation (more efficient than full tokenization)
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
    
    # For very long paragraphs, split them further
    final_chunks = []
    for chunk in chunks:
        if len(chunk.split()) > max_tokens:
            words = chunk.split()
            for i in range(0, len(words), max_tokens):
                final_chunks.append(' '.join(words[i:i+max_tokens]))
        else:
            final_chunks.append(chunk)
            
    return final_chunks

def load_all_files(base_dir: str) -> Generator[Tuple[str, str], None, None]:
    """Load all valid text files from a directory."""
    valid_extensions = {'.txt', '.md', '.markdown'}
    
    # Ensure the directory exists
    if not os.path.exists(base_dir):
        print(f"Warning: Directory {base_dir} does not exist.")
        return
        
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
                    print(f"Warning: Could not decode {filepath}. Skipping.")
                    continue