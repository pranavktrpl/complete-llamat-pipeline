#!/usr/bin/env python3
"""
Very small chunker.

Steps
1. Read a plain-text research paper whose sections have already been wrapped
   with <SEC> … </SEC> tags (see get_text_from_xmls.py edit above).
2. Split the file on those tags to obtain the individual sections.
3. Slice every section into non-overlapping word chunks
   of a user-specified maximum size.
4. Save **only** the list of chunk texts (no metadata) to
   checkpoints/chunks.json.
"""

import argparse
import json
import os
import re
from typing import List
from .split_sentences import split_sentences


def split_into_sections(text: str) -> List[str]:
    """
    Returns a list with the raw text of every <SEC> … </SEC> block
    (order is preserved).
    """
    pattern = re.compile(r"<SEC>\s*(.*?)\s*</SEC>", re.DOTALL)
    return [match.strip() for match in pattern.findall(text)]


def chunk_section(section_text: str, max_words: int) -> List[str]:
    """
    Break a single section into sentence-aware chunks of at most `max_words`
    words.  A chunk is a sequence of **complete sentences**; we never split
    in the middle of a sentence.
    """
    # Use the sci-spacy model by default (change if you only have std spaCy).
    # sentences = split_sentences(section_text, model="sci-spacy")
    sentences = split_sentences(section_text, model="std-spacy")

    chunks: List[str] = []
    current_chunk_sentences: List[str] = []
    current_word_count = 0

    for sent in sentences:
        sent_words = sent.split()

        # If adding this sentence would overflow, close the current chunk first
        if current_word_count + len(sent_words) > max_words and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences).strip())
            current_chunk_sentences = []
            current_word_count = 0

        current_chunk_sentences.append(sent)
        current_word_count += len(sent_words)

    # Add whatever is left
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences).strip())

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple <SEC>-based chunker.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the TXT file produced by get_text_from_xmls.py",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=200,
        help="Maximum number of words per chunk (default: 200)",
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints",
        help="Directory to put checkpoints/chunks.json (default: checkpoints)",
    )
    args = parser.parse_args()

    # 1. read the paper
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. split into sections
    sections = split_into_sections(text)

    # 3. chunk every section
    chunks: List[str] = []
    for sec in sections:
        chunks.extend(chunk_section(sec, args.chunk_size))

    # 4. write to checkpoints/chunks.json
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"✅  Wrote {len(chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main()
