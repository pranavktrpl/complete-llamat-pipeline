#!/usr/bin/env python3
"""
Run composition-extraction on every chunk produced by chunking.py.

Required files
--------------
prompts/extraction_prompt.txt   – the static extraction prompt
checkpoints/chunks.json         – list[str] produced by chunking.py

Output
------
checkpoints/composition_results.json
    [
      {"chunk_idx": 0, "input": "<chunk text>", "raw_output": "<LLM reply>"},
      ...
    ]
"""

import json
import os
from typing import List, Dict, Any

from utils.call_llamat import (
    llamat_text_completion,
    Message,
    LlamaTCompletionRequest,
)

SYSTEM_PROMPT_PATH   = "prompts/extraction/system_prompt_composition.txt"
USER_PROMPT_PATH   = "prompts/extraction/user_prompt_composition.txt"
CHUNKS_PATH   = "checkpoints/chunks.json"
OUT_PATH      = "checkpoints/composition_outputs.json"
MAX_TOKENS    = 128          # generation length
TEMPERATURE   = 0.0          # deterministic output
SAVE_METADATA = False           # set to True to dump per-chunk metadata

METADATA_OUT = "checkpoints/composition_metadata.json"


def load_system_prompt(SYSTEM_PROMPT_PATH) -> str:
    """
    Read the full prompt from disk **and remove** the trailing
    """
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        raw = f.read().rstrip()
    return raw

def load_user_prompt(USER_PROMPT_PATH) -> str:
    """
    Read the full prompt from disk **and remove** the trailing
    'Now extract compositions …' part so that it lives in the user message.
    """
    with open(USER_PROMPT_PATH, "r", encoding="utf-8") as f:
        raw = f.read().rstrip()
    return raw

def make_user_prompt(passage: str, user_prompt_path: str = "prompts/extraction_user_prompt_composition.txt") -> str:
    """
    Build the user-role content:

    Now extract compositions from this PASSAGE:

    ```<passage>```
    """
    base_prompt = load_user_prompt(user_prompt_path)
    prompt = base_prompt.replace("{{passage}}", passage)
    return prompt


def load_chunks(CHUNKS_PATH) -> List[str]:
    """Read chunks.json (created by chunking.py)."""
    if not os.path.isfile(CHUNKS_PATH):
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_compositions_from_chunks(chunks: List[str], raw_outputs_file: str, system_prompt_path: str, 
                                   user_prompt_path: str, max_tokens: int = 128, temperature: float = 0.0,
                                   save_metadata: bool = False, metadata_file: str = None) -> List[Dict[str, Any]]:
    """Extract compositions from chunks using LLM."""
    print(f"🤖 Extracting compositions from {len(chunks)} chunks")
    
    # Load prompts
    system_prompt = load_system_prompt(system_prompt_path)
    
    results = []
    all_meta = []  # only used when save_metadata is True
    
    for idx, chunk_text in enumerate(chunks):
        print(f"  Processing chunk {idx+1}/{len(chunks)}")
        
        # Create user prompt
        user_prompt = make_user_prompt(chunk_text, user_prompt_path)
        
        try:
            # Create messages
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]
            
            # Create request
            request = LlamaTCompletionRequest(
                messages=messages,
                temperature=temperature,
                max_new_tokens=max_tokens,
            )
            
            # Call LLM
            reply, metadata = llamat_text_completion(request)
            
            results.append({
                "chunk_idx": idx,
                "input": chunk_text,
                "raw_output": reply
                # "metadata": metadata if save_metadata else None
            })
            
            if save_metadata:
                meta_record = {"chunk_idx": idx, **metadata}
                all_meta.append(meta_record)
                
        except Exception as e:
            print(f"    ❌ Error processing chunk {idx}: {e}")
            results.append({
                "chunk_idx": idx,
                "input": chunk_text,
                "raw_output": "",
                "error": str(e)
            })
    
    # Save raw outputs
    os.makedirs(os.path.dirname(raw_outputs_file), exist_ok=True)
    with open(raw_outputs_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Optionally save metadata file
    if save_metadata and metadata_file and all_meta:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_meta, f, indent=2, ensure_ascii=False)
        print(f"✅ Metadata saved to: {metadata_file}")
    
    print(f"✅ Raw outputs saved to: {raw_outputs_file}")
    return results


def main() -> None:
    """Legacy main function for backward compatibility."""
    chunks = load_chunks(CHUNKS_PATH)
    extract_compositions_from_chunks(
        chunks=chunks,
        raw_outputs_file=OUT_PATH,
        system_prompt_path=SYSTEM_PROMPT_PATH,
        user_prompt_path=USER_PROMPT_PATH,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        save_metadata=SAVE_METADATA,
        metadata_file=METADATA_OUT if SAVE_METADATA else None
    )


if __name__ == "__main__":
    main()
