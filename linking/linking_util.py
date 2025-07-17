#!/usr/bin/env python3
"""
Run composition-linking on extracted compositions and paper text.

Required files
--------------
prompts/linking/system_prompt.txt   – the static linking system prompt
prompts/linking/user_prompt.txt     – the linking user prompt template
checkpoints/compositions.json       – extracted compositions from extraction step
input/{paper_id}/research-paper-text.txt  – full paper text

Output
------
checkpoints/linking_results_raw.json
    [
      {"query_idx": 0, "query": "<property query>", "raw_output": "<LLM reply>"},
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

SYSTEM_PROMPT_PATH = "prompts/linking/system_prompt.txt"
USER_PROMPT_PATH = "prompts/linking/user_prompt.txt"
COMPOSITIONS_PATH = "output/S0167273808006176/compositions.json"
CHUNKS_PATH   = "checkpoints/chunks.json"
QUERY_PATH    = "checkpoints/queries.json"
OUT_PATH = "checkpoints/linking_results_raw.json"
MAX_TOKENS = 64           # shorter output for linking
TEMPERATURE = 0.0         # deterministic output
SAVE_METADATA = False     # set to True to dump per-query metadata

METADATA_OUT = "checkpoints/linking_metadata.json"


def load_system_prompt(system_prompt_path: str) -> str:
    """Read the linking system prompt from disk."""
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        return f.read().rstrip()


def load_user_prompt(user_prompt_path: str) -> str:
    """Read the linking user prompt template from disk."""
    with open(user_prompt_path, "r", encoding="utf-8") as f:
        return f.read().rstrip()


def make_user_prompt(chunk_context: str, query: str, compositions: List[str], user_prompt_path: str) -> str:
    """
    Build the user-role content by substituting placeholders in the user prompt template.
    """
    template = load_user_prompt(user_prompt_path)
    
    # Format compositions as a numbered list
    compositions_text = "\n".join(f"{i+1}. {comp}" for i, comp in enumerate(compositions))
    
    prompt = template.replace("{{PAPER_TEXT_CONTEXT}}", chunk_context)
    prompt = prompt.replace("{{QUERY}}", query)
    prompt = prompt.replace("{{CANDIDATE_COMPOSITIONS}}", compositions_text)
    
    return prompt


# def load_paper_text(paper_id: str) -> str:
#     """Load the full paper text for linking context."""
#     paper_path = f"input/{paper_id}/research-paper-text.txt"
#     if not os.path.isfile(paper_path):
#         raise FileNotFoundError(f"Paper text not found: {paper_path}")
    
#     with open(paper_path, "r", encoding="utf-8") as f:
#         return f.read()

def load_chunks(CHUNKS_PATH) -> List[str]:
    """Read chunks.json (created by chunking.py)."""
    if not os.path.isfile(CHUNKS_PATH):
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_compositions(compositions_path: str) -> List[str]:
    """Load extracted compositions from the extraction step."""
    if not os.path.isfile(compositions_path):
        raise FileNotFoundError(f"Compositions file not found: {compositions_path}")
    
    with open(compositions_path, "r", encoding="utf-8") as f:
        return json.load(f)


def link_compositions_to_properties(chunks: List[str], property_queries: List[str], compositions: List[str], 
                                  system_prompt_path: str, user_prompt_path: str, output_file: str,
                                  max_tokens: int = 64, temperature: float = 0.0,
                                  save_metadata: bool = False, metadata_file: str = None) -> List[Dict[str, Any]]:
    """Link compositions to material properties using LLM."""
    print(f"🔗 Linking {len(compositions)} compositions to {len(property_queries)} properties using {len(chunks)} context chunks")
    
    # Load prompts
    system_prompt = load_system_prompt(system_prompt_path)
    
    results = []
    all_meta = []  # only used when save_metadata is True
    
    for idx_query, query in enumerate(property_queries):
        for idx_chunks, chunk_text in enumerate(chunks):

            print(f"  Processing query {idx_query+1}/{len(property_queries)}: {query[:50]}... with chunk {idx_chunks+1}/{len(chunks)}: {chunk_text[:50]}...")
            
            # Create user prompt
            user_prompt = make_user_prompt(chunk_text, query, compositions, user_prompt_path)
            
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
                    "query_idx": idx_query,
                    "chunk_idx": idx_chunks,
                    "query": query,
                    "raw_output": reply.strip()
                })
                
                if save_metadata:
                    meta_record = {"query_idx": idx_query, "chunk_idx": idx_chunks, **metadata}
                    all_meta.append(meta_record)
                    
            except Exception as e:
                print(f"    ❌ Error processing query {idx_query}: {e}")
                results.append({
                    "query_idx": idx_query,
                    "chunk_idx": idx_chunks,
                    "query": query,
                    "raw_output": "",
                    "error": str(e)
                })
    
    # Save linking results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Optionally save metadata file
    if save_metadata and metadata_file and all_meta:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_meta, f, indent=2, ensure_ascii=False)
        print(f"✅ Metadata saved to: {metadata_file}")
    
    print(f"✅ Linking results saved to: {output_file}")
    return results


def main(paper_id: str = "S0167273808006176", property_queries: List[str] = None) -> None:
    """Main function for linking compositions to properties."""
    if property_queries is None:
        # Default example queries - in practice these would come from somewhere else
        property_queries = [
            "Activation energy : 1.64 eV",
            "Band gap : 2.3 eV",
            "Thermal conductivity : 10 W/mK"
        ]
    
    # Load inputs
    chunks = load_chunks(CHUNKS_PATH)
    compositions = load_compositions(COMPOSITIONS_PATH)
    
    # Run linking
    link_compositions_to_properties(
        chunks=chunks,
        property_queries=property_queries,
        compositions=compositions,
        system_prompt_path=SYSTEM_PROMPT_PATH,
        user_prompt_path=USER_PROMPT_PATH,
        output_file=OUT_PATH,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        save_metadata=SAVE_METADATA,
        metadata_file=METADATA_OUT if SAVE_METADATA else None
    )


if __name__ == "__main__":
    main()
