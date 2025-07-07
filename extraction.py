
#!/usr/bin/env python3
"""
Complete LLaMAT Pipeline for Materials Composition Extraction

This pipeline processes research papers to extract material compositions.
Takes a PII directory as input and performs the full extraction workflow.
"""

import os
import json
import argparse
from typing import Dict, Any, List
from pathlib import Path

from utils.rawPaperProcessing.get_text_from_xmls import get_contents
from extraction.chunking import split_into_sections, chunk_section
from extraction.split_sentences import split_sentences
from extraction.extraction_util import (
    load_system_prompt, 
    load_user_prompt, 
    make_user_prompt,
    extract_compositions_from_chunks
)

from extraction.process_outputs import extract_compositions_from_raw_output

from utils.call_llamat import (
    llamat_text_completion,
    Message,
    LlamaTCompletionRequest,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for the extraction pipeline."""
    
    # Chunking parameters
    CHUNK_SIZE = 500  # Maximum words per chunk
    SENTENCE_MODEL = "sci-spacy"  # Model for sentence splitting
    
    # LLM parameters
    MAX_TOKENS = 128  # Maximum tokens for LLM generation
    TEMPERATURE = 0.0  # Temperature for LLM (0.0 = deterministic)
    
    # Metadata parameters
    SAVE_METADATA = False  # Whether to save metadata for each chunk
    VERBOSE = True  # Whether to print verbose output
    
    # File paths
    SYSTEM_PROMPT_PATH = "prompts/extraction_system_prompt_composition.txt"
    USER_PROMPT_PATH = "prompts/extraction_user_prompt_composition.txt"
    
    # Directory structure
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    CHECKPOINTS_DIR = "checkpoints"
    
    # File names
    RAW_XML_FILE = "raw_xml.xml"
    EXTRACTED_TEXT_FILE = "research-paper-text.txt"
    CHUNKS_FILE = "chunks.json"
    RAW_OUTPUTS_FILE = "compositions_model_raw.json"
    METADATA_FILE = "compositions_metadata.json"
    FINAL_COMPOSITIONS_FILE = "compositions.json"

# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================

def setup_directories(pii_id: str, config: Config) -> Dict[str, str]:
    """Set up the directory structure for processing."""
    
    paths = {
        'input_dir': os.path.join(config.INPUT_DIR, pii_id),
        'output_dir': os.path.join(config.OUTPUT_DIR, pii_id),
        'checkpoint_dir': os.path.join(config.CHECKPOINTS_DIR, pii_id),
    }
    
    # Create output and checkpoint directories
    os.makedirs(paths['output_dir'], exist_ok=True)
    os.makedirs(paths['checkpoint_dir'], exist_ok=True)
    
    # Add file paths
    paths.update({
        'xml_file': os.path.join(paths['input_dir'], config.RAW_XML_FILE),
        'extracted_text_file': os.path.join(paths['input_dir'], config.EXTRACTED_TEXT_FILE),
        'chunks_file': os.path.join(paths['checkpoint_dir'], config.CHUNKS_FILE),
        'raw_outputs_file': os.path.join(paths['checkpoint_dir'], config.RAW_OUTPUTS_FILE),
        'metadata_file': os.path.join(paths['checkpoint_dir'], config.METADATA_FILE),
        'final_compositions_file': os.path.join(paths['output_dir'], config.FINAL_COMPOSITIONS_FILE),
    })
    
    return paths

def extract_text_from_xml(xml_path: str, output_path: str, config: Config) -> None:
    """Extract and save text from XML file."""
    if config.VERBOSE: print(f"📄 Extracting text from XML: {xml_path}")
    
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    # Extract text with section tags
    sections_to_exclude = ['references', 'bibliography', 'ref', 'bibliographies', 'reference']
    extracted_text = get_contents(xml_path, exclude_sections=sections_to_exclude)
    
    # Save extracted text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    
    if config.VERBOSE: print(f"✅ Extracted text saved to: {output_path}")

def create_chunks(text_file: str, chunks_file: str, config: Config) -> List[str]:
    """Create chunks from the extracted text."""
    if config.VERBOSE: print(f"🔪 Creating chunks from: {text_file}")

    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()    
    sections = split_into_sections(text)
    if config.VERBOSE: print(f"Found {len(sections)} sections")
        
    chunks = []
    for i, section in enumerate(sections):
        section_chunks = chunk_section(section, config.CHUNK_SIZE)
        chunks.extend(section_chunks)
        if config.VERBOSE: print(f"  Section {i+1}: {len(section_chunks)} chunks")    
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    if config.VERBOSE: print(f"✅ Created {len(chunks)} chunks, saved to: {chunks_file}")
    return chunks

# def extract_compositions_from_chunks_pipeline(chunks: List[str], raw_outputs_file: str, metadata_file: str, config: Config) -> List[Dict[str, Any]]:
#     """Extract compositions from chunks using LLM - Pipeline wrapper."""
    
#     return extract_compositions_from_chunks(
#         chunks=chunks,
#         raw_outputs_file=raw_outputs_file,
#         system_prompt_path=config.SYSTEM_PROMPT_PATH,
#         user_prompt_path=config.USER_PROMPT_PATH,
#         max_tokens=config.MAX_TOKENS,
#         temperature=config.TEMPERATURE,
#         save_metadata=config.SAVE_METADATA,
#         metadata_file=metadata_file if config.SAVE_METADATA else None
#     )

def process_final_outputs(raw_outputs: List[Dict[str, Any]], final_file: str, config: Config) -> List[str]:
    """Process raw outputs to extract final compositions."""
    if config.VERBOSE: print(f"🔍 Processing {len(raw_outputs)} raw outputs")
    
    all_compositions = []
    
    for result in raw_outputs:
        if "raw_output" in result and result["raw_output"]:
            compositions = extract_compositions_from_raw_output(result["raw_output"])
            all_compositions.extend(compositions)
    
    # Remove duplicates while preserving order
    unique_compositions = []
    seen = set()
    for comp in all_compositions:
        if comp not in seen:
            unique_compositions.append(comp)
            seen.add(comp)
    
    # Save final compositions
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(unique_compositions, f, indent=2, ensure_ascii=False)
    
    if config.VERBOSE: print(f"✅ Found {len(unique_compositions)} unique compositions")
    if config.VERBOSE: print(f"✅ Final compositions saved to: {final_file}")
    
    return unique_compositions

def run_pipeline(pii_id: str, config: Config = None) -> Dict[str, Any]:
    """Run the complete extraction pipeline."""
    if config is None:
        config = Config()
    
    print(f"🚀 Starting pipeline for PII: {pii_id}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Setup directories
    paths = setup_directories(pii_id, config)
    
    # Step 1: Extract text from XML
    extract_text_from_xml(paths['xml_file'], paths['extracted_text_file'], config)
    
    # Step 2: Create chunks
    chunks = create_chunks(paths['extracted_text_file'], paths['chunks_file'], config)
    
    # Step 3: Extract compositions using LLM
    raw_outputs = extract_compositions_from_chunks(
        chunks=chunks,
        raw_outputs_file=paths['raw_outputs_file'],
        system_prompt_path=config.SYSTEM_PROMPT_PATH,
        user_prompt_path=config.USER_PROMPT_PATH,
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        save_metadata=config.SAVE_METADATA,
        metadata_file=paths['metadata_file'] if config.SAVE_METADATA else None
    )
    
    # Step 4: Process final outputs
    final_compositions = process_final_outputs(raw_outputs, paths['final_compositions_file'], config)
    
    # Summary
    summary = {
        'pii_id': pii_id,
        'num_chunks': len(chunks),
        'num_llm_calls': len(raw_outputs),
        'num_compositions': len(final_compositions),
        'files_created': [
            paths['extracted_text_file'],
            paths['chunks_file'],
            paths['raw_outputs_file'],
            paths['final_compositions_file']
        ]
    }
    
    if config.VERBOSE: print(f"🎉 Pipeline completed successfully!")
    if config.VERBOSE: print(f"📊 Summary: {summary['num_compositions']} compositions extracted from {summary['num_chunks']} chunks")
    
    return summary

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Complete LLaMAT Pipeline for Materials Composition Extraction")
    parser.add_argument(
        "--pii",
        required=True,
        help="PII directory name (e.g., 'S0167273808006176')"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Maximum words per chunk (default: 200)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum tokens for LLM generation (default: 128)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM generation (default: 0.0)"
    )
    parser.add_argument(
        "--save_metadata",
        action="store_true",
        help="Save metadata for each chunk (default: False)"
    )    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output (default: True)"
    )

    
    args = parser.parse_args()
    
    # Create custom config
    config = Config()
    config.CHUNK_SIZE = args.chunk_size
    config.MAX_TOKENS = args.max_tokens
    config.TEMPERATURE = args.temperature
    config.SAVE_METADATA = args.save_metadata
    config.VERBOSE = args.verbose

    try:
        # Run the pipeline
        summary = run_pipeline(args.pii, config)
        
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        for key, value in summary.items():
            if key == 'files_created':
                print(f"{key}: {len(value)} files")
                for file in value:
                    print(f"  - {file}")
            else:
                print(f"{key}: {value}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 