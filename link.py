#!/usr/bin/env python3
"""
Complete LLaMAT Linking Pipeline for Composition-Property Linking

This pipeline processes extracted compositions and links them to material properties.
Takes a PII directory as input and performs the full linking workflow.
"""

import os
import json
import argparse
from typing import Dict, Any, List
from pathlib import Path

from linking.process_matskraft_tables import get_matskraft_property_queries
from linking.linking_util import (
    load_chunks,
    load_compositions,
    link_compositions_to_properties
)
from linking.process_outputs import process_linking_results_enhanced

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for the linking pipeline."""
    
    # LLM parameters
    # MAX_TOKENS = 64  # Maximum tokens for LLM generation (shorter for linking)
    MAX_TOKENS = 256  # Increase for structures-reasoning backed output
    TEMPERATURE = 0.0  # Temperature for LLM (0.0 = deterministic)
    
    # Metadata parameters
    SAVE_METADATA = True  # Whether to save metadata for each query
    VERBOSE = True  # Whether to print verbose output
    
    # File paths
    # SYSTEM_PROMPT_PATH = "prompts/linking/system_prompt.txt"
    # USER_PROMPT_PATH = "prompts/linking/user_prompt.txt"
    # File paths for reasoning-backed output
    SYSTEM_PROMPT_PATH = "prompts/linking/system_reasoned_prompt.txt"
    USER_PROMPT_PATH = "prompts/linking/user_reasoning_prompt.txt"
    
    # Directory structure
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    CHECKPOINTS_DIR = "checkpoints"
    
    # File names
    CHUNKS_FILE = "chunks.json"
    COMPOSITIONS_FILE = "compositions.json"
    QUERIES_FILE = "queries.json"
    RAW_LINKING_OUTPUTS_FILE = "linking_results_raw.json"
    PROCESSED_LINKING_OUTPUTS_FILE = "linking_results.json"
    STRUCTURED_LINKING_OUTPUTS_FILE = "linking_results_structured.json"
    METADATA_FILE = "linking_metadata.json"

# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================

def setup_directories(pii_id: str, config: Config) -> Dict[str, str]:
    """Set up the directory structure for linking processing."""
    
    paths = {
        'input_dir': os.path.join(config.INPUT_DIR, pii_id),
        'output_dir': os.path.join(config.OUTPUT_DIR, pii_id),
        'checkpoint_dir': os.path.join(config.CHECKPOINTS_DIR, pii_id),
    }
    
    # Create directories if they don't exist
    os.makedirs(paths['output_dir'], exist_ok=True)
    os.makedirs(paths['checkpoint_dir'], exist_ok=True)
    
    # Add file paths
    paths.update({
        'chunks_file': os.path.join(paths['checkpoint_dir'], config.CHUNKS_FILE),
        'compositions_file': os.path.join(paths['output_dir'], config.COMPOSITIONS_FILE),
        'queries_file': os.path.join(paths['checkpoint_dir'], config.QUERIES_FILE),
        'raw_linking_outputs_file': os.path.join(paths['checkpoint_dir'], config.RAW_LINKING_OUTPUTS_FILE),
        'processed_linking_outputs_file': os.path.join(paths['output_dir'], config.PROCESSED_LINKING_OUTPUTS_FILE),
        'metadata_file': os.path.join(paths['checkpoint_dir'], config.METADATA_FILE),
    })
    
    return paths

def extract_and_save_property_queries(pii_id: str, queries_file: str, config: Config) -> List[str]:
    """Extract property queries from Matskraft table and save them."""
    if config.VERBOSE: print(f"📋 Extracting property queries for PII: {pii_id}")
    
    try:
        # Get property queries from Matskraft table
        property_queries = get_matskraft_property_queries(pii_id)
        
        if not property_queries:
            print(f"⚠️  No property queries found for PII: {pii_id}")
            return []
        
        # Save queries to JSON file
        with open(queries_file, 'w', encoding='utf-8') as f:
            json.dump(property_queries, f, indent=2, ensure_ascii=False)
        
        if config.VERBOSE: 
            print(f"✅ Found {len(property_queries)} property queries")
            print(f"✅ Queries saved to: {queries_file}")
            if config.VERBOSE and len(property_queries) <= 10:
                print("   Sample queries:")
                for i, query in enumerate(property_queries[:5], 1):
                    print(f"     {i}. {query}")
                if len(property_queries) > 5:
                    print(f"     ... and {len(property_queries) - 5} more")
        
        return property_queries
        
    except Exception as e:
        print(f"❌ Error extracting property queries: {e}")
        raise

def load_required_files(paths: Dict[str, str], config: Config) -> tuple[List[str], List[str], List[str]]:
    """Load chunks, compositions, and queries from files."""
    if config.VERBOSE: print(f"📂 Loading required files")
    
    # Load chunks
    if not os.path.exists(paths['chunks_file']):
        raise FileNotFoundError(f"Chunks file not found: {paths['chunks_file']}")
    chunks = load_chunks(paths['chunks_file'])
    if config.VERBOSE: print(f"   Loaded {len(chunks)} chunks")
    
    # Load compositions
    if not os.path.exists(paths['compositions_file']):
        raise FileNotFoundError(f"Compositions file not found: {paths['compositions_file']}")
    compositions = load_compositions(paths['compositions_file'])
    if config.VERBOSE: print(f"   Loaded {len(compositions)} compositions")
    
    # Load queries
    if not os.path.exists(paths['queries_file']):
        raise FileNotFoundError(f"Queries file not found: {paths['queries_file']}")
    with open(paths['queries_file'], 'r', encoding='utf-8') as f:
        queries = json.load(f)
    if config.VERBOSE: print(f"   Loaded {len(queries)} property queries")
    
    return chunks, compositions, queries

def run_linking_pipeline(chunks: List[str], compositions: List[str], queries: List[str], 
                        paths: Dict[str, str], config: Config) -> List[Dict[str, Any]]:
    """Run the composition-property linking using LLM."""
    if config.VERBOSE: print(f"🔗 Starting linking process")
    
    # Run linking
    linking_results = link_compositions_to_properties(
        chunks=chunks,
        property_queries=queries,
        compositions=compositions,
        system_prompt_path=config.SYSTEM_PROMPT_PATH,
        user_prompt_path=config.USER_PROMPT_PATH,
        output_file=paths['raw_linking_outputs_file'],
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        save_metadata=config.SAVE_METADATA,
        metadata_file=paths['metadata_file'] if config.SAVE_METADATA else None
    )
    
    if config.VERBOSE: print(f"✅ Linking completed: {len(linking_results)} results generated")
    
    return linking_results

def run_pipeline(pii_id: str, config: Config = None) -> Dict[str, Any]:
    """Run the complete linking pipeline."""
    if config is None:
        config = Config()
    
    print(f"🚀 Starting linking pipeline for PII: {pii_id}")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Setup directories
    paths = setup_directories(pii_id, config)
    
    # Step 1: Extract and save property queries
    queries = extract_and_save_property_queries(pii_id, paths['queries_file'], config)
    
    if not queries:
        print("⚠️  No property queries found. Stopping pipeline.")
        return {
            'pii_id': pii_id,
            'num_queries': 0,
            'num_compositions': 0,
            'num_chunks': 0,
            'num_linking_results': 0,
            'num_processed_results': 0,
            'status': 'stopped_no_queries'
        }
    
    # Step 2: Load required files
    chunks, compositions, queries_loaded = load_required_files(paths, config)
    
    # Step 3: Run linking pipeline
    linking_results = run_linking_pipeline(chunks, compositions, queries_loaded, paths, config)
    
    # Step 4: Process raw linking results with enhanced function
    processed_results = process_linking_results_enhanced(
        input_file=paths['raw_linking_outputs_file'], 
        output_file=paths['processed_linking_outputs_file'],
        structured_output_file=os.path.join(paths['output_dir'], config.STRUCTURED_LINKING_OUTPUTS_FILE)
    )
    
    # Summary
    summary = {
        'pii_id': pii_id,
        'num_queries': len(queries_loaded),
        'num_compositions': len(compositions),
        'num_chunks': len(chunks),
        'num_linking_results': len(linking_results),
        'num_processed_results': len(processed_results),
        'files_created': [
            paths['queries_file'],
            paths['raw_linking_outputs_file'],
            paths['processed_linking_outputs_file'],
            paths['processed_linking_outputs_file'].replace('.json', '_structured.json'),
        ],
        'status': 'completed'
    }
    
    if config.SAVE_METADATA:
        summary['files_created'].append(paths['metadata_file'])
    
    if config.VERBOSE: print(f"🎉 Linking pipeline completed successfully!")
    if config.VERBOSE: print(f"📊 Summary: {summary['num_linking_results']} raw results → {summary['num_processed_results']} processed query-composition pairs from {summary['num_queries']} queries")
    
    return summary

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Complete LLaMAT Linking Pipeline for Composition-Property Linking")
    parser.add_argument(
        "--pii",
        required=True,
        help="PII directory name (e.g., 'S0167273808006176')"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens for LLM generation (default: 64)"
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
        help="Save metadata for each query (default: False)"
    )    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output (default: True)"
    )

    args = parser.parse_args()
    
    # Create custom config
    config = Config()
    config.MAX_TOKENS = args.max_tokens
    config.TEMPERATURE = args.temperature
    config.SAVE_METADATA = args.save_metadata
    config.VERBOSE = args.verbose

    try:
        # Run the pipeline
        summary = run_pipeline(args.pii, config)
        
        print("\n" + "="*60)
        print("LINKING PIPELINE SUMMARY")
        print("="*60)
        for key, value in summary.items():
            if key == 'files_created':
                print(f"{key}: {len(value)} files")
                for file in value:
                    print(f"  - {file}")
            else:
                print(f"{key}: {value}")
        
    except Exception as e:
        print(f"❌ Linking pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
