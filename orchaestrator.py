#!/usr/bin/env python3
"""
Orchestrator for Complete LLaMAT Pipeline
Processes all PII directories in input/ using extraction and linking pipelines.

This script provides centralized configuration management for all hyperparameters
across both the extraction and linking pipelines. All parameters can be configured
either by modifying the OrchestratorConfig class or via command-line arguments.

Usage Examples:
    # Process all PIIs with default settings
    python orchaestrator.py
    
    # Process specific PII with custom chunk size
    python orchaestrator.py --pii S0167273808006176 --extraction-chunk-size 500
    
    # Run with increased token limits and verbose output
    python orchaestrator.py --extraction-max-tokens 256 --linking-max-tokens 512 --verbose
    
    # Process with different temperature settings for experimentation
    python orchaestrator.py --extraction-temperature 0.1 --linking-temperature 0.2
    
    # Save all metadata and stop on first error
    python orchaestrator.py --save-extraction-metadata --save-linking-metadata --stop-on-error

Key Features:
    - Centralized hyperparameter management
    - Command-line argument support for all major parameters
    - Individual PII processing or batch processing
    - Comprehensive error handling and reporting
    - Detailed configuration and progress logging
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

import link
import extract


# =============================================================================
# ORCHESTRATOR CONFIGURATION - Centralized Hyperparameters
# =============================================================================

class OrchestratorConfig:
    """Centralized configuration for the complete LLaMAT pipeline."""
    
    # =============================================================================
    # EXTRACTION PIPELINE CONFIGURATION
    # =============================================================================
    
    # Chunking parameters
    EXTRACTION_CHUNK_SIZE = 700  # Maximum words per chunk
    EXTRACTION_SENTENCE_MODEL = "sci-spacy"  # Model for sentence splitting: "sci-spacy" or "std-spacy"
    
    # LLM parameters for extraction
    EXTRACTION_MAX_TOKENS = 128  # Maximum tokens for LLM generation
    EXTRACTION_TEMPERATURE = 0.0  # Temperature for LLM (0.0 = deterministic)
    
    # Extraction metadata parameters
    EXTRACTION_SAVE_METADATA = False  # Whether to save metadata for each chunk
    EXTRACTION_VERBOSE = True  # Whether to print verbose output
    
    # Extraction file paths
    EXTRACTION_SYSTEM_PROMPT_PATH = "prompts/extraction/system_prompt_composition.txt"
    EXTRACTION_USER_PROMPT_PATH = "prompts/extraction/user_prompt_composition.txt"
    
    # =============================================================================
    # LINKING PIPELINE CONFIGURATION
    # =============================================================================
    
    # LLM parameters for linking
    LINKING_MAX_TOKENS = 256  # Maximum tokens for LLM generation (increased for structured reasoning)
    LINKING_TEMPERATURE = 0.0  # Temperature for LLM (0.0 = deterministic)
    
    # Linking metadata parameters
    LINKING_SAVE_METADATA = True  # Whether to save metadata for each query
    LINKING_VERBOSE = True  # Whether to print verbose output
    
    # Linking file paths (using reasoning-backed prompts)
    LINKING_SYSTEM_PROMPT_PATH = "prompts/linking/system_reasoned_prompt.txt"
    LINKING_USER_PROMPT_PATH = "prompts/linking/user_reasoning_prompt.txt"
    
    # =============================================================================
    # SHARED DIRECTORY STRUCTURE
    # =============================================================================
    
    # Directory structure
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    CHECKPOINTS_DIR = "checkpoints"
    
    # =============================================================================
    # SHARED FILE NAMES
    # =============================================================================
    
    # Input files
    RAW_XML_FILE = "raw_xml.xml"
    MATSKRAFT_TABLE_FILE = "Matskraft_table.csv"
    
    # Extraction pipeline files
    EXTRACTED_TEXT_FILE = "research-paper-text.txt"
    CHUNKS_FILE = "chunks.json"
    EXTRACTION_RAW_OUTPUTS_FILE = "compositions_model_raw.json"
    EXTRACTION_METADATA_FILE = "compositions_metadata.json"
    FINAL_COMPOSITIONS_FILE = "compositions.json"
    
    # Linking pipeline files
    QUERIES_FILE = "queries.json"
    LINKING_RAW_OUTPUTS_FILE = "linking_results_raw.json"
    LINKING_PROCESSED_OUTPUTS_FILE = "linking_results.json"
    LINKING_STRUCTURED_OUTPUTS_DIR = "structured_linking_results"
    LINKING_FINAL_ANSWERS_FILE = "final_disambiguated_answers.json"
    LINKING_METADATA_FILE = "linking_metadata.json"
    
    # =============================================================================
    # ADVANCED CONFIGURATION
    # =============================================================================
    
    # Text processing parameters
    XML_EXCLUDE_SECTIONS = ['references', 'bibliography', 'ref', 'bibliographies', 'reference']
    
    # Disambiguation parameters
    DISAMBIGUATION_HIGH_CONFIDENCE_THRESHOLD = 0.8  # High confidence if >= 80% agreement
    DISAMBIGUATION_MEDIUM_CONFIDENCE_THRESHOLD = 0.5  # Medium confidence if >= 50% agreement
    
    # Error handling
    MAX_RETRIES_PER_CHUNK = 3  # Maximum retries for failed LLM calls
    CONTINUE_ON_ERROR = True  # Whether to continue processing other PIIs if one fails
    
    # Performance parameters
    CHUNK_PROCESSING_BATCH_SIZE = 1  # Number of chunks to process in parallel (future enhancement)
    
    # =============================================================================
    # CONFIGURATION SUMMARY
    # =============================================================================
    """
    All hyperparameters consolidated in one place for easy tuning:
    
    EXTRACTION PIPELINE:
    - EXTRACTION_CHUNK_SIZE: Controls text chunking granularity (700 words default)
    - EXTRACTION_MAX_TOKENS: LLM output length for composition extraction (128 default)
    - EXTRACTION_TEMPERATURE: Determinism vs creativity (0.0 = deterministic)
    - EXTRACTION_SENTENCE_MODEL: Sentence splitting model (sci-spacy recommended)
    - EXTRACTION_SAVE_METADATA: Whether to save detailed extraction metadata
    - EXTRACTION_VERBOSE: Whether to print detailed extraction progress
    
    LINKING PIPELINE:
    - LINKING_MAX_TOKENS: LLM output length for property linking (256 default)
    - LINKING_TEMPERATURE: Determinism vs creativity (0.0 = deterministic)
    - LINKING_SAVE_METADATA: Whether to save detailed linking metadata
    - LINKING_VERBOSE: Whether to print detailed linking progress
    
    SYSTEM CONFIGURATION:
    - Directory paths (INPUT_DIR, OUTPUT_DIR, CHECKPOINTS_DIR)
    - File naming conventions for all intermediate and output files
    - Prompt template paths for both pipelines
    - Error handling behavior (CONTINUE_ON_ERROR)
    - Text processing parameters (XML_EXCLUDE_SECTIONS)
    - Disambiguation thresholds for confidence scoring
    
    All parameters can be overridden via command-line arguments for experimentation.
    """
    
    def get_extraction_config(self):
        """Create extraction config with orchestrator parameters."""
        config = extract.Config()
        config.CHUNK_SIZE = self.EXTRACTION_CHUNK_SIZE
        config.SENTENCE_MODEL = self.EXTRACTION_SENTENCE_MODEL
        config.MAX_TOKENS = self.EXTRACTION_MAX_TOKENS
        config.TEMPERATURE = self.EXTRACTION_TEMPERATURE
        config.SAVE_METADATA = self.EXTRACTION_SAVE_METADATA
        config.VERBOSE = self.EXTRACTION_VERBOSE
        config.SYSTEM_PROMPT_PATH = self.EXTRACTION_SYSTEM_PROMPT_PATH
        config.USER_PROMPT_PATH = self.EXTRACTION_USER_PROMPT_PATH
        config.INPUT_DIR = self.INPUT_DIR
        config.OUTPUT_DIR = self.OUTPUT_DIR
        config.CHECKPOINTS_DIR = self.CHECKPOINTS_DIR
        config.RAW_XML_FILE = self.RAW_XML_FILE
        config.EXTRACTED_TEXT_FILE = self.EXTRACTED_TEXT_FILE
        config.CHUNKS_FILE = self.CHUNKS_FILE
        config.RAW_OUTPUTS_FILE = self.EXTRACTION_RAW_OUTPUTS_FILE
        config.METADATA_FILE = self.EXTRACTION_METADATA_FILE
        config.FINAL_COMPOSITIONS_FILE = self.FINAL_COMPOSITIONS_FILE
        return config
    
    def get_linking_config(self):
        """Create linking config with orchestrator parameters."""
        config = link.Config()
        config.MAX_TOKENS = self.LINKING_MAX_TOKENS
        config.TEMPERATURE = self.LINKING_TEMPERATURE
        config.SAVE_METADATA = self.LINKING_SAVE_METADATA
        config.VERBOSE = self.LINKING_VERBOSE
        config.SYSTEM_PROMPT_PATH = self.LINKING_SYSTEM_PROMPT_PATH
        config.USER_PROMPT_PATH = self.LINKING_USER_PROMPT_PATH
        config.INPUT_DIR = self.INPUT_DIR
        config.OUTPUT_DIR = self.OUTPUT_DIR
        config.CHECKPOINTS_DIR = self.CHECKPOINTS_DIR
        config.CHUNKS_FILE = self.CHUNKS_FILE
        config.COMPOSITIONS_FILE = self.FINAL_COMPOSITIONS_FILE
        config.QUERIES_FILE = self.QUERIES_FILE
        config.RAW_LINKING_OUTPUTS_FILE = self.LINKING_RAW_OUTPUTS_FILE
        config.PROCESSED_LINKING_OUTPUTS_FILE = self.LINKING_PROCESSED_OUTPUTS_FILE
        config.STRUCTURED_LINKING_OUTPUTS_FILE = self.LINKING_STRUCTURED_OUTPUTS_DIR
        config.FINAL_ANSWERS_FILE = self.LINKING_FINAL_ANSWERS_FILE
        config.METADATA_FILE = self.LINKING_METADATA_FILE
        return config


def get_pii_directories(input_dir: str = "input") -> List[str]:
    """Get all PII directory names from the input directory."""
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return []
    
    pii_dirs = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            pii_dirs.append(item)
    
    return sorted(pii_dirs)


def ensure_directories():
    """Ensure output and checkpoints directories exist."""
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)


def process_pii(pii_id: str, orchestrator_config: OrchestratorConfig = None) -> Dict[str, Any]:
    """Process a single PII through both extraction and linking pipelines."""
    if orchestrator_config is None:
        orchestrator_config = OrchestratorConfig()
    
    print(f"\n{'='*80}")
    print(f"PROCESSING PII: {pii_id}")
    print(f"{'='*80}")
    
    results = {
        'pii_id': pii_id,
        'extraction_success': False,
        'linking_success': False,
        'extraction_summary': None,
        'linking_summary': None,
        'error': None
    }
    
    try:
        # Step 1: Run extraction pipeline
        print(f"\n🔬 Running extraction pipeline for {pii_id}...")
        extraction_config = orchestrator_config.get_extraction_config()
        extraction_summary = extract.run_pipeline(pii_id, extraction_config)
        results['extraction_success'] = True
        results['extraction_summary'] = extraction_summary
        print(f"✅ Extraction completed for {pii_id}")
        
        # Step 2: Run linking pipeline
        print(f"\n🔗 Running linking pipeline for {pii_id}...")
        linking_config = orchestrator_config.get_linking_config()
        linking_summary = link.run_pipeline(pii_id, linking_config)
        results['linking_success'] = True
        results['linking_summary'] = linking_summary
        print(f"✅ Linking completed for {pii_id}")
        
    except Exception as e:
        error_msg = f"Error processing {pii_id}: {str(e)}"
        print(f"❌ {error_msg}")
        if not orchestrator_config.CONTINUE_ON_ERROR:
            raise
        results['error'] = error_msg
    
    return results


def main():
    """Main orchestrator function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLaMAT Pipeline Orchestrator - Process research papers for composition extraction and property linking")
    
    # Pipeline selection
    parser.add_argument(
        "--pii",
        type=str,
        help="Process only specific PII directory (e.g., 'S0167273808006176')"
    )
    
    # Extraction hyperparameters
    parser.add_argument(
        "--extraction-chunk-size",
        type=int,
        default=700,
        help="Maximum words per chunk for extraction (default: 700)"
    )
    parser.add_argument(
        "--extraction-max-tokens",
        type=int,
        default=128,
        help="Maximum tokens for extraction LLM generation (default: 128)"
    )
    parser.add_argument(
        "--extraction-temperature",
        type=float,
        default=0.0,
        help="Temperature for extraction LLM (default: 0.0)"
    )
    parser.add_argument(
        "--sentence-model",
        choices=["sci-spacy", "std-spacy"],
        default="sci-spacy",
        help="Sentence splitting model (default: sci-spacy)"
    )
    
    # Linking hyperparameters
    parser.add_argument(
        "--linking-max-tokens",
        type=int,
        default=256,
        help="Maximum tokens for linking LLM generation (default: 256)"
    )
    parser.add_argument(
        "--linking-temperature",
        type=float,
        default=0.0,
        help="Temperature for linking LLM (default: 0.0)"
    )
    
    # Metadata and debugging
    parser.add_argument(
        "--save-extraction-metadata",
        action="store_true",
        help="Save metadata for extraction chunks (default: False)"
    )
    parser.add_argument(
        "--save-linking-metadata",
        action="store_true",
        help="Save metadata for linking queries (default: True)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (default: True)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose output"
    )
    
    # Error handling
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop processing if any PII fails (default: continue on error)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting LLaMAT Pipeline Orchestrator")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Initialize centralized configuration with command line overrides
    orchestrator_config = OrchestratorConfig()
    
    # Apply command line arguments
    orchestrator_config.EXTRACTION_CHUNK_SIZE = args.extraction_chunk_size
    orchestrator_config.EXTRACTION_MAX_TOKENS = args.extraction_max_tokens
    orchestrator_config.EXTRACTION_TEMPERATURE = args.extraction_temperature
    orchestrator_config.EXTRACTION_SENTENCE_MODEL = args.sentence_model
    orchestrator_config.LINKING_MAX_TOKENS = args.linking_max_tokens
    orchestrator_config.LINKING_TEMPERATURE = args.linking_temperature
    orchestrator_config.EXTRACTION_SAVE_METADATA = args.save_extraction_metadata
    orchestrator_config.LINKING_SAVE_METADATA = args.save_linking_metadata
    orchestrator_config.CONTINUE_ON_ERROR = not args.stop_on_error
    
    # Handle verbose flags
    if args.quiet:
        orchestrator_config.EXTRACTION_VERBOSE = False
        orchestrator_config.LINKING_VERBOSE = False
    elif args.verbose:
        orchestrator_config.EXTRACTION_VERBOSE = True
        orchestrator_config.LINKING_VERBOSE = True
    
    # Print configuration summary
    print("\n📋 PIPELINE CONFIGURATION")
    print("=" * 50)
    print(f"Extraction chunk size: {orchestrator_config.EXTRACTION_CHUNK_SIZE} words")
    print(f"Extraction max tokens: {orchestrator_config.EXTRACTION_MAX_TOKENS}")
    print(f"Extraction temperature: {orchestrator_config.EXTRACTION_TEMPERATURE}")
    print(f"Linking max tokens: {orchestrator_config.LINKING_MAX_TOKENS}")
    print(f"Linking temperature: {orchestrator_config.LINKING_TEMPERATURE}")
    print(f"Sentence model: {orchestrator_config.EXTRACTION_SENTENCE_MODEL}")
    print(f"Save extraction metadata: {orchestrator_config.EXTRACTION_SAVE_METADATA}")
    print(f"Save linking metadata: {orchestrator_config.LINKING_SAVE_METADATA}")
    print(f"Verbose output: {orchestrator_config.EXTRACTION_VERBOSE}")
    print(f"Continue on error: {orchestrator_config.CONTINUE_ON_ERROR}")
    
    # Ensure required directories exist
    ensure_directories()
    
    # Determine which PIIs to process
    if args.pii:
        # Process only the specified PII
        pii_directories = [args.pii]
        pii_input_path = os.path.join(orchestrator_config.INPUT_DIR, args.pii)
        if not os.path.exists(pii_input_path):
            print(f"❌ Specified PII directory not found: {pii_input_path}")
            sys.exit(1)
        print(f"\n📋 Processing specific PII: {args.pii}")
    else:
        # Get all PII directories
        pii_directories = get_pii_directories(orchestrator_config.INPUT_DIR)
        
        if not pii_directories:
            print(f"❌ No PII directories found in {orchestrator_config.INPUT_DIR}/")
            sys.exit(1)
        
        print(f"\n📋 Found {len(pii_directories)} PII directories to process:")
        for pii in pii_directories:
            print(f"   - {pii}")
    
    # Process each PII
    all_results = []
    successful_extractions = 0
    successful_linkings = 0
    
    for pii_id in pii_directories:
        result = process_pii(pii_id, orchestrator_config)
        all_results.append(result)
        
        if result['extraction_success']:
            successful_extractions += 1
        if result['linking_success']:
            successful_linkings += 1
    
    # Print final summary
    print(f"\n{'='*80}")
    print("ORCHESTRATOR SUMMARY")
    print(f"{'='*80}")
    print(f"Total PIIs processed: {len(pii_directories)}")
    print(f"Successful extractions: {successful_extractions}/{len(pii_directories)}")
    print(f"Successful linkings: {successful_linkings}/{len(pii_directories)}")
    
    # Detailed results
    print(f"\nDETAILED RESULTS:")
    print("-" * 80)
    for result in all_results:
        pii_id = result['pii_id']
        extraction_status = "✅" if result['extraction_success'] else "❌"
        linking_status = "✅" if result['linking_success'] else "❌"
        
        print(f"{pii_id:20} | Extraction: {extraction_status} | Linking: {linking_status}")
        
        if result['extraction_success'] and result['extraction_summary']:
            ext_summary = result['extraction_summary']
            print(f"{'':20}   └─ Extracted {ext_summary['num_compositions']} compositions from {ext_summary['num_chunks']} chunks")
        
        if result['linking_success'] and result['linking_summary']:
            link_summary = result['linking_summary']
            print(f"{'':20}   └─ Processed {link_summary['num_queries']} queries → {link_summary['num_final_answers']} final answers")
        
        if result['error']:
            print(f"{'':20}   └─ Error: {result['error']}")
    
    # Exit status
    if successful_extractions == len(pii_directories) and successful_linkings == len(pii_directories):
        print(f"\n🎉 All PIIs processed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some PIIs failed processing. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
