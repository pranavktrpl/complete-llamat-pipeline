#!/usr/bin/env python3
"""
Orchestrator for Complete LLaMAT Pipeline
Processes all PII directories in input/ using extraction and linking pipelines.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Import the pipeline modules
from extraction import run_pipeline as run_extraction_pipeline, Config as ExtractionConfig
from link import run_pipeline as run_linking_pipeline, Config as LinkingConfig


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


def process_pii(pii_id: str) -> Dict[str, Any]:
    """Process a single PII through both extraction and linking pipelines."""
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
        extraction_config = ExtractionConfig()
        extraction_summary = run_extraction_pipeline(pii_id, extraction_config)
        results['extraction_success'] = True
        results['extraction_summary'] = extraction_summary
        print(f"✅ Extraction completed for {pii_id}")
        
        # Step 2: Run linking pipeline
        print(f"\n🔗 Running linking pipeline for {pii_id}...")
        linking_config = LinkingConfig()
        linking_summary = run_linking_pipeline(pii_id, linking_config)
        results['linking_success'] = True
        results['linking_summary'] = linking_summary
        print(f"✅ Linking completed for {pii_id}")
        
    except Exception as e:
        error_msg = f"Error processing {pii_id}: {str(e)}"
        print(f"❌ {error_msg}")
        results['error'] = error_msg
    
    return results


def main():
    """Main orchestrator function."""
    print("🚀 Starting LLaMAT Pipeline Orchestrator")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Ensure required directories exist
    ensure_directories()
    
    # Get all PII directories
    pii_directories = get_pii_directories()
    
    if not pii_directories:
        print("❌ No PII directories found in input/")
        sys.exit(1)
    
    print(f"📋 Found {len(pii_directories)} PII directories to process:")
    for pii in pii_directories:
        print(f"   - {pii}")
    
    # Process each PII
    all_results = []
    successful_extractions = 0
    successful_linkings = 0
    
    for pii_id in pii_directories:
        result = process_pii(pii_id)
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
