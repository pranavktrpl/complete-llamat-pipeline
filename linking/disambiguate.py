#!/usr/bin/env python3
"""
Process structured query JSON files to extract final disambiguated answers.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
import glob

def process_structured_query_files(structured_files_dir: str, output_dir: str) -> str:
    """
    Process structured query JSON files to extract final answers with confidence scoring.
    
    Args:
        structured_files_dir: Path to directory containing structured_query_*.json files
        output_dir: Path to directory where final output JSON should be saved
    
    Returns:
        Path to the created output file
    """
    
    structured_files_path = Path(structured_files_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all structured query JSON files
    pattern = str(structured_files_path / "*query_*_structured.json")
    query_files = glob.glob(pattern)
    
    if not query_files:
        # Try alternative pattern
        pattern = str(structured_files_path / "structured_query_*.json")
        query_files = glob.glob(pattern)
    
    if not query_files:
        raise FileNotFoundError(f"No structured query files found in {structured_files_dir}")
    
    print(f"📂 Found {len(query_files)} structured query files")
    
    # Process each query file
    final_results = {}
    
    for query_file in sorted(query_files):
        print(f"   Processing: {os.path.basename(query_file)}")
        
        with open(query_file, 'r', encoding='utf-8') as f:
            query_data = json.load(f)
        
        # Extract query name and process results
        if query_data:
            query_name = query_data[0].get("query", f"Query from {os.path.basename(query_file)}")
            final_answer = process_single_query_results(query_data, query_name)
            final_results[query_name] = final_answer
    
    # Save final results
    output_file = output_path / "final_disambiguated_answers.json"
    
    output_data = {
        "metadata": {
            "total_queries": len(final_results),
            "input_directory": str(structured_files_path),
            "processing_method": "majority_voting_with_confidence"
        },
        "final_answers": final_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Final results saved to: {output_file}")
    return str(output_file)

def process_single_query_results(query_results: List[Dict[str, Any]], query_name: str) -> Dict[str, Any]:
    """
    Process results for a single query to extract the most appropriate answer.
    
    Args:
        query_results: List of structured results for this query
        query_name: Name/text of the query
    
    Returns:
        Dictionary with most_appropriate_output, confidence, and alternatives
    """
    
    # Extract all compositions from the results
    compositions = []
    confidence_info = {}  # Track confidence info per composition
    
    for result in query_results:
        composition = result.get("composition", "").strip()
        confidence = result.get("confidence", "unknown")
        
        if composition:  # Only include non-empty compositions
            compositions.append(composition)
            
            # Track confidence information
            if composition not in confidence_info:
                confidence_info[composition] = {
                    "llm_confidences": [],
                    "evidences": [],
                    "reasonings": []
                }
            
            confidence_info[composition]["llm_confidences"].append(confidence)
            confidence_info[composition]["evidences"].append(result.get("evidence", ""))
            confidence_info[composition]["reasonings"].append(result.get("reasoning", ""))
    
    if not compositions:
        return {
            "most_appropriate_output": None,
            "confidence": 0.0,
            "confidence_level": "none",
            "supporting_chunks": 0,
            "total_chunks": len(query_results),
            "alternatives": [],
            "metadata": {
                "query": query_name,
                "note": "No valid compositions found in any chunk"
            }
        }
    
    # Count composition frequencies
    comp_counts = Counter(compositions)
    total_chunks = len(compositions)
    
    # Get most frequent composition
    most_common_comp, frequency = comp_counts.most_common(1)[0]
    confidence_score = frequency / total_chunks
    
    # Determine confidence level
    if confidence_score >= 0.8:
        confidence_level = "high"
    elif confidence_score >= 0.5:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    # Build alternatives list
    alternatives = []
    for comp, count in comp_counts.most_common()[1:]:  # Skip the most common
        alt_confidence = count / total_chunks
        
        alternatives.append({
            "composition": comp,
            "confidence": alt_confidence,
            "supporting_chunks": count,
            "llm_confidence_distribution": Counter(confidence_info[comp]["llm_confidences"])
        })
    
    # Get LLM confidence distribution for the main answer
    main_llm_confidences = Counter(confidence_info[most_common_comp]["llm_confidences"])
    
    return {
        "most_appropriate_output": most_common_comp,
        "confidence": confidence_score,
        "confidence_level": confidence_level,
        "supporting_chunks": frequency,
        "total_chunks": total_chunks,
        "alternatives": alternatives,
        "metadata": {
            "query": query_name,
            "llm_confidence_distribution": dict(main_llm_confidences),
            "total_compositions_found": len(comp_counts),
            "composition_distribution": dict(comp_counts)
        }
    }

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process structured query files to extract final answers")
    parser.add_argument(
        "structured_files_dir", 
        help="Path to directory containing structured query JSON files"
    )
    parser.add_argument(
        "output_dir",
        help="Path to directory where final output should be saved"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed processing information"
    )
    
    args = parser.parse_args()
    
    try:
        output_file = process_structured_query_files(
            args.structured_files_dir, 
            args.output_dir
        )
        
        if args.verbose:
            # Print summary
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            print("\n🎯 PROCESSING SUMMARY")
            print("=" * 50)
            
            for query, result in results["final_answers"].items():
                print(f"\nQuery: {query}")
                print(f"  Final Answer: {result['most_appropriate_output']}")
                print(f"  Confidence: {result['confidence_level']} ({result['confidence']:.2%})")
                print(f"  Evidence: {result['supporting_chunks']}/{result['total_chunks']} chunks")
                
                if result['alternatives']:
                    print(f"  Alternatives:")
                    for alt in result['alternatives']:
                        print(f"    - {alt['composition']} ({alt['confidence']:.2%})")
        
        print(f"\n✅ Processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error processing files: {e}")
        raise

if __name__ == "__main__":
    main()
