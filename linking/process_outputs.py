import json
import os
from typing import List, Dict, Any

def process_linking_results(input_file: str = "checkpoints/linking_results_raw.json", 
                          output_file: str = "output/S0167273808006176/linking_results.json") -> List[Dict[str, str]]:
    """
    Process raw linking results and extract query-composition pairs.
    
    Expected input format from linking_results_raw.json:
    [
      {"query_idx": 0, "chunk_idx": 0, "query": "<property query>", "raw_output": "<LLM reply>"},
      ...
    ]
    
    Output format:
    [
      {"query": "<property query>", "composition": "<extracted composition>"},
      ...
    ]
    
    Returns list of query-composition pairs, sorted by query.
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_results = json.load(f)
    
    # Process results and extract compositions
    processed_results = []
    
    for result in raw_results:
        if "error" in result:
            # Skip results with errors
            continue
            
        query = result.get("query", "")
        raw_output = result.get("raw_output", "").strip()
        
        if not raw_output:
            continue
        
        # Extract composition from raw output
        # The model might return just the composition name or an index
        composition = extract_composition_from_output(raw_output)
        
        if composition:
            processed_results.append({
                "query": query,
                "composition": composition
            })
    
    # Sort by query to group similar queries together
    processed_results.sort(key=lambda x: x["query"])
    
    # Save to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Processed linking results saved to: {output_file}")
    print(f"   Found {len(processed_results)} query-composition pairs")
    
    return processed_results


def extract_composition_from_output(raw_output: str) -> str:
    """
    Extract composition from model output.
    
    The model might return:
    - Just the composition name (e.g., "Sr0.94Ti0.9Nb0.1O3")
    - An index (e.g., "1" or "Composition 1")
    - A sentence containing the composition
    """
    if not raw_output:
        return ""
    
    # Clean up the output
    output = raw_output.strip()
    
    # # If it's a simple composition formula (contains chemical elements)
    # # Look for patterns like "Sr0.94Ti0.9Nb0.1O3", "MnO2", etc.
    # import re
    
    # # Pattern to match chemical formulas
    # chem_formula_pattern = r'[A-Z][a-z]?(?:\d*\.?\d*)*(?:[A-Z][a-z]?(?:\d*\.?\d*)*)*'
    
    # # First, try to find a direct chemical formula
    # formulas = re.findall(chem_formula_pattern, output)
    # if formulas:
    #     # Return the longest formula found (likely the most complete)
    #     return max(formulas, key=len)
    
    # # If no formula found, check if it's just a number (index reference)
    # if output.isdigit():
    #     return f"Composition {output}"
    
    # # Check for patterns like "Composition 1", "composition 2", etc.
    # comp_match = re.search(r'composition\s*(\d+)', output, re.IGNORECASE)
    # if comp_match:
    #     return f"Composition {comp_match.group(1)}"
    
    # # If nothing else works, return the cleaned output as is
    return output


def main():
    """Main function to process linking results."""
    try:
        results = process_linking_results()
        
        # Print summary by query
        query_counts = {}
        for result in results:
            query = result["query"]
            if query not in query_counts:
                query_counts[query] = 0
            query_counts[query] += 1
        
        print("\n📊 Summary by query:")
        for query, count in query_counts.items():
            print(f"  '{query[:50]}...': {count} compositions")
            
    except Exception as e:
        print(f"❌ Error processing linking results: {e}")


if __name__ == "__main__":
    main()
