import json
import re
import os
from typing import List, Dict, Any

#Enhanced function for processing the newer structured output
def process_linking_results_enhanced(
        input_file: str = "checkpoints/linking_results_raw.json",
        output_file: str = "output/S0167273808006176/linking_results.json",
        structured_output_dir: str = "output/S0167273808006176/structured_linking_results",
) -> List[Dict[str, str]]:
    """
    Process raw linking results and extract both simple and structured query-composition pairs.

    The ``structured_output_dir`` parameter is expected to be a *directory*.
    A separate JSON file will be created for each unique query, e.g.::

        structured_output_dir/
            ├─ linking_query_1_structured.json
            ├─ linking_query_2_structured.json
            └─ ...

    Each file contains the list of structured answers belonging to that query.
    The simple list of query-to-composition pairs is still written to
    ``output_file`` unchanged.
    """
    # --- Load raw results ----------------------------------------------------
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        raw_results = json.load(f)

    # --- Parse results -------------------------------------------------------
    processed_results: List[Dict[str, str]] = []
    structured_results: List[Dict[str, Any]] = []

    for result in raw_results:
        if "error" in result:
            continue

        query = result.get("query", "")
        raw_output = result.get("raw_output", "").strip()

        if not raw_output:
            continue

        # Extract all compositions and structured results from the raw output
        all_compositions = extract_all_compositions_from_output(raw_output)
        all_structured_results = extract_all_linking_results(raw_output)

        # Process simple compositions
        for composition in all_compositions:
            if composition:
                processed_results.append({"query": query, "composition": composition})

        # Process structured results
        for i, full_result in enumerate(all_structured_results):
            full_result["query"] = query
            full_result["query_idx"] = result.get("query_idx", -1)
            full_result["chunk_idx"] = result.get("chunk_idx", -1)
            full_result["result_idx"] = i  # Index within the array for this query-chunk pair
            structured_results.append(full_result)

    # --- Sort for determinism ------------------------------------------------
    processed_results.sort(key=lambda x: x["query"])
    structured_results.sort(key=lambda x: (x["query"], x.get("query_idx", 0)))

    # --- Write simple list ---------------------------------------------------
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)

    # --- Write per-query structured files ------------------------------------
    os.makedirs(structured_output_dir, exist_ok=True)

    # Group structured results by query
    query_to_results: Dict[str, List[Dict[str, Any]]] = {}
    for res in structured_results:
        query_to_results.setdefault(res["query"], []).append(res)

    for idx, (query, results) in enumerate(query_to_results.items(), start=1):
        file_name = f"linking_query_{idx}_structured.json"
        file_path = os.path.join(structured_output_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Simple linking results saved to: {output_file}")
    print(
        f"✅ Structured linking results saved to directory: {structured_output_dir} (" \
        f"{len(query_to_results)} files)"
    )
    print(f"   Found {len(processed_results)} query-composition pairs")

    return processed_results

def extract_all_compositions_from_output(raw_output: str) -> List[str]:
    """
    Extract all compositions from structured JSON output.
    """
    if not raw_output:
        return []
    
    try:
        # Try to parse as JSON
        output_data = json.loads(raw_output.strip())
        
        # Handle case where output is a list of objects
        if isinstance(output_data, list):
            compositions = []
            for item in output_data:
                if isinstance(item, dict):
                    comp = item.get("composition", "").strip()
                    if comp:
                        compositions.append(comp)
            return compositions
        # Handle case where output is a single object
        elif isinstance(output_data, dict):
            comp = output_data.get("composition", "").strip()
            return [comp] if comp else []
        else:
            return []
    except json.JSONDecodeError:
        # Fallback: use the original single extraction method
        comp = extract_composition_from_output(raw_output)
        return [comp] if comp else []

def extract_all_linking_results(raw_output: str) -> List[Dict[str, str]]:
    """
    Extract all structured results from JSON output.
    """
    if not raw_output:
        return []
    
    try:
        output_data = json.loads(raw_output.strip())
        
        # Handle case where output is a list of objects
        if isinstance(output_data, list):
            results = []
            for item in output_data:
                if isinstance(item, dict):
                    result = {
                        "composition": item.get("composition", "").strip(),
                        "confidence": item.get("confidence", "none"),
                        "evidence": item.get("evidence", ""),
                        "reasoning": item.get("reasoning", "")
                    }
                    results.append(result)
            return results
        # Handle case where output is a single object
        elif isinstance(output_data, dict):
            result = {
                "composition": output_data.get("composition", "").strip(),
                "confidence": output_data.get("confidence", "none"),
                "evidence": output_data.get("evidence", ""),
                "reasoning": output_data.get("reasoning", "")
            }
            return [result]
        else:
            return []
    except json.JSONDecodeError:
        # Fallback: use the original single extraction method
        result = extract_full_linking_result(raw_output)
        return [result] if result.get("composition") else []

#New function for extracting the composition from the structured JSON output
def extract_composition_from_output(raw_output: str) -> str:
    """
    Extract composition from structured JSON output.
    """
    if not raw_output:
        return ""
    
    try:
        # Try to parse as JSON
        output_data = json.loads(raw_output.strip())
        
        # Handle case where output is a list of objects
        if isinstance(output_data, list) and len(output_data) > 0:
            # Return the composition from the first object (highest confidence)
            return output_data[0].get("composition", "").strip()
        # Handle case where output is a single object
        elif isinstance(output_data, dict):
            return output_data.get("composition", "").strip()
        else:
            return ""
    except json.JSONDecodeError:
        # Fallback: try to extract composition from unstructured output
        output = raw_output.strip()
        
        # Look for JSON-like pattern
        json_match = re.search(r'\{"composition":\s*"([^"]+)"', output)
        if json_match:
            return json_match.group(1)
        
        # If it's just a composition name, return it
        return output

def extract_full_linking_result(raw_output: str) -> Dict[str, str]:
    """
    Extract full structured result from JSON output.
    """
    if not raw_output:
        return {"composition": "", "confidence": "none", "evidence": "", "reasoning": ""}
    
    try:
        output_data = json.loads(raw_output.strip())
        
        # Handle case where output is a list of objects
        if isinstance(output_data, list) and len(output_data) > 0:
            # Use the first object (highest confidence)
            first_result = output_data[0]
            return {
                "composition": first_result.get("composition", "").strip(),
                "confidence": first_result.get("confidence", "none"),
                "evidence": first_result.get("evidence", ""),
                "reasoning": first_result.get("reasoning", "")
            }
        # Handle case where output is a single object
        elif isinstance(output_data, dict):
            return {
                "composition": output_data.get("composition", "").strip(),
                "confidence": output_data.get("confidence", "none"),
                "evidence": output_data.get("evidence", ""),
                "reasoning": output_data.get("reasoning", "")
            }
        else:
            return {"composition": "", "confidence": "none", "evidence": "", "reasoning": ""}
    except json.JSONDecodeError:
        # Fallback for unstructured output
        return {
            "composition": extract_composition_from_output(raw_output),
            "confidence": "unknown",
            "evidence": "",
            "reasoning": "Unstructured output - could not parse JSON"
        }

def main():
    """Main function to process linking results."""
    try:
        results = process_linking_results_enhanced()
        
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
