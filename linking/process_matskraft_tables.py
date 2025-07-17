#!/usr/bin/env python3
"""
Process Matskraft table CSV to extract property queries for linking.

This module reads the Matskraft_table.csv file and extracts property queries
in the format required by linking_util.py for composition-property linking.
"""

import csv
import ast
import os
from typing import List, Tuple, Any


def parse_property_tuple(property_str: str) -> Tuple[str, float, str]:
    """
    Parse a property string from CSV into its components.
    
    Args:
        property_str: String representation of tuple like "('Activation energy', 0.67, 'eV')"
    
    Returns:
        Tuple of (property_name, value, unit)
    """
    try:
        # Use ast.literal_eval to safely parse the tuple string
        parsed_tuple = ast.literal_eval(property_str)
        if isinstance(parsed_tuple, tuple) and len(parsed_tuple) == 3:
            property_name, value, unit = parsed_tuple
            return str(property_name), float(value), str(unit)
        else:
            raise ValueError(f"Invalid tuple format: {property_str}")
    except Exception as e:
        raise ValueError(f"Failed to parse property tuple '{property_str}': {e}")


def format_property_query(property_name: str, value: float, unit: str) -> str:
    """
    Format property components into the query string format required by linking_util.py.
    
    Args:
        property_name: Name of the property (e.g., "Activation energy")
        value: Numerical value of the property
        unit: Unit of the property (e.g., "eV")
    
    Returns:
        Formatted string like "Activation energy : 1.64 eV"
    """
    return f"{property_name} : {value} {unit}"


def load_property_queries_from_csv(csv_path: str) -> List[str]:
    """
    Load property queries from Matskraft table CSV file.
    
    Args:
        csv_path: Path to the Matskraft_table.csv file
    
    Returns:
        List of formatted property query strings
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    property_queries = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            property_str = row.get('Property', '').strip()
            
            # Skip empty properties
            if not property_str:
                continue
                
            try:
                # Parse the property tuple
                property_name, value, unit = parse_property_tuple(property_str)
                
                # Format as query string
                query = format_property_query(property_name, value, unit)
                property_queries.append(query)
                
            except ValueError as e:
                print(f"Warning: Skipping invalid property '{property_str}': {e}")
                continue
    
    # Remove duplicates while preserving order
    unique_queries = []
    seen = set()
    for query in property_queries:
        if query not in seen:
            unique_queries.append(query)
            seen.add(query)
    
    return unique_queries


def get_matskraft_property_queries(paper_id: str = "S0167273808006176") -> List[str]:
    """
    Get property queries from Matskraft table for the specified paper.
    
    Args:
        paper_id: Paper identifier for locating the CSV file
    
    Returns:
        List of formatted property query strings suitable for linking_util.py
    """
    csv_path = f"input/{paper_id}/Matskraft_table.csv"
    return load_property_queries_from_csv(csv_path)


def main():
    """Test function to demonstrate usage."""
    try:
        queries = get_matskraft_property_queries()
        print(f"Found {len(queries)} property queries:")
        for i, query in enumerate(queries, 1):
            print(f"  {i}. {query}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
