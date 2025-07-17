import os
import pickle
import argparse
from typing import List, Dict, Any, Optional

from bs4 import BeautifulSoup
from tqdm import tqdm

# CHANGE 1: Import the full table extractor
from mit_utils.table_extractor import TableExtractor
import csv
import json


def clean_table_data(table: List[List[Any]]) -> List[List[str]]:
    """
    Clean table data by removing None values and excessive empty cells.
    
    Args:
        table: Raw table data (list of lists)
        
    Returns:
        Cleaned table data
    """
    if not table:
        return []
    
    cleaned_table = []
    
    for row in table:
        # Convert None values to empty strings and clean up
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append('')
            elif isinstance(cell, str):
                # Strip whitespace and convert "None" strings to empty
                cell_clean = cell.strip()
                if cell_clean.lower() == 'none':
                    cleaned_row.append('')
                else:
                    cleaned_row.append(cell_clean)
            else:
                cleaned_row.append(str(cell).strip())
        
        # Remove trailing empty cells from the row
        while cleaned_row and cleaned_row[-1] == '':
            cleaned_row.pop()
        
        # Only add non-empty rows
        if any(cell.strip() for cell in cleaned_row if cell):
            cleaned_table.append(cleaned_row)
    
    return cleaned_table


def normalize_table_columns(table: List[List[str]], max_empty_ratio: float = 0.8) -> List[List[str]]:
    """
    Normalize table columns by padding only when necessary and removing empty columns.
    
    Args:
        table: Cleaned table data
        max_empty_ratio: Maximum ratio of empty cells allowed in a column
        
    Returns:
        Normalized table
    """
    if not table:
        return []
    
    # Find the actual maximum meaningful column count
    max_cols = max(len(row) for row in table) if table else 0
    
    if max_cols == 0:
        return []
    
    # Pad rows to max_cols but don't overdo it
    normalized_table = []
    for row in table:
        padded_row = row + [''] * (max_cols - len(row))
        normalized_table.append(padded_row)
    
    # Remove columns that are mostly empty
    if normalized_table:
        columns_to_keep = []
        for col_idx in range(max_cols):
            col_values = [row[col_idx] if col_idx < len(row) else '' for row in normalized_table]
            non_empty_count = sum(1 for val in col_values if val.strip())
            empty_ratio = 1 - (non_empty_count / len(col_values))
            
            if empty_ratio <= max_empty_ratio:
                columns_to_keep.append(col_idx)
        
        # Keep only meaningful columns
        if columns_to_keep:
            filtered_table = []
            for row in normalized_table:
                filtered_row = [row[col_idx] if col_idx < len(row) else '' for col_idx in columns_to_keep]
                filtered_table.append(filtered_row)
            return filtered_table
    
    return normalized_table


# CHANGE 2: New function that uses the full ML pipeline
def get_tables_full_ml(xml_path: str, domain_name: str = None, use_ml: bool = True) -> List[Dict[str, Any]]:
    """
    Extract tables using the full ML-powered table extractor.
    
    Args:
        xml_path (str): Path to the XML file to process.
        domain_name (str): Domain for materials-specific processing ('steel', 'aluminum', etc.)
        use_ml (bool): Whether to use ML classification or simple heuristics
        
    Returns:
        List[Dict[str, Any]]: List of processed table objects with rich metadata
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    # CHANGE 3: Initialize with domain and embedding parameters
    try:
        te = TableExtractor(
            domain_name=domain_name, 
            embedding_loc='bin/fasttext_embeddings-MINIFIED.model'
        )
    except Exception as e:
        print(f"Warning: Could not load embeddings: {e}")
        print("Falling back to simple extraction...")
        return get_tables_simple(xml_path)
    
    # Extract DOI from XML
    doi = None
    try:
        with open(xml_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'xml')
            doi_tags = soup.find_all('xocs:doi')
            if doi_tags:
                doi = doi_tags[0].text.strip()
    except Exception as e:
        raise ValueError(f"Error reading XML file: {e}")
    
    if doi is None:
        raise ValueError(f"DOI not found in XML file: {xml_path}")
    
    te.doi = doi
    
    # CHANGE 4: Handle the full return format with references
    try:
        # The full version returns: tables, captions, footers, table_refs, caption_refs
        tables, captions, footers, table_refs, caption_refs = te.get_xml_tables(xml_path)
    except Exception as e:
        raise ValueError(f"Error extracting tables from XML: {e}")
    
    # CHANGE 5: Use ML pipeline if requested and models are available
    processed_tables = []
    
    if use_ml and tables:
        try:
            # Load the classifier
            with open('bin/word_classifier_python3.pkl', 'rb') as f:
                te.clf = pickle.load(f)
            
            # Run the full ML pipeline
            cols, rows, col_inds, row_inds = te.get_headers(tables)
            pred_cols, pred_rows = te.classify_table_headers(cols, rows)
            
            # Determine table orientations and composition flags
            orients = []
            composition_flags = []
            for pred_col, pred_row, col, row in zip(pred_cols, pred_rows, cols, rows):
                orient, composition_flag = te.determine_table_orientation(pred_col, pred_row, col, row)
                orients.append(orient)
                composition_flags.append(composition_flag)
            
            # Construct table objects using the full pipeline
            table_objects = []
            for table, row_ind, col_ind, orient, ref in zip(tables, row_inds, col_inds, orients, table_refs):
                try:
                    table_obj = te.construct_table_object(orient, table, row_ind, col_ind, ref)
                    table_objects.append(table_obj)
                except Exception as e:
                    print(f"Warning: Failed to construct table object: {e}")
                    table_objects.append(None)
            
            # Process final table objects with metadata
            for i, (table_obj, comp_flag, caption, footer, capt_ref) in enumerate(
                zip(table_objects, composition_flags, captions, footers, caption_refs)
            ):
                if table_obj is not None:
                    # Add metadata
                    table_obj['order'] = i
                    table_obj['paper_doi'] = doi
                    table_obj['composition_table'] = comp_flag
                    table_obj['caption'] = caption if caption else ''
                    table_obj['footer'] = footer if footer else {}
                    
                    if capt_ref:
                        table_obj['caption_ref'] = capt_ref
                    
                    # Clean composition tables if needed
                    if comp_flag and hasattr(te, 'remaining'):
                        table_obj = te.clean_composition_table(table_obj, remaining=te.remaining)
                    
                    processed_tables.append(table_obj)
                    
        except Exception as e:
            print(f"Warning: ML pipeline failed: {e}")
            print("Falling back to simple processing...")
            use_ml = False
    
    # CHANGE 6: Fallback to simple processing if ML fails or is disabled
    if not use_ml or not processed_tables:
        for i, (table, caption, footer) in enumerate(zip(tables, captions, footers)):
            # Clean and normalize the table data
            cleaned_table = clean_table_data(table)
            normalized_table = normalize_table_columns(cleaned_table)
            
            # Skip empty tables
            if not normalized_table:
                continue
                
            # Check for trivial table
            if (len(normalized_table) == 1 and len(normalized_table[0]) == 1 and 
                not normalized_table[0][0].strip()):
                continue
            
            # Create simple table info
            table_info = {
                'order': i,
                'paper_doi': doi,
                'doi': doi,  # Keep both for compatibility
                'act_table': normalized_table,
                'caption': caption if caption else '',
                'footer': footer if footer else {},
                'composition_table': False,  # Default since we're not using ML
                'num_rows': len(normalized_table),
                'num_cols': len(normalized_table[0]) if normalized_table else 0
            }
            processed_tables.append(table_info)
    
    return processed_tables


# CHANGE 7: Keep the simple version as fallback
def get_tables_simple(xml_path: str) -> List[Dict[str, Any]]:
    """
    Simple table extraction using the MIT version (fallback).
    """
    # Import the simple version for fallback
    from mit_utils.mit_table_extractor import TableExtractor as SimpleExtractor
    
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    te = SimpleExtractor()
    
    # Extract DOI
    doi = None
    try:
        with open(xml_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'xml')
            doi_tags = soup.find_all('xocs:doi')
            if doi_tags:
                doi = doi_tags[0].text.strip()
    except Exception as e:
        raise ValueError(f"Error reading XML file: {e}")
    
    if doi is None:
        raise ValueError(f"DOI not found in XML file: {xml_path}")
    
    te.doi = doi
    
    try:
        tables, captions, footers = te.get_xml_tables(xml_path)
    except Exception as e:
        raise ValueError(f"Error extracting tables from XML: {e}")
    
    processed_tables = []
    for table, caption, footer in zip(tables, captions, footers):
        cleaned_table = clean_table_data(table)
        normalized_table = normalize_table_columns(cleaned_table)
        
        if not normalized_table:
            continue
            
        table_info = {
            'doi': doi,
            'act_table': normalized_table,
            'caption': caption if caption else '',
            'footer': footer if footer else {},
            'num_rows': len(normalized_table),
            'num_cols': len(normalized_table[0]) if normalized_table else 0
        }
        processed_tables.append(table_info)
    
    return processed_tables


# CHANGE 8: Update the main get_tables function
def get_tables(xml_path: str, domain_name: str = None, use_ml: bool = True) -> List[Dict[str, Any]]:
    """
    Extract tables from a single XML file with optional ML enhancement.
    
    Args:
        xml_path (str): Path to the XML file to process.
        domain_name (str): Domain for materials-specific processing 
                          ('steel', 'aluminum', 'titanium', 'zeolites', etc.)
        use_ml (bool): Whether to use ML classification (requires model files)
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing table information.
    """
    try:
        return get_tables_full_ml(xml_path, domain_name, use_ml)
    except Exception as e:
        print(f"Warning: Full ML extraction failed: {e}")
        print("Falling back to simple extraction...")
        return get_tables_simple(xml_path)


# Keep all your existing save functions unchanged...
def save_tables_as_json(tables: List[Dict[str, Any]], output_path: str) -> None:
    """Save extracted tables to a JSON file with metadata and separate CSV files for table data."""
    # Get the directory where the JSON file will be saved
    output_dir = os.path.dirname(output_path)
    
    json_data = {
        'num_tables': len(tables),
        'extraction_timestamp': None,
        'tables': []
    }
    
    for idx, table_info in enumerate(tables):
        table_id = idx + 1
        csv_filename = f"table_{table_id}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        # Save the table data as CSV
        table_data = table_info.get('act_table', [])
        if table_data:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                for row in table_data:
                    writer.writerow(row)
        
        # Create JSON entry with reference to CSV file instead of raw data
        table_entry = {
            'table_id': table_id,
            'doi': table_info.get('doi', table_info.get('paper_doi', '')),
            'caption': table_info.get('caption', ''),
            'footer': table_info.get('footer', {}),
            'csv_file': csv_filename,
            'num_rows': table_info.get('num_rows', len(table_data)),
            'num_cols': table_info.get('num_cols', len(table_data[0]) if table_data else 0),
            'composition_table': table_info.get('composition_table', False)
        }
        json_data['tables'].append(table_entry)
    
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=2, ensure_ascii=False)


def save_tables_as_csv(tables: List[Dict[str, Any]], output_dir: str, base_name: str) -> None:
    """Save extracted tables as separate CSV files with metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        'doi': tables[0].get('doi', tables[0].get('paper_doi', '')) if tables else None,
        'num_tables': len(tables),
        'tables_metadata': []
    }
    
    for idx, table_info in enumerate(tables):
        table_num = idx + 1
        
        csv_path = os.path.join(output_dir, f"{base_name}_table_{table_num}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            for row in table_info.get('act_table', []):
                writer.writerow(row)
        
        metadata['tables_metadata'].append({
            'table_id': table_num,
            'caption': table_info.get('caption', ''),
            'footer': table_info.get('footer', {}),
            'csv_file': f"{base_name}_table_{table_num}.csv"
        })
    
    metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as meta_file:
        json.dump(metadata, meta_file, indent=2, ensure_ascii=False)


def save_single_table_csv(tables: List[Dict[str, Any]], output_path: str) -> None:
    """Save extracted tables as a single CSV file with table separators."""
    with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        for idx, table_info in enumerate(tables):
            writer.writerow([f"=== TABLE {idx + 1} ==="])
            writer.writerow([f"DOI: {table_info.get('doi', table_info.get('paper_doi', ''))}"])
            writer.writerow([f"Caption: {table_info.get('caption', '')}"])
            writer.writerow([f"Footer: {json.dumps(table_info.get('footer', {}))}"])
            writer.writerow([])
            
            for row in table_info.get('act_table', []):
                writer.writerow(row)
            
            writer.writerow([])


# CHANGE 9: Update main function with new parameters
def main(input_dir, output_dir, domain_name=None, use_ml=True):
    """
    Processes XML files using the full ML pipeline.
    
    Args:
        input_dir (str): Path to the directory containing all XML files.
        output_dir (str): Path to the directory where JSON files will be saved.
        domain_name (str): Domain for materials-specific processing.
        use_ml (bool): Whether to use ML classification.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    xml_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.xml')])
    
    for xml_file in tqdm(xml_files, desc="Processing XML files"):
        xml_path = os.path.join(input_dir, xml_file)
        base_name = os.path.splitext(xml_file)[0]
        json_path = os.path.join(output_dir, f"{base_name}_tables.json")
        
        if os.path.exists(json_path):
            os.remove(json_path)
        
        try:
            # Use the enhanced get_tables function
            processed_tables = get_tables(xml_path, domain_name=domain_name, use_ml=use_ml)
            
            if not processed_tables:
                print(f"No valid tables found in {xml_file}.")
                continue
            
            save_tables_as_json(processed_tables, json_path)
            
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            continue
    
    print(f"Processing complete. Extracted tables are saved in {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tables from XML files using ML-enhanced pipeline.")
    parser.add_argument('--input_dir', type=str, required=True, 
                       help="Path to the directory containing all XML files.")
    parser.add_argument('--output_dir', type=str, required=True,
                       help="Path to the directory where JSON files will be saved.")
    parser.add_argument('--domain', type=str, default=None,
                       choices=['steel', 'aluminum', 'titanium', 'zeolites', 'geopolymers', 'alloys'],
                       help="Domain for materials-specific processing.")
    parser.add_argument('--no-ml', action='store_true',
                       help="Disable ML classification (use simple extraction).")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.domain, not args.no_ml)
