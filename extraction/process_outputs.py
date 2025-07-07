import json
from typing import List

def extract_compositions_from_raw_output(raw_output: str) -> List[str]:
    """
    Extract compositions from standardized model output.
    
    Expected input format:
    [{"idx": 1, "composition": "Sr0.94Ti0.9Nb0.1O3"}, {"idx": 2, "composition": "MnO2"}]
    
    Returns list of composition strings.
    """
    if not raw_output or not raw_output.strip():
        return []
    
    try:
        parsed = json.loads(raw_output.strip())
        
        if isinstance(parsed, list):
            compositions = []
            for item in parsed:
                if isinstance(item, dict) and "composition" in item:
                    comp = item["composition"]
                    if comp and isinstance(comp, str):
                        compositions.append(comp.strip())
            return compositions
        
    except json.JSONDecodeError:
        pass
    
    return []
