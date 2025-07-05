import spacy
from spacy.language import Language
from typing import List, Literal


@Language.component("scientific_sentence_boundaries")
def scientific_sentence_boundaries(doc):
    """Custom component for handling scientific text sentence boundaries"""
    for token in doc:
        # Don't split on periods in chemical formulas
        if token.text.endswith('.'):
            if any(char.isupper() for char in token.text[:-1]):  # Chemical formula
                token.is_sent_start = False
            if any(char.isdigit() for char in token.text):  # Measurement
                token.is_sent_start = False
                
        # Don't split on chemical formula patterns
        if token.i > 0:
            prev_token = doc[token.i - 1]
            # Pattern like "H2O" or "NaCl"
            if (prev_token.text[0].isupper() and 
                any(c.isdigit() for c in prev_token.text)):
                token.is_sent_start = False
    return doc


def _load_model(model_type: str):
    """Load the appropriate spaCy model based on type"""
    if model_type == "sci-spacy":
        # Try to load sci-spacy model
        nlp = spacy.load("en_core_sci_sm")
        print("Using sci-spacy model: en_core_sci_sm")
    else:
        # Load standard spaCy model
        nlp = spacy.load("en_core_web_sm")
        print("Using standard spaCy model: en_core_web_sm")
    
    return nlp


def split_sentences(text: str, model: Literal["std-spacy", "sci-spacy"] = "sci-spacy") -> List[str]:
    """
    Split scientific text into sentences.
    
    Args:
        text (str): The input text to split
        model (str): Model to use - "spacy" for standard or "sci-spacy" for scientific
        
    Returns:
        List[str]: List of sentences
    """
    # Load appropriate model
    nlp = _load_model(model)
    
    # Add custom scientific text component if not already added
    if "scientific_sentence_boundaries" not in nlp.pipe_names:
        nlp.add_pipe("scientific_sentence_boundaries", before="parser")
    
    # Process text
    doc = nlp(text)
    
    # Extract sentences and clean them
    sentences = [str(sent).strip() for sent in doc.sents if str(sent).strip()]
    
    return sentences


def split_to_dict(text: str, model: Literal["std-spacy", "sci-spacy"] = "sci-spacy") -> dict:
    """
    Split text into sentences and return as numbered dictionary.
    
    Args:
        text (str): The input text to split
        model (str): Model to use - "spacy" for standard or "sci-spacy" for scientific
        
    Returns:
        dict: {"sentence_1": "...", "sentence_2": "...", ...}
    """
    sentences = split_sentences(text, model=model)
    return {f"sentence_{i}": sentence for i, sentence in enumerate(sentences, 1)}


def test_models(text: str) -> dict:
    """
    Test both models on the same text for comparison.
    
    Args:
        text (str): Sample text to test
        
    Returns:
        dict: Results from both models for comparison
    """
    results = {}
    
    # Test standard spaCy
    try:
        spacy_sentences = split_sentences(text, model="sci-spacy")
        results["spacy"] = {
            "count": len(spacy_sentences),
            "sentences": spacy_sentences
        }
    except Exception as e:
        results["spacy"] = {"error": str(e)}
    
    # Test sci-spacy
    try:
        scispacy_sentences = split_sentences(text, model="sci-spacy")
        results["sci-spacy"] = {
            "count": len(scispacy_sentences),
            "sentences": scispacy_sentences
        }
    except Exception as e:
        results["sci-spacy"] = {"error": str(e)}
    
    return results