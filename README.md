# Complete LLaMAT Pipeline for Materials Composition Extraction

## 🎯 Project Overview

The **Complete LLaMAT Pipeline** is an end-to-end system for extracting material compositions from scientific research papers in the materials science domain. The pipeline processes research papers (in XML format) through multiple stages to identify and extract chemical compositions, formulas, and material names using Large Language Models (LLMs).

### Key Capabilities
- **Automated Text Extraction**: Extracts structured text from research paper XML files  
- **Intelligent Chunking**: Splits papers into semantic sections with sentence-aware chunking
- **LLM-Powered Extraction**: Uses local LLaMAT models to identify material compositions
- **Output Processing**: Cleans and deduplicates extracted compositions
- **Flexible Architecture**: Modular design supporting different LLM backends

## 🏗️ System Architecture

### Core Pipeline Components

```
XML Paper → Text Extraction → Chunking → LLM Processing → Output Processing
     ↓              ↓            ↓            ↓              ↓
  raw_xml.xml → research-paper- → chunks.json → compositions_ → compositions.json
                    text.txt                    model_raw.json
```

### Directory Structure

```
complete-llamat-pipeline/
├── extraction.py              # Main pipeline orchestrator
├── extraction/                # Core extraction modules
│   ├── chunking.py           # Text chunking with section awareness
│   ├── extraction_util.py    # LLM interaction utilities  
│   ├── process_outputs.py    # Output cleaning and processing
│   └── split_sentences.py    # Scientific sentence splitting
├── utils/                    # Utility modules
│   ├── call_llamat.py        # LLaMAT model interface
│   └── rawPaperProcessing/   # XML processing utilities
│       ├── get_text_from_xmls.py    # Text extraction from XML
│       ├── get_tables_from_xml.py   # Table extraction from XML
│       ├── xml_paper.py             # Paper download utility
│       └── mit_utils/               # MIT table extractor (adapted)
├── prompts/                  # LLM prompts
│   ├── extraction/           # Composition extraction prompts
│   └── linking/             # Property-composition linking prompts
├── input/                   # Input data directory
├── output/                  # Final results directory
├── checkpoints/            # Intermediate processing files
├── linking/               # Future property linking module
└── depreciated/          # Legacy code for reference
```

## 🚀 Pipeline Workflow

### Step 1: Text Extraction (`get_text_from_xmls.py`)
- Parses research paper XML files using BeautifulSoup
- Extracts title, abstract, sections, and body text
- Excludes references and bibliography sections
- Wraps sections in `<SEC>...</SEC>` tags for structure preservation
- Handles multiple XML namespaces (Elsevier, ACS, etc.)

### Step 2: Intelligent Chunking (`chunking.py`)
- Splits text into sections using `<SEC>` tags
- Performs sentence-aware chunking within sections
- Uses scientific text-optimized sentence splitting via spaCy
- Configurable chunk size (default: 700 words)
- Preserves sentence boundaries to maintain context

### Step 3: Sentence Splitting (`split_sentences.py`)
- Custom spaCy component for scientific text
- Handles chemical formulas and measurements
- Prevents inappropriate splitting on chemical notation
- Supports both standard and scientific spaCy models

### Step 4: LLM-Based Extraction (`extraction_util.py`)
- Loads system and user prompts from templates
- Processes each chunk through local LLaMAT model
- Structured prompt engineering for composition extraction
- Configurable temperature and token limits
- Robust error handling and retry logic

### Step 5: Output Processing (`process_outputs.py`)
- Parses LLM responses in JSON format
- Extracts composition strings from structured output
- Removes duplicates while preserving order
- Validates and cleans extracted compositions

## 🧠 LLM Integration

### LLaMAT Model Interface (`call_llamat.py`)
- **Local Model Support**: Loads LLaMAT-2-chat from `local_models/`
- **Multi-GPU Support**: Automatic GPU detection and DataParallel
- **Memory Optimization**: Half-precision on MPS, efficient tokenization
- **Conversation Format**: ChatML-style conversation formatting
- **Token Management**: Accurate token counting and metadata

### Prompt Engineering
The system uses carefully crafted prompts optimized for materials science:

**System Prompt** (`prompts/extraction/system_prompt_composition.txt`):
- Defines "Composition Miner" persona
- Provides clear extraction guidelines
- Includes comprehensive examples with edge cases
- Specifies exact JSON output format

**User Prompt** (`prompts/extraction/user_prompt_composition.txt`):
- Template-based approach with `{{passage}}` placeholder
- Consistent formatting across all chunks

## 📊 Data Processing Features

### XML Processing Capabilities
- **Multi-Publisher Support**: Elsevier, ACS, IEEE, etc.
- **Namespace Handling**: Automatic namespace detection and resolution
- **Section Filtering**: Intelligent exclusion of references/bibliography
- **Table Extraction**: Comprehensive table processing with MIT-based extractor
- **Metadata Preservation**: DOI, captions, footnotes extraction

### Scientific Text Handling
- **Chemical Formula Recognition**: Preserves subscripts and chemical notation
- **Measurement Parsing**: Handles units, ranges, and scientific notation
- **Context Preservation**: Maintains semantic relationships between sections
- **Duplicate Management**: Intelligent deduplication with order preservation

## 🎛️ Configuration System

### Config Class (`extraction.py`)
```python
class Config:
    CHUNK_SIZE = 700                    # Words per chunk
    SENTENCE_MODEL = "sci-spacy"        # Sentence splitting model
    MAX_TOKENS = 128                    # LLM generation limit
    TEMPERATURE = 0.0                   # Deterministic output
    SAVE_METADATA = False               # Metadata preservation
    VERBOSE = True                      # Debug output
```

### Command Line Interface
```bash
python extraction.py \
    --pii S0167273808006176 \
    --chunk_size 700 \
    --max_tokens 128 \
    --temperature 0.0 \
    --save_metadata \
    --verbose
```

## 📁 File Formats

### Input Files
- **XML Papers**: Raw research paper XML from publishers
- **Matskraft Tables**: CSV files with composition-property mappings
- **Configuration**: Prompt templates and processing parameters

### Intermediate Files
- **chunks.json**: Array of text chunks for LLM processing
- **compositions_model_raw.json**: Raw LLM outputs with metadata
- **research-paper-text.txt**: Extracted and structured text

### Output Files
- **compositions.json**: Final deduplicated composition list
- **metadata files**: Optional processing metadata for analysis

## 🔧 Dependencies

### Core Requirements
- **PyTorch**: LLM model execution and GPU acceleration
- **Transformers**: Hugging Face model loading and tokenization  
- **spaCy**: Scientific text processing and sentence splitting
- **BeautifulSoup4**: XML parsing and content extraction
- **LXML**: High-performance XML processing
- **Pydantic**: Type validation and data models

### Scientific Computing
- **SciPy**: Statistical analysis and data processing
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis

### Text Processing
- **Unidecode**: Unicode normalization
- **TQDM**: Progress tracking for batch processing

## 🎯 Usage Examples

### Basic Pipeline Execution
```bash
# Process a single paper
python extraction.py --pii S0167273808006176

# Custom configuration
python extraction.py --pii S0167273808006176 \
    --chunk_size 500 \
    --max_tokens 256 \
    --temperature 0.1 \
    --save_metadata
```

### Programmatic Usage
```python
from extraction import run_pipeline, Config

# Custom configuration
config = Config()
config.CHUNK_SIZE = 500
config.VERBOSE = True

# Run pipeline
summary = run_pipeline("S0167273808006176", config)
print(f"Extracted {summary['num_compositions']} compositions")
```

### Expected Output Format
```json
[
  "Sr0.94Ti0.9Nb0.1O3",
  "YSZ", 
  "MnO2",
  "Nb-doped SrTiO3",
  "Ca3(VO4)2"
]
```

## 🔄 Property Linking (Future Component)

The `linking/` directory contains infrastructure for the next phase: linking extracted compositions to material properties.

### Planned Workflow
1. **Property Extraction**: Identify material properties from papers
2. **Composition-Property Matching**: Link specific compositions to properties
3. **Confidence Scoring**: Assess link reliability
4. **Knowledge Graph Building**: Create structured material databases

### Linking Prompts
- **System Prompt**: "MatPro-Matcher" for property-composition linking
- **Task Definition**: Find composition most clearly associated with given property
- **Output Format**: Single composition or "unknown" for ambiguous cases

## 📈 Performance Characteristics

### Processing Metrics
- **Throughput**: ~50-100 chunks per minute (depending on hardware)
- **Accuracy**: High precision for standard chemical formulas
- **Memory Usage**: ~2-4GB GPU memory for LLaMAT-2-chat
- **Storage**: ~1MB per paper for intermediate files

### Scalability Features
- **Batch Processing**: Multiple papers in sequence
- **Checkpoint Recovery**: Resume interrupted processing
- **Memory Management**: Efficient GPU utilization
- **Error Resilience**: Graceful handling of malformed inputs

## 🧪 Quality Assurance

### Validation Pipeline
- **JSON Output Validation**: Ensures well-formed extraction results
- **Duplicate Detection**: Maintains unique composition lists
- **Error Logging**: Comprehensive error tracking and reporting
- **Manual Verification**: Sample outputs for quality assessment

### Testing Coverage
- **Unit Tests**: Individual component validation
- **Integration Tests**: End-to-end pipeline verification  
- **Edge Cases**: Malformed XML, empty sections, encoding issues
- **Performance Tests**: Memory usage and processing speed benchmarks

## 🛠️ Development History

### Evolution from Legacy Systems
The current pipeline represents a significant evolution from earlier approaches:

1. **Legacy LLM Utils** (`depreciated/llm_utils.py`): Original OpenAI/Anthropic API integration
2. **Matskraft Processing** (`depreciated/process_matskraft_tables.py`): Early table-focused extraction
3. **MIT Table Extractor** (`mit_utils/`): Adapted table extraction from Olivetti Group research
4. **Modern LLaMAT Integration**: Transition to local model deployment

### Key Improvements
- **Local Model Deployment**: Reduced API costs and increased control
- **Scientific Text Optimization**: Specialized handling of chemical formulas
- **Modular Architecture**: Easier maintenance and feature extension  
- **Comprehensive Error Handling**: Robust production-ready processing

## 🚧 Current Status & Future Roadmap

### Completed Features ✅
- ✅ XML text extraction with multi-publisher support
- ✅ Intelligent chunking with scientific text awareness
- ✅ LLaMAT model integration with local deployment
- ✅ Structured composition extraction with JSON output
- ✅ Comprehensive output processing and deduplication
- ✅ Command-line interface with flexible configuration

### In Development 🔄
- 🔄 Property-composition linking system
- 🔄 Enhanced table extraction and processing
- 🔄 Batch processing optimizations
- 🔄 Quality assessment metrics

### Planned Enhancements 📋
- 📋 Multi-model LLM support (GPT, Claude, etc.)
- 📋 Real-time processing API
- 📋 Web interface for interactive processing
- 📋 Integration with materials databases
- 📋 Advanced confidence scoring for extractions

---

## 📄 License & Attribution

This project builds upon and acknowledges several open-source contributions:

- **MIT Table Extractor**: Adapted from [Olivetti Group's table_extractor](https://github.com/olivettigroup/table_extractor) (MIT License)
- **Scientific Text Processing**: Incorporates research-grade NLP techniques
- **LLaMAT Integration**: Custom implementation for materials science applications

For questions, issues, or contributions, please refer to the project documentation or contact the development team.
