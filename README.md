# PDF Extractor

## Overview

PDF Extractor is a Python tool that processes PDF files to extract text and images. It supports both native text extraction and OCR (Optical Character Recognition) with multi-language support, including Hungarian.

### Key Features

- **Batch Processing**: Process individual PDF files or entire directories of PDFs
- **Dual Text Extraction**: 
  - Native extraction (faster, but may have formatting issues)
  - OCR-based extraction (better for scanned documents, supports multiple languages)
- **Image Extraction**: Extracts all images from PDFs
- **Organized Output**: 
  - Creates a separate directory for each PDF
  - Saves text files with clear page markers and image references
  - Saves all extracted images alongside text
- **Hungarian Language Support**: Special support for Hungarian accented characters via Tesseract OCR
- **AI-Enhanced Text**: Optional AI processing to combine and improve extracted text using OpenAI API
- **MCP Upload**: Optional upload functionality via MCP protocol

## Installation

### Prerequisites

- Python 3.6+
- Tesseract OCR (for OCR functionality)
- OpenAI API key (optional, for AI-enhanced text processing)

### Setup

1. **Install Python dependencies**:

```bash
pip install -r requirements.txt
```

2. **Install Tesseract OCR**:

   - **Windows**: Download from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt install tesseract-ocr`

3. **Install language data for Tesseract** (for Hungarian support):

   - **Windows**: Download `hun.traineddata` from [https://github.com/tesseract-ocr/tessdata](https://github.com/tesseract-ocr/tessdata) and place it in the Tesseract `tessdata` directory
   - **macOS**: `brew install tesseract-lang`
   - **Linux**: `sudo apt install tesseract-ocr-hun`

4. **Set up OpenAI API key** (optional, for AI-enhanced text):

   - Create a `.env` file based on the provided `.env.example`
   - Add your OpenAI API key to the `.env` file
   - Alternatively, set the `OPENAI_API_KEY` environment variable or pass it via command line

## Usage

### Command Line Options

```bash
python pdf_uploader.py <pdf_path> [options]
```

Where `<pdf_path>` can be either a single PDF file or a directory containing PDFs.

#### Options:

- `--export`: Export text and images locally (default mode is upload)
- `--output_dir <dir>`: Base directory for exports (default: './exports')
- `--sse_url <url>`: SSE negotiation URL for MCP upload
- `--force-ocr`: Force OCR processing and create exported-ocr.txt
- `--lang <code>`: OCR language (e.g., eng, hun). Default is English
- `--openai-api-key <key>`: OpenAI API key for AI-friendly version creation
- `--openai-model <model>`: OpenAI model to use (default: o4-mini)

### Examples

#### Export a single PDF with native text extraction:

```bash
python pdf_uploader.py document.pdf --export --output_dir ./my_exports
```

#### Export a single PDF with OCR (English):

```bash
python pdf_uploader.py document.pdf --export --force-ocr
```

#### Export a single PDF with Hungarian OCR:

```bash
python pdf_uploader.py document.pdf --export --force-ocr --lang hun
```

#### Export a single PDF with AI-enhanced text processing:

```bash
python pdf_uploader.py document.pdf --export --force-ocr --openai-api-key YOUR_API_KEY
```

#### Export a single PDF with a specific OpenAI model:

```bash
python pdf_uploader.py document.pdf --export --force-ocr --openai-api-key YOUR_API_KEY --openai-model gpt-4
```

#### Process all PDFs in a directory with Hungarian OCR and AI enhancement:

```bash
python pdf_uploader.py ./my_pdfs/ --export --force-ocr --lang hun --openai-api-key YOUR_API_KEY
```

#### Upload a PDF via MCP:

```bash
python pdf_uploader.py document.pdf --sse_url https://example.com/sse
```

## Environment Variables

You can set the following environment variables in a `.env` file:

```
# OpenAI API Key for AI-friendly text processing
OPENAI_API_KEY=your_openai_api_key_here

# OpenAI model to use for AI processing
OPENAI_MODEL=o4-mini

# OCR Language (default is English)
OCR_LANGUAGE=eng  # or hun for Hungarian
```

## Output Structure

For each PDF processed with `--export`, the tool creates:

```
<output_dir>/
  └── <pdf_name>/
      ├── exported.txt         # Text from native extraction
      ├── exported-ocr.txt     # Text from OCR (if --force-ocr is used)
      ├── exported-ai.txt      # AI-enhanced text (if OpenAI API key is provided)
      └── <pdf_name>_page<N>_img<M>.<ext>  # Extracted images
```

### Text File Format

The text files include:
- Page markers (`-- Page X --`)
- Formatted text with preserved paragraphs
- Image references with clear markers

## VS Code Integration

The repository includes VS Code tasks for common operations:

- **Export PDFs**: Standard export with native text extraction
- **Export PDFs with OCR**: Export with OCR using English language
- **Export PDFs with Hungarian OCR**: Export with OCR optimized for Hungarian text
- **Export PDFs with AI Processing**: Export with OCR and AI-enhanced text processing
- **Upload PDFs**: Upload to MCP with native text extraction
- **Upload PDFs with OCR**: Upload to MCP with OCR text extraction

## Troubleshooting

### OCR Issues

- If OCR fails, the tool will automatically fall back to native extraction
- For best results with Hungarian text, ensure the Hungarian language data is properly installed
- Verify Tesseract installation with: `tesseract --list-langs` (should show 'hun' in the list)

### AI Processing Issues

- If the OpenAI package is not installed, AI processing will be skipped
- If the API key is invalid or missing, AI processing will be skipped
- If API calls fail, the tool will retry with exponential backoff
- If all retries fail, it will fall back to using the native text

### Image Extraction

- Some PDFs may contain images in formats that cannot be extracted
- The tool extracts all images it can identify and saves them in their original format

## License

MIT
