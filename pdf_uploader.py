import fitz
import requests
import os
import json
import base64
from urllib.parse import urljoin
from PIL import Image
import pytesseract
import io
import textwrap
import re
import time

# Optional imports - don't fail if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    pass  # dotenv is optional

# Default OpenAI model to use
DEFAULT_OPENAI_MODEL = "o4-mini"

class PDFUploader:
    """
    Processes PDFs: extracts text via OCR and images, supports MCP upload or local export.
    - Auto-detects directories and processes all .pdf files
    - --export mode: each PDF gets its own folder under output_dir
      with extracted.txt (including [IMAGE: filename] placeholders) and image files
    - upload mode: negotiates SSE, sends JSON-RPC calls for text/images
    """

    def __init__(self, pdf_path, sse_url=None, export=False, output_dir=None, force_ocr=False, lang='eng', openai_api_key=None, openai_model=None):
        self.pdf_path = pdf_path
        self.sse_url = sse_url
        self.export = export
        self.output_dir = output_dir or os.getcwd()
        self.post_url = None
        self.force_ocr = force_ocr
        self.lang = lang  # OCR language, e.g., 'eng' for English, 'hun' for Hungarian
        self.openai_api_key = openai_api_key
        # Get model from environment variable if not provided as argument
        self.openai_model = openai_model or os.environ.get('OPENAI_MODEL') or DEFAULT_OPENAI_MODEL

    def negotiate_endpoint(self):
        resp = requests.get(self.sse_url, stream=True)
        resp.raise_for_status()
        buffer = []
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if line.startswith("data:"):
                buffer.append(line[len("data:"):].strip())
            elif line == "":
                payload = "".join(buffer)
                buffer.clear()
                try:
                    data = json.loads(payload)
                    endpoint = data.get("postEndpointUri")
                except json.JSONDecodeError:
                    endpoint = payload.strip('"')
                if endpoint:
                    self.post_url = urljoin(self.sse_url, endpoint)
                    print(f"Negotiated postEndpointUri: {self.post_url}")
                    return
        raise RuntimeError("No postEndpointUri received via SSE")

    def _collect_pdfs(self, path):
        if os.path.isfile(path) and path.lower().endswith('.pdf'):
            return [path]
        collected = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith('.pdf'):
                    collected.append(os.path.join(root, f))
        return collected

    def _pdf_dir(self):
        base = os.path.splitext(os.path.basename(self.pdf_path))[0]
        dirpath = os.path.join(self.output_dir, base)
        os.makedirs(dirpath, exist_ok=True)
        return dirpath

    def extract_text(self, use_ocr=False):
        """
        Extract text from PDF using either native extraction or OCR.
        Args:
            use_ocr: If True, use OCR for text extraction. If False, use native extraction.
        Returns:
            List of text content for each page.
        """
        doc = fitz.open(self.pdf_path)
        total_pages = len(doc)
        pages_text = []
        
        for p in range(total_pages):
            page = doc.load_page(p)
            
            if use_ocr:
                try:
                    # Use OCR for text extraction with specified language
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes()))
                    text = pytesseract.image_to_string(img, lang=self.lang)
                    print(f"Page {p+1}/{total_pages}: OCR successful using language '{self.lang}'")
                except Exception as e:
                    print(f"Page {p+1}/{total_pages}: OCR failed ({e}), using native extraction")
                    text = page.get_text()
            else:
                # Use native extraction
                if p % 10 == 0 or p == total_pages - 1:  # Show progress every 10 pages or on the last page
                    print(f"Processing page {p+1}/{total_pages} using native extraction")
                text = page.get_text()
            
            # Improve text formatting
            text = self._improve_text_formatting(text)
            pages_text.append(text)
            
        return pages_text
    
    def _improve_text_formatting(self, text):
        """
        Improve text formatting by fixing common OCR and PDF extraction issues:
        - Remove excessive spaces
        - Fix broken words (hyphenation at line breaks)
        - Preserve paragraph structure
        """
        # Fix hyphenated words broken across lines
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Remove excessive spaces
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Preserve paragraph breaks but remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text

    def extract_images(self):
        doc = fitz.open(self.pdf_path)
        base = os.path.splitext(os.path.basename(self.pdf_path))[0]
        images_by_page = {}
        out_dir = self._pdf_dir() if self.export else os.getcwd()
        for pnum, page in enumerate(doc, start=1):
            images_by_page[pnum] = []
            for idx, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                img_data = doc.extract_image(xref)
                data = img_data['image']
                ext = img_data['ext']
                fname = f"{base}_page{pnum}_img{idx}.{ext}"
                images_by_page[pnum].append(fname)
                with open(os.path.join(out_dir, fname), 'wb') as f:
                    f.write(data)
        return images_by_page

    def export_data(self):
        """
        Export text (preserving paragraphs) and images to local directory.
        Wrap each original line for readability without collapsing words.
        Creates exported.txt with native extraction and exported-ocr.txt with OCR if forced.
        If OpenAI API key is provided, also creates an AI-friendly version.
        """
        out_dir = self._pdf_dir()
        
        # Extract images first
        images = self.extract_images()
        
        # Always create native text extraction
        print("\nExtracting text using native extraction:")
        native_txt_pages = self.extract_text(use_ocr=False)
        self._write_text_file(native_txt_pages, images, out_dir, "exported.txt")
        print(f"Created native text file: {os.path.join(out_dir, 'exported.txt')}")
        
        # Create OCR text file if forced
        ocr_txt_pages = None
        if self.force_ocr:
            print("\nExtracting text using OCR:")
            ocr_txt_pages = self.extract_text(use_ocr=True)
            self._write_text_file(ocr_txt_pages, images, out_dir, "exported-ocr.txt")
            print(f"Created OCR text file: {os.path.join(out_dir, 'exported-ocr.txt')}")
        
        # Create AI-friendly version if API key is provided
        if self.openai_api_key and self.force_ocr and ocr_txt_pages:
            print("\nCreating AI-friendly version using ChatGPT API...")
            self._create_ai_friendly_version(native_txt_pages, ocr_txt_pages, images, out_dir)
            
        print(f"\nExported PDF '{self.pdf_path}' to directory '{out_dir}'")
        
    def _create_ai_friendly_version(self, native_txt_pages, ocr_txt_pages, images, out_dir):
        """
        Create an AI-friendly version of the extracted text using OpenAI API.
        This combines the native and OCR text to create a more accurate and structured version.
        If OpenAI is not available or no API key is provided, this will be skipped.
        """
        # Check if OpenAI is available
        if not globals().get('OPENAI_AVAILABLE', False):
            print("OpenAI package not installed. Skipping AI-friendly version creation.")
            print("To enable this feature, install the openai package: pip install openai")
            return
        
        # Get API key from various sources
        api_key = self.openai_api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("OpenAI API key not found. Skipping AI-friendly version creation.")
            print("To enable this feature, provide an API key via:")
            print("  - Command line: --openai-api-key YOUR_KEY")
            print("  - Environment variable: OPENAI_API_KEY=YOUR_KEY")
            print("  - .env file: OPENAI_API_KEY=YOUR_KEY")
            return
            
        # Determine if the model is an Anthropic (Claude) model
        is_claude_model = self.openai_model.startswith('claude-') or self.openai_model.startswith('anthropic/')
        # Determine if the model is an OpenAI o-series model
        is_o_series = self.openai_model.startswith('o') and not self.openai_model.startswith('openai/')
        # Determine if the model is a GPT model
        is_gpt_model = 'gpt' in self.openai_model.lower()
        
        try:
            print(f"Creating AI-friendly version using OpenAI API with model '{self.openai_model}'...")
            # Create OpenAI client (compatible with openai>=1.0.0)
            client = openai.OpenAI(api_key=api_key)
            
            # Print model type information for debugging
            model_types = []
            if is_o_series:
                model_types.append("OpenAI o-series")
                print(f"Using o-series model parameters (max_completion_tokens)")
            elif is_gpt_model:
                model_types.append("GPT")
                print(f"Using standard model parameters (max_tokens)")
            elif is_claude_model:
                model_types.append("Claude")
                print(f"Using standard model parameters (max_tokens)")
            else:
                print(f"Using standard model parameters (max_tokens)")
                
            if model_types:
                print(f"Detected model type: {', '.join(model_types)}")
            else:
                print("Using default model parameters")
            
            # Prepare combined text with image references
            total_pages = min(len(native_txt_pages), len(ocr_txt_pages))
            all_native_text = []
            all_ocr_text = []
            all_image_refs = []
            
            # Collect all text and image references
            for i in range(total_pages):
                all_native_text.append(f"--- Page {i+1} ---\n{native_txt_pages[i]}")
                all_ocr_text.append(f"--- Page {i+1} ---\n{ocr_txt_pages[i]}")
                
                # Get image references for this page
                if i+1 in images and images[i+1]:
                    page_images = []
                    for img in images[i+1]:
                        page_images.append(f"[IMAGE: {img}]")
                    all_image_refs.append(f"--- Images on page {i+1} ---\n" + "\n".join(page_images))
            
            # Combine all text
            full_native_text = "\n\n".join(all_native_text)
            full_ocr_text = "\n\n".join(all_ocr_text)
            
            # Estimate token count (rough approximation: 4 chars = 1 token)
            estimated_tokens = (len(full_native_text) + len(full_ocr_text)) // 4
            max_input_tokens = 16000  # Conservative limit for most models
            
            # Determine if we need to process in chunks
            if estimated_tokens > max_input_tokens:
                print(f"Text is too large for a single API call (est. {estimated_tokens} tokens). Processing in chunks...")
                # Process in chunks of approximately 3 pages at a time
                chunk_size = max(1, total_pages // (estimated_tokens // max_input_tokens + 1))
                return self._process_in_chunks(client, native_txt_pages, ocr_txt_pages, images, out_dir, chunk_size, is_o_series, is_gpt_model, is_claude_model)
            else:
                print(f"Processing entire document in a single API call (est. {estimated_tokens} tokens)")
                # Create prompt for the entire document
                prompt = f"""I have extracted text from a PDF document using two methods: native extraction and OCR.
                Please create a clean, well-formatted version that combines the best of both extractions.
                Preserve paragraph structure, fix any OCR errors, and ensure the text is coherent and readable.
                Maintain page markers and image references in your output.
                
                NATIVE EXTRACTION:\n{full_native_text}\n\nOCR EXTRACTION:\n{full_ocr_text}"""
                
                # Call OpenAI API with retry logic
                max_retries = 3
                cleaned_text = ""
                for attempt in range(max_retries):
                    try:
                        # Prepare API call parameters based on model type
                        params = {
                            "model": self.openai_model,
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant that cleans and formats text extracted from PDFs."},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.3
                        }
                        
                        # Add the appropriate token parameter based on model type
                        if is_o_series:
                            params["max_completion_tokens"] = 8000
                        else:
                            params["max_tokens"] = 8000
                            
                        # Make the API call with the appropriate parameters
                        response = client.chat.completions.create(**params)
                        
                        # Extract the cleaned text
                        cleaned_text = response.choices[0].message.content
                        print("Successfully processed entire document with AI")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # Exponential backoff
                            print(f"API call failed: {e}. Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            print(f"Failed to process document after {max_retries} attempts: {e}")
                            # Fall back to native text with page markers
                            cleaned_text = full_native_text
                
                # Add image references
                if all_image_refs:
                    cleaned_text += "\n\n" + "\n\n".join(all_image_refs)
                
                # Write the AI-friendly version to file
                ai_file = os.path.join(out_dir, "exported-ai.txt")
                with open(ai_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                
                print(f"Created AI-friendly text file: {ai_file}")
                return
            
        except Exception as e:
            print(f"Error creating AI-friendly version: {e}")
            
            # Attempt to fallback to a different model if there's an API error
            if "model" in str(e).lower() or "parameter" in str(e).lower():
                fallback_model = "gpt-3.5-turbo" if self.openai_model != "gpt-3.5-turbo" else "gpt-4"
                print(f"\nAttempting fallback to {fallback_model}...")
                try:
                    # Save original model name
                    original_model = self.openai_model
                    # Set fallback model
                    self.openai_model = fallback_model
                    # Try again with fallback model
                    self._create_ai_friendly_version(native_txt_pages, ocr_txt_pages, images, out_dir)
                    print(f"Successfully processed with fallback model {fallback_model}")
                    return
                except Exception as fallback_e:
                    print(f"Fallback to {fallback_model} also failed: {fallback_e}")
                    # Restore original model name
                    self.openai_model = original_model
    
    def _process_in_chunks(self, client, native_txt_pages, ocr_txt_pages, images, out_dir, chunk_size, is_o_series, is_gpt_model, is_claude_model):
        """
        Process the document in chunks when it's too large for a single API call.
        """
        total_pages = min(len(native_txt_pages), len(ocr_txt_pages))
        chunks = []
        
        # Process document in chunks
        for chunk_start in range(0, total_pages, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_pages)
            print(f"Processing chunk: pages {chunk_start+1}-{chunk_end} of {total_pages}")
            
            # Prepare text for this chunk
            chunk_native = []
            chunk_ocr = []
            for i in range(chunk_start, chunk_end):
                chunk_native.append(f"--- Page {i+1} ---\n{native_txt_pages[i]}")
                chunk_ocr.append(f"--- Page {i+1} ---\n{ocr_txt_pages[i]}")
            
            full_native_chunk = "\n\n".join(chunk_native)
            full_ocr_chunk = "\n\n".join(chunk_ocr)
            
            # Create prompt for this chunk
            prompt = f"""I have extracted text from pages {chunk_start+1}-{chunk_end} of a PDF using two methods: native extraction and OCR.
            Please create a clean, well-formatted version that combines the best of both extractions.
            Preserve paragraph structure, fix any OCR errors, and ensure the text is coherent and readable.
            Maintain page markers in your output.
            
            NATIVE EXTRACTION:\n{full_native_chunk}\n\nOCR EXTRACTION:\n{full_ocr_chunk}"""
            
            # Call OpenAI API with retry logic
            max_retries = 3
            chunk_text = ""
            for attempt in range(max_retries):
                try:
                    # Prepare API call parameters based on model type
                    params = {
                        "model": self.openai_model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that cleans and formats text extracted from PDFs."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3
                    }
                    
                    # Add the appropriate token parameter based on model type
                    if is_o_series:
                        params["max_completion_tokens"] = 4000
                    else:
                        params["max_tokens"] = 4000
                        
                    # Make the API call with the appropriate parameters
                    response = client.chat.completions.create(**params)
                    
                    # Extract the cleaned text
                    chunk_text = response.choices[0].message.content
                    print(f"Successfully processed chunk {chunk_start+1}-{chunk_end}")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"API call failed: {e}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to process chunk {chunk_start+1}-{chunk_end} after {max_retries} attempts: {e}")
                        # Fall back to native text
                        chunk_text = full_native_chunk
            
            chunks.append(chunk_text)
        
        # Combine all chunks
        combined_text = "\n\n".join(chunks)
        
        # Add image references
        image_refs = []
        for page_num, page_images in images.items():
            if page_images:
                refs = [f"[IMAGE: {img}]" for img in page_images]
                image_refs.append(f"--- Images on page {page_num} ---\n" + "\n".join(refs) + "\n--- End of images ---")
        
        if image_refs:
            combined_text += "\n\n" + "\n\n".join(image_refs)
        
        # Write the AI-friendly version to file
        ai_file = os.path.join(out_dir, "exported-ai.txt")
        with open(ai_file, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        print(f"Created AI-friendly text file: {ai_file}")
        return
    
    def _write_text_file(self, txt_pages, images, out_dir, filename):
        """
        Write text content to file with page markers and image placeholders.
        Args:
            txt_pages: List of text content for each page
            images: Dictionary of image filenames by page number
            out_dir: Directory to write the file to
            filename: Name of the output file
        """
        out_file = os.path.join(out_dir, filename)
        
        lines = []
        for i, txt in enumerate(txt_pages, start=1):
            lines.append(f"-- Page {i} --")
            last_blank = False
            started = False
            
            for orig_line in txt.split('\n'):
                stripped = orig_line.strip()
                if stripped:
                    started = True
                    wrapped = textwrap.fill(stripped, width=120)
                    for lw in wrapped.split('\n'):
                        lines.append(lw)
                    last_blank = False
                else:
                    if started and not last_blank:
                        lines.append('')
                        last_blank = True
            
            # Add image placeholders for this page
            if i in images and images[i]:
                lines.append('')
                lines.append("--- Images on this page ---")
                for img in images.get(i, []):
                    lines.append(f"[IMAGE: {img}]")
                lines.append("--- End of images ---")
            
            lines.append('')
            
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"Created text file: {out_file}")

    def upload(self):
        self.negotiate_endpoint()
        # Use OCR if forced, otherwise use native extraction
        txt_pages = self.extract_text(use_ocr=self.force_ocr)
        self.upload_text(txt_pages)
        images_by_page = self.extract_images()
        self.upload_images(images_by_page)

    def upload_text(self, txt_pages):
        payload = {"jsonrpc":"2.0","method":"uploadText",
                   "params":{"text":"\n".join(txt_pages)},"id":1}
        resp = requests.post(self.post_url, json=payload)
        resp.raise_for_status()
        print(f"Text upload status: {resp.status_code}")

    def upload_images(self, images_by_page):
        idx = 2
        for pnum, flist in images_by_page.items():
            for fname in flist:
                with open(fname, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('ascii')
                payload = {"jsonrpc":"2.0","method":"uploadImage",
                           "params":{"filename":fname,"data":b64},"id":idx}
                resp = requests.post(self.post_url, json=payload)
                resp.raise_for_status()
                print(f"Image '{fname}' upload status: {resp.status_code}")
                idx += 1

    def run(self):
        paths = self._collect_pdfs(self.pdf_path)
        for pdf in paths:
            self.pdf_path = pdf
            print(f"Processing '{pdf}' (export={self.export})")
            if self.export:
                self.export_data()
            else:
                self.upload()

if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Batch PDF processing: export or MCP upload')
    parser.add_argument('pdf_path', help='PDF file or directory path')
    parser.add_argument('--export', action='store_true', help='Export text/images locally')
    parser.add_argument('--output_dir', default='./exports', help='Base dir for exports')
    parser.add_argument('--sse_url', help='SSE negotiation URL for MCP upload')
    parser.add_argument('--force-ocr', action='store_true', help='Force OCR processing and create exported-ocr.txt')
    parser.add_argument('--lang', default='eng', help='OCR language (e.g., eng, hun). Default is English.')
    parser.add_argument('--openai-api-key', help='OpenAI API key for AI-friendly version creation')
    parser.add_argument('--openai-model', default=DEFAULT_OPENAI_MODEL, help=f'OpenAI model to use for AI processing (default: {DEFAULT_OPENAI_MODEL})')
    args = parser.parse_args()
    
    # Check for API key in environment variable if not provided as argument
    openai_api_key = args.openai_api_key or os.environ.get('OPENAI_API_KEY')

    uploader = PDFUploader(
        pdf_path=args.pdf_path,
        sse_url=args.sse_url,
        export=args.export,
        output_dir=args.output_dir,
        force_ocr=args.force_ocr,
        lang=args.lang,
        openai_api_key=openai_api_key,
        openai_model=args.openai_model
    )
    uploader.run()
