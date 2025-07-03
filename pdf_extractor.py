import os
import fitz  # PyMuPDF
from PIL import Image
import argparse
import io
import time
import re
from dotenv import load_dotenv

# Local module imports
from ai_processor import AIProcessor
from ocr_processor import OCRProcessor
from sse_uploader import SSEUploader

class PDFExtractor:
    """
    A class to handle PDF processing, including text extraction (native and OCR),
    image extraction, and uploading the results.
    """
    def __init__(self, file_path, upload_url=None, export=False, output_dir=None, force_ocr=False, use_ocr=False, lang='eng', openai_api_key=None, openai_model=None):
        """
        Initialize the PDFExtractor with file paths and options.
        
        Args:
            file_path: Path to the PDF file or directory of PDF files.
            upload_url: URL to upload the extracted data to.
            export: If True, export data to local files instead of uploading.
            output_dir: Directory to save exported files.
            force_ocr: If True, use OCR even if native text is available and will regenerate OCR results.
            use_ocr: If True, use OCR but only if no previous OCR results exist.
            lang: Language for Tesseract OCR.
            openai_api_key: OpenAI API key for AI-based text cleaning.
            openai_model: OpenAI model to use for AI-based text cleaning.
        """
        self.file_path = file_path
        self.upload_url = upload_url
        self.export = export
        self.output_dir = output_dir or os.getcwd()
        self.force_ocr = force_ocr
        self.use_ocr = use_ocr
        
        # Initialize processors
        self.ocr_processor = OCRProcessor(lang=lang)
        self.ai_processor = AIProcessor(api_key=openai_api_key, model=openai_model)
        self.sse_uploader = SSEUploader(url=upload_url)

    def _collect_pdfs(self, path):
        if os.path.isfile(path) and path.lower().endswith('.pdf'):
            return [path]
        collected = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith('.pdf'):
                    collected.append(os.path.join(root, f))
        return collected

    def _pdf_dir(self, pdf_file):
        base = os.path.splitext(os.path.basename(pdf_file))[0]
        dirpath = os.path.join(self.output_dir, base)
        os.makedirs(dirpath, exist_ok=True)
        return dirpath

    def extract_text(self, pdf_file, use_ocr=False):
        """
        Extract text from PDF using either native extraction or OCR.
        Args:
            pdf_file: Path to the PDF file.
            use_ocr: If True, use OCR for text extraction. If False, use native extraction.
        Returns:
            List of text content for each page.
        """
        doc = fitz.open(pdf_file)
        total_pages = len(doc)
        pages_text = []
        
        for p in range(total_pages):
            page = doc.load_page(p)
            
            if use_ocr:
                pix = page.get_pixmap(dpi=300)
                text = self.ocr_processor.extract_text_from_image(pix, p + 1, total_pages)
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

    def extract_images(self, pdf_file):
        doc = fitz.open(pdf_file)
        base = os.path.splitext(os.path.basename(pdf_file))[0]
        images_by_page = {}
        out_dir = self._pdf_dir(pdf_file) if self.export else os.getcwd()
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

    def export_data(self, pdf_file):
        """
        Export text (preserving paragraphs) and images to local directory.
        Wrap each original line for readability without collapsing words.
        Creates exported.txt with native extraction and exported-ocr.txt with OCR if forced.
        If OpenAI API key is provided, also creates an AI-friendly version.
        """
        out_dir = self._pdf_dir(pdf_file)
        
        # Extract images first
        images = self.extract_images(pdf_file)
        
        # Always create native text extraction
        print("\nExtracting text using native extraction:")
        native_txt_pages = self.extract_text(pdf_file, use_ocr=False)
        self._write_text_file(native_txt_pages, images, out_dir, "exported.txt")
        print(f"Created native text file: {os.path.join(out_dir, 'exported.txt')}")
        
        # Create or load OCR text file if needed
        ocr_txt_pages = None
        ocr_file = os.path.join(out_dir, "exported-ocr.txt")
        
        # OCR processing logic:
        # 1. force_ocr: Always run OCR, even if file exists
        # 2. use_ocr & file doesn't exist: Run OCR to create the file
        # 3. use_ocr & file exists: Use existing file without running OCR
        # 4. Neither force_ocr nor use_ocr: No OCR processing
        
        # Case 3: File exists and we're not forcing re-generation
        if not self.force_ocr and os.path.exists(ocr_file):
            print(f"\nFound existing OCR file, skipping OCR step: {ocr_file}")
            with open(ocr_file, 'r', encoding='utf-8') as f:
                # Load the OCR text from the existing file
                ocr_content = f.read()
                # Split by page markers to get the individual pages
                page_splits = re.split(r'--- Page \d+ ---', ocr_content)[1:] # Skip the first empty split
                ocr_txt_pages = [page.strip() for page in page_splits]
        # Case 1 & 2: Run OCR if forced or if needed and we want OCR
        elif self.force_ocr or self.use_ocr:
            print("\nExtracting text using OCR:")
            ocr_txt_pages = self.extract_text(pdf_file, use_ocr=True)
            self._write_text_file(ocr_txt_pages, images, out_dir, "exported-ocr.txt")
            print(f"Created OCR text file: {ocr_file}")
        
        # Create AI-friendly version if API key is provided and OCR results are available
        if self.ai_processor.is_available() and ocr_txt_pages:
            print("\nCreating AI-friendly version using ChatGPT API...")
            self.ai_processor.create_ai_friendly_version(native_txt_pages, ocr_txt_pages, images, out_dir)
            
        print(f"\nExported PDF '{pdf_file}' to directory '{out_dir}'")
        
    def _write_text_file(self, pages_text, images, out_dir, filename):
        """
        Write the extracted text to a file, including image references.
        """
        full_text = []
        for i, text in enumerate(pages_text):
            full_text.append(f"--- Page {i+1} ---")
            full_text.append(text)
            
            # Add image references for this page
            if i+1 in images and images[i+1]:
                full_text.append(f"\n--- Images on page {i+1} ---")
                for img in images[i+1]:
                    full_text.append(f"[IMAGE: {img}]")
                full_text.append("--- End of images ---")
        
        with open(os.path.join(out_dir, filename), 'w', encoding='utf-8') as f:
            f.write("\n".join(full_text))

    def upload_data(self, pdf_file):
        """
        Extract data and upload it to the SSE endpoint.
        """
        if not self.sse_uploader.url:
            print("No upload URL provided. Use --export to save locally.")
            return

        if not self.sse_uploader.post_url:
            self.sse_uploader.negotiate_endpoint()

        text_pages = self.extract_text(pdf_file, use_ocr=self.force_ocr)
        images = self.extract_images(pdf_file)

        data = {
            "text_pages": text_pages,
            "images": images
        }

        self.sse_uploader.upload(data)

    def run(self):
        """
        Run the PDF processing based on the provided options.
        """
        pdf_files = self._collect_pdfs(self.file_path)
        if not pdf_files:
            print(f"No PDF files found in '{self.file_path}'")
            return

        for pdf_file in pdf_files:
            print(f"\nProcessing '{pdf_file}' (export={self.export})")
            if self.export:
                self.export_data(pdf_file)
            else:
                self.upload_data(pdf_file)

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description='PDF Text and Image Extractor')
    parser.add_argument('file_path', help='Path to a PDF file or a directory containing PDF files')
    parser.add_argument('--upload_url', help='URL to upload extracted data')
    parser.add_argument('--export', action='store_true', help='Export data to local files instead of uploading')
    parser.add_argument('--output_dir', help='Directory to save exported files')
    parser.add_argument('--force-ocr', action='store_true', help='Force OCR extraction even if native text is available and regenerate if OCR file exists')
    parser.add_argument('--use-ocr', action='store_true', help='Use OCR if needed, but only run OCR if no previous results exist')
    parser.add_argument('--lang', default=os.environ.get('OCR_LANGUAGE', 'eng'), help='Language for Tesseract OCR (e.g., eng, hun). Can also be set with OCR_LANGUAGE environment variable.')
    parser.add_argument('--openai-api-key', default=os.environ.get('OPENAI_API_KEY'), help='OpenAI API key for AI-based text cleaning. Can also be set with OPENAI_API_KEY environment variable.')
    parser.add_argument('--openai-model', default=os.environ.get('OPENAI_MODEL'), help='OpenAI model to use. Can also be set with OPENAI_MODEL environment variable.')

    args = parser.parse_args()

    processor = PDFExtractor(
        file_path=args.file_path,
        upload_url=args.upload_url,
        export=args.export,
        output_dir=args.output_dir,
        force_ocr=args.force_ocr,
        use_ocr=args.use_ocr,
        lang=args.lang,
        openai_api_key=args.openai_api_key,
        openai_model=args.openai_model
    )
    print("\n--- Configuration ---")
    print(f"File/Directory: {processor.file_path}")
    print(f"Mode: {'Export' if processor.export else 'Upload'}")
    if processor.export:
        print(f"Output Directory: {processor.output_dir}")
    if processor.upload_url:
        print(f"Upload URL: {processor.upload_url}")
    print(f"Force OCR: {processor.force_ocr}")
    print(f"OCR Language: {processor.ocr_processor.lang}")
    if processor.ai_processor.is_available():
        print(f"AI Post-processing: Enabled")
        print(f"OpenAI Model: {processor.ai_processor.model}")
    else:
        print(f"AI Post-processing: Disabled")
    print("---------------------\n")

    processor.run()