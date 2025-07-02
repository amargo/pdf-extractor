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

class PDFUploader:
    """
    Processes PDFs: extracts text via OCR and images, supports MCP upload or local export.
    - Auto-detects directories and processes all .pdf files
    - --export mode: each PDF gets its own folder under output_dir
      with extracted.txt (including [IMAGE: filename] placeholders) and image files
    - upload mode: negotiates SSE, sends JSON-RPC calls for text/images
    """

    def __init__(self, pdf_path, sse_url=None, export=False, output_dir=None, force_ocr=False, lang='eng'):
        self.pdf_path = pdf_path
        self.sse_url = sse_url
        self.export = export
        self.output_dir = output_dir or os.getcwd()
        self.post_url = None
        self.force_ocr = force_ocr
        self.lang = lang  # OCR language, e.g., 'eng' for English, 'hun' for Hungarian

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
        if self.force_ocr:
            print("\nExtracting text using OCR:")
            ocr_txt_pages = self.extract_text(use_ocr=True)
            self._write_text_file(ocr_txt_pages, images, out_dir, "exported-ocr.txt")
            print(f"Created OCR text file: {os.path.join(out_dir, 'exported-ocr.txt')}")
            
        print(f"\nExported PDF '{self.pdf_path}' to directory '{out_dir}'")
        
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
    parser = argparse.ArgumentParser(description='Batch PDF processing: export or MCP upload')
    parser.add_argument('pdf_path', help='PDF file or directory path')
    parser.add_argument('--export', action='store_true', help='Export text/images locally')
    parser.add_argument('--output_dir', default='./exports', help='Base dir for exports')
    parser.add_argument('--sse_url', help='SSE negotiation URL for MCP upload')
    parser.add_argument('--force-ocr', action='store_true', help='Force OCR processing and create exported-ocr.txt')
    parser.add_argument('--lang', default='eng', help='OCR language (e.g., eng, hun). Default is English.')
    args = parser.parse_args()

    uploader = PDFUploader(
        pdf_path=args.pdf_path,
        sse_url=args.sse_url,
        export=args.export,
        output_dir=args.output_dir,
        force_ocr=args.force_ocr,
        lang=args.lang
    )
    uploader.run()
