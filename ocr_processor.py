import pytesseract
from PIL import Image
import io

class OCRProcessor:
    """
    Handles OCR-based text extraction from PDF pages.
    """
    def __init__(self, lang='eng'):
        """
        Initialize the OCR processor with the desired language.
        
        Args:
            lang: The language for Tesseract OCR (e.g., 'eng', 'hun').
        """
        self.lang = lang

    def extract_text_from_image(self, pix, page_num, total_pages):
        """
        Extracts text from a pixmap image using Tesseract OCR.

        Args:
            pix: A fitz.Pixmap object.
            page_num: The page number being processed (for logging).

        Returns:
            str: The extracted OCR text, or an empty string if an error occurs.
        """
        try:
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            ocr_text = pytesseract.image_to_string(img, lang=self.lang)
            print(f"Page {page_num}/{total_pages}: OCR successful using language '{self.lang}'")
            return ocr_text
        except pytesseract.TesseractNotFoundError:
            print("Tesseract is not installed or not in your PATH.")
            print("Please install Tesseract and add it to your system's PATH.")
            return ""
        except Exception as e:
            print(f"An error occurred during OCR on page {page_num}: {e}")
            return ""