"""
OCR parsing - gated OCR with graceful fallback
"""
import cv2
import numpy as np
import re
import logging

logger = logging.getLogger(__name__)

# Try to import pytesseract, but don't fail if not available
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    logger.info("pytesseract is available")
except ImportError:
    logger.warning("pytesseract not installed. OCR will be disabled. Install with: pip install pytesseract")
except Exception as e:
    logger.warning(f"pytesseract import error: {e}")


class OCRParser:
    """
    Parse text from scanner cells using OCR (phase 1 fields only)
    """
    
    def __init__(self, config):
        """
        Initialize OCR parser
        
        Args:
            config: Configuration module
        """
        self.config = config
        self.enabled = config.ENABLE_OCR and TESSERACT_AVAILABLE
        self.force_ocr = config.FORCE_OCR
        self.ocr_only_if_gap_green = config.OCR_ONLY_IF_GAP_GREEN
        self.symbol_regex = re.compile(config.SYMBOL_REGEX)
        
        if not TESSERACT_AVAILABLE and config.ENABLE_OCR:
            logger.warning("OCR is enabled in config but pytesseract is not available")
    
    def should_run_ocr(self, gap_color_bucket: str) -> bool:
        """
        Determine if OCR should run based on gating rules
        
        Args:
            gap_color_bucket: Color classification (GREEN, WARM, etc.)
            
        Returns:
            True if OCR should run
        """
        if not self.enabled:
            return False
        
        if self.force_ocr:
            return True
        
        if self.ocr_only_if_gap_green:
            return gap_color_bucket == 'GREEN'
        
        # Default: run OCR
        return True
    
    def parse_row(self, columns: dict) -> dict:
        """
        Parse phase 1 fields from row columns
        
        Args:
            columns: Dict with column crops (time_hits, symbol, price)
            
        Returns:
            Dict with parsed fields and success flag
        """
        result = {
            'parsed': False,
            'row_time': None,
            'hits_5s': None,
            'symbol': None,
            'price': None,
            'raw_text': {}
        }
        
        if not self.enabled:
            return result
        
        try:
            # Parse time/hits column
            if 'time_hits' in columns:
                time_hits_text = self._ocr_image(columns['time_hits'])
                result['raw_text']['time_hits'] = time_hits_text
                
                # Extract time (e.g., "09:15:02 am")
                time_match = re.search(r'(\d{1,2}:\d{2}:\d{2}\s*[ap]m)', time_hits_text, re.IGNORECASE)
                if time_match:
                    result['row_time'] = time_match.group(1).strip()
                
                # Extract hits (e.g., "3 in 5sec")
                hits_match = re.search(r'(\d+)\s*in\s*5\s*sec', time_hits_text, re.IGNORECASE)
                if hits_match:
                    result['hits_5s'] = int(hits_match.group(1))
            
            # Parse symbol column
            if 'symbol' in columns:
                symbol_text = self._ocr_image(columns['symbol'])
                result['raw_text']['symbol'] = symbol_text
                
                # Extract and validate symbol
                symbol = self._extract_symbol(symbol_text)
                if symbol:
                    result['symbol'] = symbol
            
            # Parse price column
            if 'price' in columns:
                price_text = self._ocr_image(columns['price'])
                result['raw_text']['price'] = price_text
                
                # Extract price
                price = self._extract_price(price_text)
                if price is not None:
                    result['price'] = price
            
            # Mark as successfully parsed if we got at least symbol
            if result['symbol']:
                result['parsed'] = True
        
        except Exception as e:
            logger.error(f"OCR parsing error: {e}", exc_info=True)
        
        return result
    
    def _ocr_image(self, img: np.ndarray) -> str:
        """
        Run OCR on preprocessed image
        
        Args:
            img: Image to OCR
            
        Returns:
            Extracted text
        """
        # Preprocess for better OCR
        preprocessed = self._preprocess_for_ocr(img)
        
        # Run tesseract
        text = pytesseract.image_to_string(preprocessed, config='--psm 7')
        
        return text.strip()
    
    def _preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for OCR
        
        Args:
            img: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Scale up 2x for better OCR
        h, w = gray.shape
        scaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
        # Apply threshold
        _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _extract_symbol(self, text: str) -> str | None:
        """
        Extract and validate symbol from text
        
        Args:
            text: Raw OCR text
            
        Returns:
            Symbol or None
        """
        # Clean text
        text = text.upper().strip()
        
        # Remove common OCR noise
        text = re.sub(r'[^A-Z.]', '', text)
        
        # Validate against regex
        if self.symbol_regex.match(text):
            return text
        
        return None
    
    def _extract_price(self, text: str) -> float | None:
        """
        Extract price from text
        
        Args:
            text: Raw OCR text
            
        Returns:
            Price or None
        """
        # Find numbers with optional decimal
        match = re.search(r'(\d+\.?\d*)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        return None
