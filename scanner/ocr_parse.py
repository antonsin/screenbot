"""
OCR parsing - improved accuracy with robust text cleaning and validation
"""
import cv2
import numpy as np
import re
import logging
from pathlib import Path
from datetime import datetime

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
    Parse text from scanner cells using OCR with improved accuracy
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
        
        # OCR character confusion patterns for cleaning
        self.symbol_confusions = {
            '0': 'O', '1': 'I', '5': 'S', '8': 'B',
            '|': 'I', ']': '', '[': '', '{': '', '}': '',
            '!': 'I', '/': '', '\\': '', '-': '', '_': ''
        }
        
        self.price_confusions = {
            'O': '0', 'o': '0', 'I': '1', 'i': '1', 'l': '1',
            'S': '5', 's': '5', 'B': '8', 
            '|': '1', ']': '', '[': '', 'M': '', 'm': '',
            '!': '', ',': '.', # Comma to decimal
        }
        
        # Configure Tesseract path if specified
        if TESSERACT_AVAILABLE and config.TESSERACT_CMD:
            tesseract_path = Path(config.TESSERACT_CMD)
            if tesseract_path.exists():
                pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)
                logger.info(f"Using explicit Tesseract path: {tesseract_path}")
            else:
                logger.warning(f"TESSERACT_CMD set but file not found: {tesseract_path}")
                logger.info("Falling back to PATH for Tesseract")
        elif TESSERACT_AVAILABLE:
            logger.info("Using Tesseract from PATH")
        
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
    
    def parse_row(self, columns: dict, save_debug: bool = False, debug_prefix: str = "ocr") -> dict:
        """
        Parse phase 1 fields from row columns with improved accuracy
        
        Args:
            columns: Dict with column crops (time_hits, symbol, price, gap)
            save_debug: If True, save cell crops to debug/
            debug_prefix: Prefix for debug filenames
            
        Returns:
            Dict with parsed fields, success flag, and validation notes
        """
        result = {
            'parsed': 0,  # 0=failed, 1=success
            'row_time': None,
            'hits_5s': None,
            'symbol': None,
            'price': None,
            'raw_text': {},
            'notes': []
        }
        
        if not self.enabled:
            result['notes'].append("OCR disabled")
            return result
        
        try:
            # Save debug crops if requested
            if save_debug:
                self._save_debug_crops(columns, debug_prefix)
            
            # Parse symbol column (most critical)
            if 'symbol' in columns:
                symbol_text_raw = self._ocr_image(columns['symbol'], mode='symbol')
                result['raw_text']['symbol'] = symbol_text_raw
                logger.debug(f"Symbol OCR raw: '{symbol_text_raw}'")
                
                # Extract and validate symbol
                symbol = self._extract_symbol(symbol_text_raw)
                if symbol:
                    result['symbol'] = symbol
                    logger.debug(f"Symbol extracted: '{symbol}'")
                else:
                    result['notes'].append(f"Symbol parse failed from: '{symbol_text_raw}'")
                    logger.debug(f"Symbol extraction failed from: '{symbol_text_raw}'")
            
            # Parse price column (critical)
            if 'price' in columns:
                price_text_raw = self._ocr_image(columns['price'], mode='price')
                result['raw_text']['price'] = price_text_raw
                logger.debug(f"Price OCR raw: '{price_text_raw}'")
                
                # Extract and validate price
                price, confidence = self._extract_price(price_text_raw)
                if price is not None:
                    result['price'] = price
                    logger.debug(f"Price extracted: {price} (confidence: {confidence})")
                    if confidence == 'low':
                        result['notes'].append(f"Low confidence price: {price} from '{price_text_raw}'")
                else:
                    result['notes'].append(f"Price parse failed from: '{price_text_raw}'")
                    logger.debug(f"Price extraction failed from: '{price_text_raw}'")
            
            # Parse time/hits column (less critical, for context)
            if 'time_hits' in columns:
                time_hits_text = self._ocr_image(columns['time_hits'], mode='generic')
                result['raw_text']['time_hits'] = time_hits_text
                
                # Extract time (e.g., "09:15:02 am")
                time_match = re.search(r'(\d{1,2}:\d{2}:\d{2}\s*[ap]m)', time_hits_text, re.IGNORECASE)
                if time_match:
                    result['row_time'] = time_match.group(1).strip()
                
                # Extract hits (e.g., "3 in 5sec")
                hits_match = re.search(r'(\d+)\s*in\s*5\s*sec', time_hits_text, re.IGNORECASE)
                if hits_match:
                    result['hits_5s'] = int(hits_match.group(1))
            
            # Validation: decide if parse was successful
            # Require BOTH symbol AND valid price for parsed=1
            if result['symbol'] and result['price'] is not None:
                # Additional sanity checks
                if self._validate_price_range(result['price']):
                    result['parsed'] = 1
                    logger.info(f"✓ OCR success: {result['symbol']} @ ${result['price']:.4f}")
                else:
                    result['notes'].append(f"Price {result['price']} out of valid range")
                    result['parsed'] = 0
            else:
                result['parsed'] = 0
                if not result['symbol']:
                    result['notes'].append("Missing symbol")
                if result['price'] is None:
                    result['notes'].append("Missing price")
        
        except Exception as e:
            logger.error(f"OCR parsing error: {e}", exc_info=True)
            result['notes'].append(f"Exception: {str(e)}")
            result['parsed'] = 0
        
        return result
    
    def _ocr_image(self, img: np.ndarray, mode: str = 'generic') -> str:
        """
        Run OCR on preprocessed image with mode-specific configuration
        
        Args:
            img: Image to OCR
            mode: 'symbol', 'price', or 'generic'
            
        Returns:
            Extracted text
        """
        if img is None or img.size == 0:
            return ""
        
        # Preprocess for better OCR
        preprocessed = self._preprocess_for_ocr(img, mode)
        
        # Mode-specific Tesseract configuration
        if mode == 'symbol':
            # Single word, uppercase letters only
            config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ.'
        elif mode == 'price':
            # Single word, digits and decimal
            config = '--psm 8 -c tessedit_char_whitelist=0123456789.$'
        else:
            # Generic single line
            config = '--psm 7'
        
        # Run tesseract
        try:
            text = pytesseract.image_to_string(preprocessed, config=config)
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""
    
    def _preprocess_for_ocr(self, img: np.ndarray, mode: str = 'generic') -> np.ndarray:
        """
        Preprocess image for OCR with mode-specific optimizations
        
        Args:
            img: Input image
            mode: 'symbol', 'price', or 'generic'
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Scale up for better OCR (3x for small text)
        h, w = gray.shape
        scale_factor = 3
        scaled = cv2.resize(gray, (w * scale_factor, h * scale_factor), 
                           interpolation=cv2.INTER_CUBIC)
        
        # Mode-specific preprocessing
        if mode == 'price':
            # For prices: aggressive denoising + sharp threshold
            denoised = cv2.fastNlMeansDenoising(scaled, h=10)
            # Use adaptive threshold to handle varying backgrounds
            thresh = cv2.adaptiveThreshold(denoised, 255, cv2.THRESH_BINARY, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2)
        elif mode == 'symbol':
            # For symbols: denoise + Otsu
            denoised = cv2.fastNlMeansDenoising(scaled, h=8)
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Generic: simple Otsu
            _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if background is dark (text should be black on white for Tesseract)
        mean_val = cv2.mean(thresh)[0]
        if mean_val < 127:
            thresh = cv2.bitwise_not(thresh)
        
        return thresh
    
    def _extract_symbol(self, text: str) -> str | None:
        """
        Extract and validate symbol from text with aggressive cleaning
        
        Args:
            text: Raw OCR text
            
        Returns:
            Symbol or None
        """
        if not text:
            return None
        
        # Clean text
        text = text.upper().strip()
        
        # Apply confusion corrections for symbols
        for wrong, correct in self.symbol_confusions.items():
            text = text.replace(wrong, correct)
        
        # Remove all non-alpha-dot characters
        text = re.sub(r'[^A-Z.]', '', text)
        
        # Handle common prefixes (e.g., "- COMM" or "comm -" patterns)
        # Extract the first valid ticker-like sequence
        words = text.split('.')
        for word in words:
            # Clean word
            word = word.strip()
            # Check if it's a valid ticker length
            if 1 <= len(word) <= 5:
                # Validate against regex
                candidate = word
                if '.' in text and word != text:
                    # Reconstruct with dot if original had one
                    parts = text.split('.')
                    if len(parts) == 2:
                        candidate = f"{parts[0]}.{parts[1]}"
                
                # Final validation
                if self.symbol_regex.match(candidate):
                    return candidate
        
        # Last resort: try the cleaned text as-is
        if self.symbol_regex.match(text):
            return text
        
        return None
    
    def _extract_price(self, text: str) -> tuple[float | None, str]:
        """
        Extract price from text with robust cleaning and validation
        
        Args:
            text: Raw OCR text
            
        Returns:
            Tuple of (price, confidence) where confidence is 'high' or 'low'
        """
        if not text:
            return None, 'low'
        
        original_text = text
        logger.debug(f"Price extraction from: '{original_text}'")
        
        # Apply confusion corrections for prices
        for wrong, correct in self.price_confusions.items():
            text = text.replace(wrong, correct)
        
        logger.debug(f"After confusion correction: '{text}'")
        
        # Remove currency symbols and whitespace
        text = text.replace('$', '').replace('¢', '').strip()
        
        # Remove all non-digit-decimal characters
        text = re.sub(r'[^0-9.]', '', text)
        
        logger.debug(f"After cleanup: '{text}'")
        
        if not text:
            return None, 'low'
        
        # Handle multiple decimal points (keep first)
        parts = text.split('.')
        if len(parts) > 2:
            text = parts[0] + '.' + ''.join(parts[1:])
            logger.debug(f"Multiple decimals fixed: '{text}'")
        
        # Try to parse as float
        try:
            price = float(text)
            
            # Validate: reject obviously wrong values
            if price <= 0:
                logger.debug(f"Price {price} rejected: non-positive")
                return None, 'low'
            
            # Determine confidence based on original text quality
            confidence = 'high'
            
            # Low confidence indicators
            if any(c in original_text for c in ['M', 'm', ']', '[', '|', 'O', 'o']):
                confidence = 'low'
                logger.debug(f"Low confidence due to OCR noise in: '{original_text}'")
            
            # Check if parsed price seems reasonable
            # Most penny stocks: 0.0001 to 1000
            if price < 0.0001 or price > 10000:
                logger.debug(f"Price {price} out of reasonable range")
                return None, 'low'
            
            return price, confidence
        
        except ValueError as e:
            logger.debug(f"Float conversion failed for '{text}': {e}")
            return None, 'low'
    
    def _validate_price_range(self, price: float) -> bool:
        """
        Validate that price is in a reasonable range
        
        Args:
            price: Price to validate
            
        Returns:
            True if valid
        """
        # Reasonable price range for stocks
        return 0.0001 <= price <= 10000.0
    
    def _save_debug_crops(self, columns: dict, prefix: str):
        """
        Save column crop images for debugging
        
        Args:
            columns: Dict of column crops
            prefix: Filename prefix
        """
        debug_dir = self.config.DEBUG_FRAMES_DIR / "ocr_cells"
        debug_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        for col_name, col_img in columns.items():
            if col_img is not None and col_img.size > 0:
                filename = debug_dir / f"{prefix}_{col_name}_{timestamp}.png"
                cv2.imwrite(str(filename), col_img)
        
        logger.debug(f"Saved debug crops to {debug_dir}")
