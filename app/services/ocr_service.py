"""OCR fallback for scanned / image-only PDF pages (EasyOCR, CPU)."""

import logging
from functools import lru_cache

import numpy as np
import pymupdf

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _get_reader():
    """Load the EasyOCR model once. Returns None when OCR is unavailable."""
    settings = get_settings()
    try:
        import easyocr
    except ImportError:
        logger.warning("easyocr is not installed; scanned PDF pages will be skipped")
        return None
    logger.info("Loading EasyOCR model for languages %s", settings.ocr_languages)
    return easyocr.Reader(settings.ocr_languages, gpu=False, verbose=False)


def ocr_available() -> bool:
    return get_settings().ocr_enabled and _get_reader() is not None


def ocr_page(page: pymupdf.Page) -> str:
    """Render a page to an image and return the recognised text, paragraph-grouped."""
    reader = _get_reader()
    if reader is None:
        return ""
    pix = page.get_pixmap(dpi=get_settings().ocr_dpi, colorspace=pymupdf.csRGB, alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    paragraphs = reader.readtext(image, detail=0, paragraph=True)
    return "\n\n".join(p.strip() for p in paragraphs if p.strip())
