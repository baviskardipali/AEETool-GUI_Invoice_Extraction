"""
This file uses OCR's namely: Pytesseract and EasyOCR to extract the text from the pdf. OCR's cannot extract text directly from PDF's, so it is important for it to be converted to an image format before  passing it through the OCR.
"""
from pdf2image import convert_from_path
import pytesseract
import timeit
import config
import regex as re
try:
    from PIL import Image
except ImportError:
    import Image


def extractor_pytess(pather):
    #images = convert_from_path(pather)
    images = convert_from_path(pather,poppler_path = config.POPPLER_PATH)
    for i in range(len(images)):
        return pytess(images[i])


def pytess(image):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'	
    extractedInfo = pytesseract.image_to_string((image))
    extractedInfo = " ".join(extractedInfo.split())
    extractedInfo = re.sub(
        '[^A-Za-z0-9#/-]+', ' ', extractedInfo)
    return extractedInfo
