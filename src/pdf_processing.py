import io

import pdfplumber
import pytesseract

from pdf2image import convert_from_bytes


def extract_text_from_pdf(pdf_bytes):
    extracted_text = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()

            if text:
                extracted_text.append(text)

    return "\n".join(extracted_text)


def extract_text_with_ocr(pdf_bytes):
    pages = convert_from_bytes(pdf_bytes)

    extracted_text = []

    for page in pages:
        text = pytesseract.image_to_string(page)

        if text:
            extracted_text.append(text)

    return "\n".join(extracted_text)


def extract_pdf_text(pdf_bytes):
    text = extract_text_from_pdf(pdf_bytes)

    if text.strip():
        return text

    return extract_text_with_ocr(pdf_bytes)


if __name__ == "__main__":
    print("PDF processing module ready.")