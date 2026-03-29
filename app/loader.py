from pypdf import PdfReader

def load_pdf(pdf_path):
    """Extract text from PDF file"""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += '\n' + page.extract_text()
    print(f"✅ PDF loaded — {len(full_text)} characters")
    return full_text