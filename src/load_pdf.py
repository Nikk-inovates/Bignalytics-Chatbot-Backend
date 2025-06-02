import fitz  # PyMuPDF
import os

def load_pdf_text(file_path):
    if not file_path:
        raise ValueError("❌ No file path provided.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        raise ValueError("❌ The provided file must be a PDF (.pdf extension).")

    try:
        doc = fitz.open(file_path)

        if doc.page_count == 0:
            raise ValueError("❌ The PDF has no pages.")

        text = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if not page_text.strip():
                print(f"⚠️ Page {page_num + 1} has no readable text.")
            text.append(page_text)

        full_text = "\n".join(text).strip()

        if not full_text:
            raise ValueError("❌ No readable text found in the PDF.")

        return full_text

    except Exception as e:
        raise Exception(f"❌ Failed to read PDF: {e}")