import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def save_text_to_file(text, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    input_pdf = "assets/CV-Rashid-eng.pdf"
    output_txt = "knowledge_base/my_cv.txt"

    print(f"📥 Extracting from: {input_pdf}")
    extracted_text = extract_text_from_pdf(input_pdf)
    save_text_to_file(extracted_text, output_txt)
    print(f"✅ Saved extracted text to: {output_txt}")
