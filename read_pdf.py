from pypdf import PdfReader

pdf_path = "data/sample.pdf"

reader = PdfReader(pdf_path)

for page in reader.pages:
    text = page.extract_text()
    print(text)