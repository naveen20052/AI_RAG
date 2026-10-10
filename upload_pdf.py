import os
import shutil

SOURCE_FOLDER = "data"

print("PDF Upload Setup")
print("----------------")

pdf_name = input("Enter the PDF filename inside the data folder: ").strip()

pdf_path = os.path.join(SOURCE_FOLDER, pdf_name)

if not os.path.exists(pdf_path):
    print("❌ PDF not found.")
    print(f"Put your PDF inside: {SOURCE_FOLDER}")
    exit()

if not pdf_name.lower().endswith(".pdf"):
    print("❌ Please provide a PDF file.")
    exit()

print(f"\n✅ PDF found: {pdf_name}")
print(f"Path: {pdf_path}")

print("\nNow run:")
print("python store_embeddings.py")
print("python chat.py")