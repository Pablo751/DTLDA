import os
from PyPDF2 import PdfReader

def convert_pdf_to_txt(path):
    pdf = PdfReader(path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def convert_all_pdfs_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            text = convert_pdf_to_txt(pdf_path)
            txt_filename = filename.replace(".pdf", ".txt")
            txt_path = os.path.join(folder_path, txt_filename)
            with open(txt_path, "w") as txt_file:
                txt_file.write(text)

# Replace this with your actual folder path
folder_path = "/Users/juanpablocasado/Downloads/OneDrive_1_7-8-2023"
convert_all_pdfs_in_folder(folder_path)
