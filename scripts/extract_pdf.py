import PyPDF2
import os

pdf_path = r"C:\Users\manas\Downloads\review 1 pjt.pdf"
output_path = "pdf_content.txt"

def extract_text_from_pdf(pdf_path, output_path):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += f"--- Page {page_num + 1} ---\n"
                text += page.extract_text() + "\n\n"
            
            with open(output_path, 'w', encoding='utf-8') as out_file:
                out_file.write(text)
            
            print(f"Successfully extracted text to {output_path}")
            print(f"Total pages: {len(reader.pages)}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    extract_text_from_pdf(pdf_path, output_path)
