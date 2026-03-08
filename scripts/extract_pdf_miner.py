from pdfminer.high_level import extract_text
import os

pdf_path = r"C:\Users\manas\Downloads\review 1 pjt.pdf"
output_path = "pdf_content_miner.txt"

def extract_text_miner(pdf_path, output_path):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    try:
        text = extract_text(pdf_path)
        
        with open(output_path, 'w', encoding='utf-8') as out_file:
            out_file.write(text)
        
        print(f"Successfully extracted text to {output_path}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    extract_text_miner(pdf_path, output_path)
