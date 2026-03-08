import fitz  # PyMuPDF
import easyocr
import os

pdf_path = r"C:\Users\manas\Downloads\review 1 pjt.pdf"
output_path = "pdf_content_ocr.txt"

def extract_text_ocr(pdf_path, output_path):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    try:
        print("Initializing EasyOCR (this may take a moment)...")
        reader = easyocr.Reader(['en'], gpu=False) # Use CPU to be safe and avoid CUDA issues if version mismatch
        
        doc = fitz.open(pdf_path)
        full_text = ""
        
        print(f"Processing {len(doc)} pages...")
        
        for i, page in enumerate(doc):
            print(f"Processing page {i+1}/{len(doc)}...")
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            
            # EasyOCR can take bytes directly
            results = reader.readtext(img_bytes, detail=0)
            
            page_text = "\n".join(results)
            full_text += f"--- Page {i + 1} ---\n{page_text}\n\n"
            
        with open(output_path, 'w', encoding='utf-8') as out_file:
            out_file.write(full_text)
            
        print(f"Successfully extracted text to {output_path}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    extract_text_ocr(pdf_path, output_path)
