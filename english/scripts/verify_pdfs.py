import os
import glob
from pypdf import PdfReader

# ==========================================
# CONFIGURATION
# ==========================================

# Assumes this script is in english/scripts/
# Adjust relative path if necessary
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAPERS_DIR = os.path.join(BASE_DIR, "../../../datasets/english/papers")

def check_pdf_health(file_path):
    """
    Attempts to open the PDF and read the first page.
    Returns (is_healthy, error_message).
    """
    try:
        # 1. Check if file is empty
        if os.path.getsize(file_path) == 0:
            return False, "File is empty (0 bytes)"

        # 2. Attempt to parse PDF structure
        reader = PdfReader(file_path)
        
        # 3. Attempt to read the first page
        # (Most corruption errors like EOF/Invalid Header trigger here)
        if len(reader.pages) > 0:
            _ = reader.pages[0].extract_text()
        else:
            return False, "PDF has no pages"
            
        return True, None

    except Exception as e:
        return False, str(e)

def main():
    print(f"--- Starting PDF Sanity Check ---")
    print(f"Scanning directory: {PAPERS_DIR}\n")

    if not os.path.exists(PAPERS_DIR):
        print(f"Error: Directory not found: {PAPERS_DIR}")
        return

    pdf_files = [f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf")]
    total_files = len(pdf_files)
    
    print(f"Found {total_files} PDFs. Beginning check...\n")

    healthy_files = []
    unhealthy_files = []

    for i, filename in enumerate(pdf_files):
        file_path = os.path.join(PAPERS_DIR, filename)
        
        # Simple progress indicator
        print(f"[{i+1}/{total_files}] Checking: {filename[:40]}...", end="\r")
        
        is_healthy, error_msg = check_pdf_health(file_path)
        
        if is_healthy:
            healthy_files.append(filename)
        else:
            unhealthy_files.append((filename, error_msg))

    print("\n" + "="*50)
    print("SCAN COMPLETE")
    print("="*50)

    # ------------------------------------------
    # REPORTING
    # ------------------------------------------
    
    print(f"\n✅ HEALTHY FILES: {len(healthy_files)}")
    
    print(f"\n❌ UNHEALTHY FILES: {len(unhealthy_files)}")
    if unhealthy_files:
        print("-" * 50)
        print(f"{'Filename':<50} | {'Error'}")
        print("-" * 50)
        for name, err in unhealthy_files:
            print(f"{name:<50} | {err}")
        print("-" * 50)
        
        # Save bad files to a text file for easy copy-pasting
        report_path = os.path.join(BASE_DIR, "bad_pdfs_report.txt")
        with open(report_path, "w") as f:
            for name, err in unhealthy_files:
                f.write(f"{name}\n")
        print(f"\nList of bad filenames saved to: {report_path}")
    else:
        print("\nGreat news! All PDFs appear to be valid.")

if __name__ == "__main__":
    main()