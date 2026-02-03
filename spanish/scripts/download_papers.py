import json
import os
import re
import requests
import unicodedata
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent # Points to 'english' folder
DATASETS_DIR = BASE_DIR / "datasets"
PAPERS_DIR = DATASETS_DIR / "papers"

# specific file paths
FILES_TO_PROCESS = [
    DATASETS_DIR / "test" / "es_test_data.jsonl",
    DATASETS_DIR / "train" / "es_train_data.jsonl",
    DATASETS_DIR / "validation" / "es_valid_data.jsonl"
]

def normalize_filename(text):
    """
    Converts text to a valid filename:
    1. Normalizes unicode characters (removes accents).
    2. Removes characters that aren't letters, numbers, spaces, dashes, or underscores.
    3. Truncates to 150 chars to avoid filesystem limits.
    """
    # Remove accents
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    # Replace slashes or colons (common in titles) with underscores
    text = text.replace('/', '_').replace(':', ' -')
    # Remove other invalid characters
    text = re.sub(r'[^\w\s-]', '', text).strip()
    return text[:150]

def main():
    # 1. Setup Directories
    if not PAPERS_DIR.exists():
        os.makedirs(PAPERS_DIR)
        print(f"Created directory: {PAPERS_DIR}")

    unique_papers = {} # Key: Title, Value: URL

    print("--- Reading JSONL files ---")
    
    # 2. Extract Unique Papers
    for file_path in FILES_TO_PROCESS:
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        title = entry.get("title")
                        url = entry.get("url")
                        
                        # Only add if we have both title and url
                        if title and url:
                            unique_papers[title] = url
                    except json.JSONDecodeError:
                        continue # Skip bad lines
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"Found {len(unique_papers)} unique papers. Starting download...")

    # 3. Download Papers
    paper_log = [] # To store status for paper_list file
    download_count = 0
    fail_count = 0

    for title, url in unique_papers.items():
        safe_title = normalize_filename(title)
        filename = f"{safe_title}.pdf"
        file_path = PAPERS_DIR / filename
        
        status = "PENDING"
        
        try:
            print(f"Downloading: {title[:50]}...")
            
            # Request the URL
            response = requests.get(url, timeout=15, stream=True)
            
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                status = "DOWNLOADED"
                download_count += 1
            else:
                status = f"FAILED (Status {response.status_code})"
                fail_count += 1
                
        except Exception as e:
            status = f"FAILED (Error: {str(e)})"
            fail_count += 1

        # Log entry
        paper_log.append(f"{title} | {url} | {status}")

    # 4. Create paper_list report
    report_path = PAPERS_DIR / "paper_list.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Paper Download Report\n")
        f.write("=====================\n")
        f.write("\n".join(paper_log))

    print("-" * 30)
    print(f"Process complete.")
    print(f"Report saved to: {report_path}")
    
    if fail_count == 0 and download_count > 0:
        print("All papers were downloaded successfully.")
    elif download_count == 0:
        print("No papers were found or downloaded.")
    else:
        print(f"Finished with issues. {download_count} downloaded, {fail_count} failed.")

if __name__ == "__main__":
    main()