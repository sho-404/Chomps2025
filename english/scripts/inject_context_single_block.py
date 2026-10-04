import os
import json
import glob
from pypdf import PdfReader
from openai import AzureOpenAI
from fuzzywuzzy import process # For handling title variations

# ==========================================
# CONFIGURATION & CREDENTIALS
# ==========================================

# Paths relative to english/scripts/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "../../../datasets/english")
PAPERS_DIR = os.path.join(DATASETS_DIR, "papers")

FOLDERS_TO_PROCESS = ["test", "train", "validation"]
FILE_MAPPING = {
    "test": "en_test_data.jsonl",
    "train": "en_train_data.jsonl",
    "validation": "en_valid_data.jsonl"
}

# Azure OpenAI Config
ENDPOINT = "https://llm-mosaic.cognitiveservices.azure.com/"
MODEL_NAME = "gpt-5-mini"
DEPLOYMENT = "gpt-5-mini"
SUBSCRIPTION_KEY = ""
API_VERSION = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=SUBSCRIPTION_KEY,
)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_pdf_map(papers_dir):
    """
    Creates a dictionary mapping normalized filenames to full paths.
    Returns: { 'normalized_name': 'full_path_to_pdf' }
    """
    if not os.path.exists(papers_dir):
        print(f"CRITICAL ERROR: Papers directory not found at {papers_dir}")
        exit(1)

    pdf_files = glob.glob(os.path.join(papers_dir, "*.pdf"))
    pdf_map = {}
    
    for f in pdf_files:
        basename = os.path.basename(f)
        # Normalize: lowercase, remove extension for easier matching
        clean_name = os.path.splitext(basename)[0].lower().strip()
        pdf_map[clean_name] = f
        # Also store the raw filename as a key just in case
        pdf_map[basename] = f
        
    return pdf_map

def find_best_match(title, pdf_map):
    """
    Tries to find the PDF path for a given title.
    1. Exact match (normalized)
    2. Fuzzy match
    """
    clean_title = title.lower().strip()
    
    # 1. Exact match check
    if clean_title in pdf_map:
        return pdf_map[clean_title]
    
    # 2. Fuzzy match using fuzzywuzzy
    # Get list of available keys (filenames)
    keys = list(pdf_map.keys())
    match = process.extractOne(clean_title, keys)
    
    # match is a tuple (best_match_string, score)
    # We require a reasonably high score to assume it's the same paper (e.g., > 80)
    if match and match[1] >= 85:
        return pdf_map[match[0]]
        
    return None

def extract_text_from_pdf(pdf_path):
    """Reads text from PDF using pypdf."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    except Exception as e:
        return f"[ERROR READING PDF: {e}]"

def get_context_from_llm(question, paper_text):
    """
    Asks the LLM to extract verbatim context.
    """
    # Truncate paper text if excessively long to avoid token limits (sanity check), 
    # though gpt-5-mini likely handles large context.
    # Let's cap it at roughly 100k chars for safety if needed, or send all.
    # We will send all assuming the model handles it.
    
    prompt = f"""
    You are a precise researcher. 
    
    I will provide you with a Question and the full text of a Research Paper.
    Your task is to find the specific segment of text (context) within the paper that contains the answer to the question.
    
    Rules:
    1. The context must be VERBATIM from the text.
    2. Do not paraphrase.
    3. If the answer is not explicitly in the text, return "Context not found in paper."
    4. Return ONLY the context text, nothing else.

    Question: {question}

    Paper Text:
    {paper_text[:120000]} 
    """ 
    # Slicing to 120k chars (~30-40k tokens) to be safe, though GPT-5 likely handles more. Adjust if needed.

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts verbatim text."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=4096,
            model=DEPLOYMENT
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM ERROR: {str(e)}]"

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Initializing PDF Map...")
    pdf_map = get_pdf_map(PAPERS_DIR)
    print(f"Indexed {len(pdf_map)} PDFs in {PAPERS_DIR}\n")

    # ---------------------------------------------------------
    # STEP 1: PRE-FLIGHT CHECK (List papers & match percentage)
    # ---------------------------------------------------------
    print("="*60)
    print("PHASE 1: PAPER AVAILABILITY CHECK")
    print("="*60)

    total_entries_all = 0
    found_entries_all = 0
    files_to_process_data = [] # Store data for Phase 2

    for folder in FOLDERS_TO_PROCESS:
        jsonl_filename = FILE_MAPPING[folder]
        input_path = os.path.join(DATASETS_DIR, folder, jsonl_filename)
        
        if not os.path.exists(input_path):
            print(f"Skipping {folder}: {jsonl_filename} not found.")
            continue

        print(f"\n--- Checking Folder: {folder} ({jsonl_filename}) ---")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        entries = []
        folder_found = 0
        
        for idx, line in enumerate(lines):
            data = json.loads(line)
            title = data.get('title', 'Unknown Title')
            pdf_path = find_best_match(title, pdf_map)
            
            status = "FOUND" if pdf_path else "MISSING"
            if pdf_path:
                folder_found += 1
                found_entries_all += 1
                # Store absolute path for phase 2
                data['_pdf_path'] = pdf_path 
            
            # Print status for every single paper as requested
            # Shortening title for display
            display_title = (title[:50] + '..') if len(title) > 50 else title
            print(f"[{idx+1}] {display_title:<55} | {status}")
            
            entries.append(data)
        
        total_entries_all += len(entries)
        
        # Calculate folder stats
        percentage = (folder_found / len(entries) * 100) if entries else 0
        print(f"\nSummary for {folder}: {folder_found}/{len(entries)} papers found ({percentage:.2f}%)")
        
        files_to_process_data.append({
            "folder": folder,
            "filename": jsonl_filename,
            "entries": entries,
            "input_path": input_path
        })

    # Global Stats
    global_percentage = (found_entries_all / total_entries_all * 100) if total_entries_all > 0 else 0
    print("\n" + "="*60)
    print(f"GLOBAL CHECK SUMMARY: {found_entries_all}/{total_entries_all} files found ({global_percentage:.2f}%)")
    print("="*60 + "\n")

    # Ask user to proceed? (Or just go as implied by prompt "gogo")
    print("Starting Phase 2: Context Injection via LLM...\n")

    # ---------------------------------------------------------
    # STEP 2: PROCESSING & INJECTION
    # ---------------------------------------------------------
    
    total_processed_successfully = 0
    total_failed_llm = 0
    total_skipped_no_pdf = 0

    for file_data in files_to_process_data:
        folder = file_data['folder']
        filename = file_data['filename']
        entries = file_data['entries']
        
        # Output filename: en_test_data_with_context.jsonl
        output_filename = filename.replace(".jsonl", "_with_context.jsonl")
        output_path = os.path.join(DATASETS_DIR, folder, output_filename)
        
        print(f"Processing {folder}/{filename} -> Writing to {output_filename}")
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for i, entry in enumerate(entries):
                title = entry.get('title')
                question = entry.get('question')
                pdf_path = entry.get('_pdf_path') # Retrieved in Phase 1
                
                # Helper for console progress
                print(f"Processing {i+1}/{len(entries)}: {title[:30]}...", end="\r")

                if not pdf_path:
                    # Case: Paper not found
                    entry['context'] = "" 
                    # Remove the temp key before saving
                    if '_pdf_path' in entry: del entry['_pdf_path']
                    
                    json.dump(entry, outfile)
                    outfile.write('\n')
                    total_skipped_no_pdf += 1
                    continue

                # Case: Paper found, get text
                pdf_text = extract_text_from_pdf(pdf_path)
                
                if "[ERROR" in pdf_text:
                    entry['context'] = "" # Error reading PDF
                    total_failed_llm += 1 # technically failed read, but counting here
                else:
                    # Call LLM
                    context = get_context_from_llm(question, pdf_text)
                    
                    if "[LLM ERROR" in context:
                        print(f"\nError calling LLM for {title}: {context}")
                        entry['context'] = ""
                        total_failed_llm += 1
                    else:
                        entry['context'] = context
                        total_processed_successfully += 1

                # Cleanup temp key
                if '_pdf_path' in entry: del entry['_pdf_path']

                # Write immediately
                json.dump(entry, outfile)
                outfile.write('\n')
        
        print(f"\nFinished writing {output_path}\n")

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------
    print("="*60)
    print("FINAL EXECUTION SUMMARY")
    print("="*60)
    print(f"Total Entries Checked:      {total_entries_all}")
    print(f"Papers Found (Local):       {found_entries_all}")
    print("-" * 30)
    print(f"Contexts Generated (LLM):   {total_processed_successfully}")
    print(f"Skipped (Missing PDF):      {total_skipped_no_pdf}")
    print(f"Failed (LLM/PDF Errors):    {total_failed_llm}")
    print("="*60)

if __name__ == "__main__":
    main()