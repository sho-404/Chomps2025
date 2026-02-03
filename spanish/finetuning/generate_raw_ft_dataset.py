import os
import json
import glob
from pypdf import PdfReader
from openai import AzureOpenAI

# --- Configuration ---
ENDPOINT = "https://gpt-shroom.cognitiveservices.azure.com/"
DEPLOYMENT = "gpt-5-mini"
# Using the key you provided
SUBSCRIPTION_KEY = "DPKIyguzI8NDCOsmqx4jdMI0vAB5NA9hpeTeggi0m9SIQ5nGgEpuJQQJ99BJACYeBjFXJ3w3AAABACOGkO2z"
API_VERSION = "2024-12-01-preview"

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "papers")
OUTPUT_FILE = "finetune_dataset_raw.jsonl"

# --- Client Setup ---
client = AzureOpenAI(
    api_key=SUBSCRIPTION_KEY,
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
)

def extract_text_from_pdf(pdf_path):
    """Reads a PDF and returns the full text content."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def generate_batch(paper_text, paper_title):
    """
    Asks the LLM to generate 2 examples: 
    1 Correct (Verdict 'n') and 1 Incorrect (Verdict 'y').
    Questions and Answers will be in Spanish.
    """
    
    # MODIFIED SYSTEM PROMPT FOR SPANISH OUTPUT
    system_prompt = """
    You are a data generation assistant. Your task is to create fine-tuning data based on academic papers.
    
    You must output a JSON object containing a list called "pairs".
    The list must contain exactly 2 objects:
    
    1. A "Correct" example:
       - question: Must be in SPANISH and start with "En el documento ".
       - context: A VERBATIM excerpt from the text that answers the question (Keep original language of the text).
       - answer: A correct answer grounded in the context, written in SPANISH.
       - veredict: "n" (meaning No mistakes).
       
    2. An "Incorrect" example:
       - question: Must be in SPANISH and start with "En el documento ".
       - context: A VERBATIM excerpt from the text that contains the *real* answer (Keep original language of the text).
       - answer: A plausible but factually INCORRECT answer (hallucination or wrong number) written in SPANISH that contradicts the context.
       - veredict: "y" (meaning Yes, it has mistakes).
       
    Output strictly valid JSON.
    """

    user_prompt = f"""
    Paper Title: {paper_title}
    
    Paper Text:
    {paper_text}
    
    Generate the 2 examples (1 correct, 1 incorrect) in Spanish now.
    """

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}, 
            max_completion_tokens=2000
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        return data.get("pairs", [])
        
    except Exception as e:
        print(f"Error generating batch for {paper_title}: {e}")
        return []

def main():
    # Ensure papers directory exists
    if not os.path.exists(PAPERS_DIR):
        print(f"Directory not found: {PAPERS_DIR}")
        return

    # Open output file in append mode
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        
        # Find all PDF files
        pdf_files = glob.glob(os.path.join(PAPERS_DIR, "*.pdf"))
        print(f"Found {len(pdf_files)} papers.")

        for pdf_path in pdf_files:
            file_name = os.path.basename(pdf_path)
            paper_title = os.path.splitext(file_name)[0]
            
            print(f"Processing: {file_name}...")
            
            # 1. Extract Text
            full_text = extract_text_from_pdf(pdf_path)
            if not full_text:
                continue

            # 2. Loop 4 times to get 8 items total (4 correct, 4 incorrect)
            for i in range(4):
                print(f"  - Batch {i+1}/4...")
                pairs = generate_batch(full_text, paper_title)
                
                # 3. Write results immediately to JSONL
                for item in pairs:
                    # Double check structure before writing
                    if all(k in item for k in ["question", "context", "answer", "veredict"]):
                        # Ensure ascii=False so Spanish characters (ñ, á, é) are saved correctly
                        json.dump(item, f_out, ensure_ascii=False)
                        f_out.write("\n")
                    else:
                        print("    Skipping malformed item.")
                        
    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()