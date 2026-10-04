import os
import json
from openai import AzureOpenAI

# ==========================================
# CONFIGURATION & CREDENTIALS
# ==========================================

# Paths relative to english/scripts/factuality/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "../../../../datasets/english")

# Folders to process
FOLDERS = ["test", "train", "validation"]

# Output filename (per folder)
OUTPUT_FILENAME = "approach_1_answers_fluency.jsonl"

# Azure OpenAI Config
ENDPOINT = "https://gpt-shroom.cognitiveservices.azure.com/"
DEPLOYMENT = "gpt-5-mini"
SUBSCRIPTION_KEY = ""
API_VERSION = "2025-04-01-preview"

client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=ENDPOINT,
    api_key=SUBSCRIPTION_KEY,
)

# ==========================================
# PROMPT DEFINITION (FLUENCY EVALUATOR)
# ==========================================

def evaluate_fluency(user_prompt, model_answer):
    """
    Asks the LLM if the model_answer makes sense, is coherent, and is grammatically correct 
    given the prompt.
    
    Returns:
    'y' -> Yes, it HAS fluency mistakes (it's nonsense, broken, or incoherent).
    'n' -> No, it does NOT have fluency mistakes (it flows well and makes sense).
    """
    
    # 1. Edge Case: Empty Answer
    if not model_answer or len(str(model_answer).strip()) < 1:
        print("EMPTY MODEL ANSWER!")
        return "y"

    # 2. Evaluation Prompt
    # We do NOT include context here, as fluency is about the text quality itself.
    eval_prompt = f"""
    You are a Linguistic Evaluator.
    
    I will provide you with a [User Prompt] and a [Model Answer].

    YOUR TASK:
    Determine if the [Model Answer] contains **fluency mistakes**.
    
    A "Fluency Mistake" includes:
    1. Grammatical errors or broken syntax.
    2. Incoherence or "word salad" (gibberish).
    3. Severe repetition (looping the same phrase).
    4. The answer does not make logical sense as a response to the prompt (non-sequitur).
    5. The answer cuts off mid-sentence abruptly.

    DECISION RULES:
    - Output 'n' (No mistakes) if the text is clear, coherent, and reads like a natural response.
    - Output 'y' (Yes, mistakes) if the text feels off, broken, or nonsense.

    IMPORTANT: 
    - You must output ONLY a single character: 'y' or 'n'.
    - Do not provide explanations.

    === DATA ===

    [User Prompt]:
    {user_prompt}

    [Model Answer]:
    {model_answer}
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a strict binary evaluator. You only output 'y' or 'n'."},
                {"role": "user", "content": eval_prompt}
            ],
            max_completion_tokens=16384, 
            model=DEPLOYMENT,
        )
        result = response.choices[0].message.content.strip().lower()
        # print("LLM RESPONSE: ", result) # Uncomment for debug
        
        # Strict parsing
        if 'y' in result: return 'y'
        if 'n' in result: return 'n'
        
        # Fallback
        return 'y' 

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return "y"

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Starting Fluency Evaluation (Prompt-based)...")

    for folder in FOLDERS:
        # Determine Input Filename based on folder structure
        folder_prefix = "valid" if folder == "validation" else folder
        in_file_name = f"en_{folder_prefix}_data_with_context.jsonl"

        input_path = os.path.join(DATASETS_DIR, folder, in_file_name)
        output_path = os.path.join(DATASETS_DIR, folder, OUTPUT_FILENAME)

        if not os.path.exists(input_path):
            print(f"Skipping {folder}: {in_file_name} not found.")
            continue

        print(f"\nProcessing Folder: {folder}")
        print(f"Reading: {in_file_name}")
        print(f"Writing: {OUTPUT_FILENAME}")
        print("-" * 50)

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            lines = infile.readlines()
            total = len(lines)

            for i, line in enumerate(lines):
                data = json.loads(line)
                
                # Extract relevant fields
                idx = data.get('index')
                user_prompt = data.get('prompt')
                output_text = data.get('output_text')
                
                # NOTE: We do not need 'context' for fluency checking, 
                # strictly speaking, but we still read the file same as before.

                # Run Evaluation
                # Returns 'y' if mistakes found (nonsense), 'n' if clean.
                fluency_mistake_flag = evaluate_fluency(user_prompt, output_text)

                # Construct Output Object
                result_obj = {
                    "index": idx,
                    "has_fluency_mistakes": fluency_mistake_flag, # Calculated
                    "has_factual_mistakes": "n" # Hardcoded n
                }

                # Write immediately to file
                json.dump(result_obj, outfile)
                outfile.write('\n')

                # Console Progress
                print(f"[{i+1}/{total}] ID: {idx} | Bad Fluency? {fluency_mistake_flag.upper()}", end="\r")

        print(f"\nCompleted {folder}.\n")

    print("="*60)
    print("All folders processed successfully.")

if __name__ == "__main__":
    main()