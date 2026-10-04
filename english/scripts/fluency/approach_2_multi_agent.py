import os
import json
from collections import Counter
from openai import AzureOpenAI

# ==========================================
# CONFIGURATION & CREDENTIALS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "../../../../datasets/english")

FOLDERS = ["test", "train", "validation"]

# Output filename for Fluency Approach 2
OUTPUT_FILENAME = "approach_2_answers_fluency.jsonl"

# Shared Azure Config
ENDPOINT = "https://gpt-shroom.cognitiveservices.azure.com/"
SUBSCRIPTION_KEY = ""

# --- JUDGE 1: GPT-5 Mini ---
JUDGE_1_CONFIG = {
    "deployment": "gpt-5-mini",
    "api_version": "2025-04-01-preview",
    "client": None
}

# --- JUDGE 2: GPT-4.1 Mini ---
JUDGE_2_CONFIG = {
    "deployment": "gpt-4-1-mini-2025-04-14-ft-ee77b52ad26040dfa628828f1e88b44b",
    "api_version": "2024-12-01-preview",
    "client": None
}

# --- JUDGE 3: GPT-4.1 Nano ---
JUDGE_3_CONFIG = {
    "deployment": "1-nano-2025-04-14-clean_ft_english",
    "api_version": "2024-12-01-preview",
    "client": None
}

# Initialize Clients
for config in [JUDGE_1_CONFIG, JUDGE_2_CONFIG, JUDGE_3_CONFIG]:
    config["client"] = AzureOpenAI(
        api_version=config["api_version"],
        azure_endpoint=ENDPOINT,
        api_key=SUBSCRIPTION_KEY,
    )

ALL_JUDGES = [JUDGE_1_CONFIG, JUDGE_2_CONFIG, JUDGE_3_CONFIG]

# ==========================================
# PROMPT DEFINITION (FLUENCY FOCUS)
# ==========================================

def get_fluency_eval_prompt(user_prompt, model_answer):
    """
    Standard Fluency prompt as requested, adapted for the multi-agent workflow.
    """
    return f"""
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

def query_single_judge(judge_config, eval_prompt):
    client = judge_config["client"]
    deployment = judge_config["deployment"]
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a strict binary evaluator. You only output 'y' or 'n'."},
                {"role": "user", "content": eval_prompt}
            ],
            max_completion_tokens=100, # Lower for fluency as it's a binary choice
            temperature=1.0,
            model=deployment,
        )
        result = response.choices[0].message.content.strip().lower()
        
        if 'y' in result: return 'y'
        if 'n' in result: return 'n'
        return 'y' 

    except Exception as e:
        print(f"Error with deployment {deployment}: {e}")
        return "y"

def evaluate_fluency_with_council(user_prompt, model_answer):
    """
    Asks 3 models for fluency verdict. Returns majority vote.
    """
    # Edge Case: Empty Answer
    if not model_answer or len(str(model_answer).strip()) < 1:
        return "y"

    prompt_text = get_fluency_eval_prompt(user_prompt, model_answer)
    votes = []
    
    for judge in ALL_JUDGES:
        verdict = query_single_judge(judge, prompt_text)
        votes.append(verdict)
    
    vote_counts = Counter(votes)
    majority_winner = vote_counts.most_common(1)[0][0]
    
    return majority_winner

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Starting Multi-Agent Fluency Evaluation (Council of 3)...")

    for folder in FOLDERS:
        folder_prefix = "valid" if folder == "validation" else folder
        in_file_name = f"en_{folder_prefix}_data_with_context.jsonl"

        input_path = os.path.join(DATASETS_DIR, folder, in_file_name)
        output_path = os.path.join(DATASETS_DIR, folder, OUTPUT_FILENAME)

        if not os.path.exists(input_path):
            print(f"Skipping {folder}: {in_file_name} not found.")
            continue

        print(f"\nProcessing Folder: {folder}")
        print("-" * 50)

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            lines = infile.readlines()
            total = len(lines)

            for i, line in enumerate(lines):
                data = json.loads(line)
                
                idx = data.get('index')
                user_prompt = data.get('prompt')
                output_text = data.get('output_text')

                # Run Council Evaluation
                fluency_verdict = evaluate_fluency_with_council(user_prompt, output_text)

                # Construct Output (Fluency is calculated, Factuality hardcoded 'n')
                result_obj = {
                    "index": idx,
                    "has_fluency_mistakes": fluency_verdict,
                    "has_factual_mistakes": "n"
                }

                json.dump(result_obj, outfile)
                outfile.write('\n')

                print(f"[{i+1}/{total}] ID: {idx} | Council Winner: {fluency_verdict.upper()}", end="\r")

        print(f"\nCompleted {folder}.\n")

    print("="*60)
    print("All folders processed for Fluency Approach 2.")

if __name__ == "__main__":
    main()