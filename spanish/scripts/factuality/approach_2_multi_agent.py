import os
import json
from collections import Counter
from openai import AzureOpenAI

# ==========================================
# CONFIGURATION & CREDENTIALS
# ==========================================

# Paths relative to current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "../../../../datasets/spanish")

# Folders to process
FOLDERS = ["test", "train", "validation"]

# Output filename (per folder)
OUTPUT_FILENAME = "approach_2_answers_factuality.jsonl"

# Shared Azure Config
ENDPOINT = "https://gpt-shroom.cognitiveservices.azure.com/"
SUBSCRIPTION_KEY = os.environ["AZURE_OPENAI_API_KEY"]  # set in the environment; never commit keys

# --- JUDGE 1: GPT-5 Mini (Original from Script 1) ---
JUDGE_1_CONFIG = {
    "deployment": "gpt-5-mini",
    "api_version": "2025-04-01-preview",
    "client": None 
}

# --- JUDGE 2: GPT-4.1 Mini (Fine-tuned Spanish) ---
JUDGE_2_CONFIG = {
    "deployment": "gpt-4-1-mini-2025-04-14-ft-spanish",
    "api_version": "2025-01-01-preview",
    "client": None
}

# --- JUDGE 3: GPT-4.1 Nano (Fine-tuned Spanish) ---
JUDGE_3_CONFIG = {
    "deployment": "gpt-4-1-nano-2025-04-14-ft-spanish",
    "api_version": "2025-01-01-preview",
    "client": None
}

# Initialize Clients
JUDGE_1_CONFIG["client"] = AzureOpenAI(
    api_version=JUDGE_1_CONFIG["api_version"],
    azure_endpoint=ENDPOINT,
    api_key=SUBSCRIPTION_KEY,
)

JUDGE_2_CONFIG["client"] = AzureOpenAI(
    api_version=JUDGE_2_CONFIG["api_version"],
    azure_endpoint=ENDPOINT,
    api_key=SUBSCRIPTION_KEY,
)

JUDGE_3_CONFIG["client"] = AzureOpenAI(
    api_version=JUDGE_3_CONFIG["api_version"],
    azure_endpoint=ENDPOINT,
    api_key=SUBSCRIPTION_KEY,
)

ALL_JUDGES = [JUDGE_1_CONFIG, JUDGE_2_CONFIG, JUDGE_3_CONFIG]

# ==========================================
# PROMPT DEFINITION (SPANISH)
# ==========================================

def get_eval_prompt(user_prompt, context, model_answer):
    return f"""
    Eres un Evaluador. 
    
    Te proporcionaré tres textos:
    1. [Contexto]: Contexto extraído del documento fuente (puede estar en el idioma original).
    2. [Prompt del Usuario]: La entrada original dada a un modelo de IA (en español).
    3. [Respuesta del Modelo]: La respuesta generada por ese modelo de IA (en español).

    TU TAREA:
    Determina si la [Respuesta del Modelo] está respaldada y sustentada por el [Contexto].

    REGLAS DE DECISIÓN:
    - Responde 'n' (No hay errores) si la [Respuesta del Modelo] está respaldada por el contexto y parece correcta.
    - Responde 'y' (Sí, hay errores) si la [Respuesta del Modelo]: no está respaldada por el contexto, es absurda o está vacía.

    IMPORTANTE: 
    - Debes generar SOLAMENTE un único carácter: 'y' o 'n'.
    - No proporciones explicaciones.

    === DATOS ===
    
    [Contexto]:
    {context[:60000]} 

    [Prompt del Usuario]:
    {user_prompt}

    [Respuesta del Modelo]:
    {model_answer}
    """

def query_single_judge(judge_config, eval_prompt):
    """
    Sends the prompt to a specific judge configuration.
    Returns 'y', 'n', or 'y' (as fallback for errors).
    """
    client = judge_config["client"]
    deployment = judge_config["deployment"]
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Eres un evaluador binario estricto. Solo respondes con 'y' o 'n'."},
                {"role": "user", "content": eval_prompt}
            ],
            max_completion_tokens=16384, 
            temperature=1.0,            
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment,
        )
        result = response.choices[0].message.content.strip().lower()
        
        # Strict parsing
        if 'y' in result: return 'y'
        if 'n' in result: return 'n'
        
        return 'y' # Fallback for unclear answers

    except Exception as e:
        print(f"Error with deployment {deployment}: {e}")
        return "y" # Conservative fallback

def evaluate_with_council(user_prompt, context, model_answer):
    """
    Asks 3 models. Returns the majority vote ('y' or 'n').
    """
    
    # 1. Edge Case: Empty Context
    if not context:
        print("NO CONTEXT - Automatic 'y'")
        return "y"

    # 2. Prepare Prompt (Spanish)
    prompt_text = get_eval_prompt(user_prompt, context, model_answer)
    
    votes = []
    
    # 3. Gather Votes
    print(f"Polling Council...", end=" ")
    for i, judge in enumerate(ALL_JUDGES):
        verdict = query_single_judge(judge, prompt_text)
        votes.append(verdict)
    
    # 4. Majority Rule
    vote_counts = Counter(votes)
    majority_winner = vote_counts.most_common(1)[0][0] # Returns 'y' or 'n'
    
    print(f"-> Votes: {votes} -> Winner: {majority_winner.upper()}")
    
    return majority_winner

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Starting Factuality Evaluation (Approach 2: Council of 3) [SPANISH MODE]...")

    for folder in FOLDERS:
        # Determine Input Filename (Spanish Prefix)
        folder_prefix = "valid" if folder == "validation" else folder
        in_file_name = f"es_{folder_prefix}_data_with_context.jsonl"

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
                context = data.get('context')

                print(f"[{i+1}/{total}] ID: {idx}", end=" ")

                # Run Council Evaluation
                final_verdict = evaluate_with_council(user_prompt, context, output_text)

                # Construct Output Object
                result_obj = {
                    "index": idx,
                    "has_fluency_mistakes": "n", 
                    "has_factual_mistakes": final_verdict
                }

                # Write immediately
                json.dump(result_obj, outfile)
                outfile.write('\n')

        print(f"\nCompleted {folder}.\n")

    print("="*60)
    print("All folders processed successfully.")

if __name__ == "__main__":
    main()