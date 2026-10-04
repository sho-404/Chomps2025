import os
import json
from openai import AzureOpenAI

# ==========================================
# CONFIGURATION & CREDENTIALS
# ==========================================

# Paths relative to english/scripts/factuality/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "../../../../datasets/spanish")

# Folders to process
FOLDERS = ["test", "train", "validation"]

# Output filename (per folder)
OUTPUT_FILENAME = "approach_1_answers_factuality.jsonl"

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
# PROMPT DEFINITION (EVALUATOR)
# ==========================================

def evaluate_factuality(user_prompt, context, model_answer):
    """
    Asks the LLM if the model_answer is supported by the context given the specific prompt.
    
    Returns:
    'y' -> Yes, it HAS factual mistakes (or is unsupported).
    'n' -> No, it does NOT have factual mistakes (it is correct/supported).
    """
    
    # 1. Edge Case: Empty Context
    # If we have no context (e.g., PDF wasn't found), we cannot verify factuality.
    # Therefore, we assume the answer is not supported (Risk of Hallucination).
    print("CTX LEN:", 0 if context is None else len(str(context).strip()))
    # print("CTX PREVIEW:", str(context)[:200]) # Optional debug

    if not context:
        print("NO CONTEXT HERE!")
        return "y"

    # 2. Evaluation Prompt (TRANSLATED TO SPANISH)
    # Context remains verbatim (likely English/Original), Prompt & Answer are Spanish.
    eval_prompt = f"""
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

    try:
        response = client.chat.completions.create(
            messages=[
                # System prompt translated to Spanish
                {"role": "system", "content": "Eres un evaluador binario estricto. Solo respondes con 'y' o 'n'."},
                {"role": "user", "content": eval_prompt}
            ],
            max_completion_tokens=16384, 
            model=DEPLOYMENT,
        )
        result = response.choices[0].message.content.strip().lower()
        print("LLM RESPONSE: ", result)
        
        # Strict parsing
        if 'y' in result: return 'y'
        if 'n' in result: return 'n'
        
        # Fallback if LLM outputs something weird
        return 'lol idk' 

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return "y"

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Starting Factuality Evaluation (Prompt-based) [SPANISH MODE]...")

    for folder in FOLDERS:
        # Determine Input Filename based on folder structure provided
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

                # Run Evaluation
                # Returns 'y' if mistakes found, 'n' if clean.
                factual_mistake_flag = evaluate_factuality(user_prompt, context, output_text)

                # Construct Output Object
                result_obj = {
                    "index": idx,
                    "has_fluency_mistakes": "n", # Hardcoded 'n' per instructions
                    "has_factual_mistakes": factual_mistake_flag
                }

                # Write immediately to file
                json.dump(result_obj, outfile)
                outfile.write('\n')

                # Console Progress
                print(f"[{i+1}/{total}] ID: {idx} | Mistake? {factual_mistake_flag.upper()}", end="\r")

        print(f"\nCompleted {folder}.\n")

    print("="*60)
    print("All folders processed successfully.")

if __name__ == "__main__":
    main()