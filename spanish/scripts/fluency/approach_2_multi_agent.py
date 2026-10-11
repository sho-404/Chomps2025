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

# Output filename (Approach 2: Council)
OUTPUT_FILENAME = "approach_2_answers_fluency.jsonl"

# Shared Azure Config
ENDPOINT = "https://gpt-shroom.cognitiveservices.azure.com/"
SUBSCRIPTION_KEY = os.environ["AZURE_OPENAI_API_KEY"]  # set in the environment; never commit keys

# --- JUDGE 1: GPT-5 Mini ---
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
for config in [JUDGE_1_CONFIG, JUDGE_2_CONFIG, JUDGE_3_CONFIG]:
    config["client"] = AzureOpenAI(
        api_version=config["api_version"],
        azure_endpoint=ENDPOINT,
        api_key=SUBSCRIPTION_KEY,
    )

ALL_JUDGES = [JUDGE_1_CONFIG, JUDGE_2_CONFIG, JUDGE_3_CONFIG]

# ==========================================
# PROMPT DEFINITION (SPANISH FLUENCY)
# ==========================================

def get_fluency_eval_prompt_es(user_prompt, model_answer):
    """
    Prompt en español para evaluar la fluidez lingüística.
    """
    return f"""
    Eres un Evaluador Lingüístico experto.
    
    Te proporcionaré un [Prompt del Usuario] y una [Respuesta del Modelo].

    TU TAREA:
    Determinar si la [Respuesta del Modelo] contiene **errores de fluidez**.
    
    Un "Error de Fluidez" incluye:
    1. Errores gramaticales graves o sintaxis rota.
    2. Incoherencia o "ensalada de palabras" (texto sin sentido).
    3. Repetición severa (bucles de la misma frase).
    4. La respuesta no tiene lógica como reacción al prompt (non-sequitur).
    5. La respuesta se corta abruptamente a mitad de una frase.

    REGLAS DE DECISIÓN:
    - Responde 'n' (Sin errores) si el texto es claro, coherente y se lee como una respuesta natural.
    - Responde 'y' (Sí, hay errores) si el texto se siente extraño, roto o no tiene sentido.

    IMPORTANTE: 
    - Debes generar SOLAMENTE un único carácter: 'y' o 'n'.
    - No proporciones explicaciones.

    === DATOS ===

    [Prompt del Usuario]:
    {user_prompt}

    [Respuesta del Modelo]:
    {model_answer}
    """

def query_single_judge(judge_config, eval_prompt):
    client = judge_config["client"]
    deployment = judge_config["deployment"]
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Eres un evaluador binario estricto. Solo respondes con 'y' o 'n'."},
                {"role": "user", "content": eval_prompt}
            ],
            max_completion_tokens=100, 
            temperature=1.0,            
            model=deployment,
        )
        result = response.choices[0].message.content.strip().lower()
        
        # Strict parsing
        if 'y' in result: return 'y'
        if 'n' in result: return 'n'
        
        return 'y' # Fallback conservative

    except Exception as e:
        print(f"Error with deployment {deployment}: {e}")
        return "y"

def evaluate_fluency_with_council_es(user_prompt, model_answer):
    """
    Consulta a los 3 modelos y retorna el voto de la mayoría.
    """
    # Edge Case: Respuesta vacía
    if not model_answer or len(str(model_answer).strip()) < 1:
        return "y"

    prompt_text = get_fluency_eval_prompt_es(user_prompt, model_answer)
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
    print("Starting Multi-Agent Fluency Evaluation (Approach 2: Council of 3) [SPANISH MODE]...")

    for folder in FOLDERS:
        # Usamos el prefijo 'es_' para los datasets en español
        folder_prefix = "valid" if folder == "validation" else folder
        in_file_name = f"es_{folder_prefix}_data_with_context.jsonl"

        input_path = os.path.join(DATASETS_DIR, folder, in_file_name)
        output_path = os.path.join(DATASETS_DIR, folder, OUTPUT_FILENAME)

        if not os.path.exists(input_path):
            print(f"Skipping {folder}: {in_file_name} not found.")
            continue

        print(f"\nProcessing Folder: {folder}")
        print(f"Reading: {in_file_name}")
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
                fluency_verdict = evaluate_fluency_with_council_es(user_prompt, output_text)

                # Construct Output (Fluency is evaluated, Factuality hardcoded 'n')
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
    print("All Spanish folders processed for Fluency Approach 2.")

if __name__ == "__main__":
    main()