import os
import json
from openai import AzureOpenAI

# ==========================================
# CONFIGURATION & CREDENTIALS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "../../../../datasets/spanish")

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
# PROMPT DEFINITION (FLUENCY EVALUATOR - SPANISH)
# ==========================================

def evaluate_fluency_es(user_prompt, model_answer):
    """
    Evalúa si la respuesta del modelo tiene sentido, es coherente y gramaticalmente correcta.
    
    Retorna:
    'y' -> Sí, TIENE errores de fluidez (sin sentido, roto, incoherente).
    'n' -> No, NO tiene errores de fluidez (fluye bien y es natural).
    """
    
    if not model_answer or len(str(model_answer).strip()) < 1:
        print("¡RESPUESTA DEL MODELO VACÍA!")
        return "y"

    # Evaluation Prompt in Spanish
    eval_prompt = f"""
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

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Eres un evaluador binario estricto. Solo respondes con 'y' o 'n'."},
                {"role": "user", "content": eval_prompt}
            ],
            max_completion_tokens=16384, 
            model=DEPLOYMENT,
        )
        result = response.choices[0].message.content.strip().lower()
        
        if 'y' in result: return 'y'
        if 'n' in result: return 'n'
        
        return 'y' 

    except Exception as e:
        print(f"Error durante la evaluación: {e}")
        return "y"

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Starting Fluency Evaluation (Prompt-based) [SPANISH MODE]...")

    for folder in FOLDERS:
        # Determine Input Filename (using 'es_' prefix for Spanish datasets)
        folder_prefix = "valid" if folder == "validation" else folder
        in_file_name = f"es_{folder_prefix}_data_with_context.jsonl"

        input_path = os.path.join(DATASETS_DIR, folder, in_file_name)
        output_path = os.path.join(DATASETS_DIR, folder, OUTPUT_FILENAME)

        if not os.path.exists(input_path):
            print(f"Skipping {folder}: {in_file_name} not found.")
            continue

        print(f"\nProcessing Folder: {folder}")
        print(f"Reading: {in_file_name}")
        print(f"Writing: {output_path}")
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
                
                # Run Spanish Fluency Evaluation
                fluency_mistake_flag = evaluate_fluency_es(user_prompt, output_text)

                # Construct Output Object
                result_obj = {
                    "index": idx,
                    "has_fluency_mistakes": fluency_mistake_flag, 
                    "has_factual_mistakes": "n" 
                }

                json.dump(result_obj, outfile)
                outfile.write('\n')

                print(f"[{i+1}/{total}] ID: {idx} | Fluidez Incorrecta? {fluency_mistake_flag.upper()}", end="\r")

        print(f"\nCompleted {folder}.\n")

    print("="*60)
    print("All Spanish folders processed successfully.")

if __name__ == "__main__":
    main()