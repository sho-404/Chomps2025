import os
import json
import numpy as np
from openai import AzureOpenAI
from sklearn.linear_model import LogisticRegression

# ==========================================
# CONFIGURATION & CREDENTIALS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "../../../../datasets/spanish")

TRAIN_FOLDER = "train"
INFERENCE_FOLDERS = ["validation", "test"]

# Output filename
OUTPUT_FILENAME = "approach_3_answers_fluency.jsonl"
LABEL_KEY = "has_fluency_mistakes" 

# Azure Config
ENDPOINT = "https://gpt-shroom.cognitiveservices.azure.com/"
SUBSCRIPTION_KEY = os.environ["AZURE_OPENAI_API_KEY"]  # set in the environment; never commit keys

# Judges Configuration (Updated for Spanish deployments)
JUDGES = [
    {
        "name": "gpt-5-mini",
        "deployment": "gpt-5-mini",
        "api_version": "2025-04-01-preview"
    },
    {
        "name": "gpt-4.1-mini-es",
        "deployment": "gpt-4-1-mini-2025-04-14-ft-spanish",
        "api_version": "2025-01-01-preview"
    },
    {
        "name": "gpt-4.1-nano-es",
        "deployment": "gpt-4-1-nano-2025-04-14-ft-spanish",
        "api_version": "2025-01-01-preview"
    }
]

# Initialize Clients
for judge in JUDGES:
    judge["client"] = AzureOpenAI(
        api_version=judge["api_version"],
        azure_endpoint=ENDPOINT,
        api_key=SUBSCRIPTION_KEY,
    )

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_fluency_eval_prompt_es(user_prompt, model_answer):
    """Spanish version of the fluency evaluation prompt."""
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

    [Model Answer]:
    {model_answer}
    """

def query_judge(judge_config, eval_prompt):
    """Returns 1 for 'y' (mistake), 0 for 'n' (no mistake)."""
    client = judge_config["client"]
    deployment = judge_config["deployment"]
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Eres un evaluador binario estricto. Solo respondes con 'y' o 'n'."},
                {"role": "user", "content": eval_prompt}
            ],
            max_completion_tokens=5, 
            temperature=1.0,             
            model=deployment,
        )
        content = response.choices[0].message.content.strip().lower()
        
        if 'y' in content: return 1
        if 'n' in content: return 0
        return 1 
    except Exception as e:
        print(f"Error {judge_config['name']}: {e}")
        return 1

def process_dataset_features(folder_name):
    """Processes datasets using the 'es_' prefix."""
    folder_prefix = "valid" if folder_name == "validation" else folder_name
    data_file = f"es_{folder_prefix}_data_with_context.jsonl"
    path = os.path.join(DATASETS_DIR, folder_name, data_file)

    if not os.path.exists(path):
        print(f"File not found: {path}")
        return [], []

    print(f"--- Procesando {folder_name} (Spanish Fluency Features) ---")
    
    metadata = []
    features = []

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total = len(lines)
        
        for i, line in enumerate(lines):
            data = json.loads(line)
            idx = data.get('index')
            prompt = data.get('prompt')
            output_text = data.get('output_text')

            eval_prompt = get_fluency_eval_prompt_es(prompt, output_text)
            
            row_votes = []
            print(f"[{i+1}/{total}] ID: {idx} -> Votos:", end=" ")
            
            for j_conf in JUDGES:
                vote = query_judge(j_conf, eval_prompt)
                row_votes.append(vote)
                print("Y" if vote == 1 else "N", end="")
            
            print("") 
            metadata.append(data)
            features.append(row_votes)

    return metadata, np.array(features)

def load_ground_truth_es(folder_name):
    """Loads Spanish label files (es_ prefix)."""
    label_file = f"es_{folder_name}_label.jsonl"
    path = os.path.join(DATASETS_DIR, folder_name, label_file)
    
    gt_map = {}
    if not os.path.exists(path):
        print(f"Label file not found: {path}")
        return gt_map

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Use 'y'/'n' logic consistent with the datasets
            val = 1 if data.get(LABEL_KEY) == 'y' else 0
            gt_map[data['index']] = val
    return gt_map

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Iniciando Evaluación de Fluidez [SPANISH] (Approach 3: Meta-Learner)...")

    # ---------------------------------------------------------
    # FASE 1: ENTRENAMIENTO
    # ---------------------------------------------------------
    print("=== FASE 1: ENTRENAMIENTO DEL META-LEARNER ===")
    
    train_meta, X_train = process_dataset_features(TRAIN_FOLDER)
    
    if len(train_meta) == 0:
        print("Sin datos de entrenamiento. Abortando.")
        return

    gt_map = load_ground_truth_es(TRAIN_FOLDER)
    
    y_train = []
    final_X_train = []
    
    for i, meta in enumerate(train_meta):
        idx = meta['index']
        if idx in gt_map:
            final_X_train.append(X_train[i])
            y_train.append(gt_map[idx])
    
    final_X_train = np.array(final_X_train)
    y_train = np.array(y_train)

    print(f"Muestras alineadas para entrenamiento: {len(y_train)}")

    clf = LogisticRegression(random_state=42)
    clf.fit(final_X_train, y_train)

    print(f"Pesos de los jueces (Coeficientes): {clf.coef_}")

    # ---------------------------------------------------------
    # FASE 2: INFERENCIA
    # ---------------------------------------------------------
    print("\n=== FASE 2: INFERENCIA ===")
    
    for folder in INFERENCE_FOLDERS:
        output_path = os.path.join(DATASETS_DIR, folder, OUTPUT_FILENAME)
        meta_data, X_test = process_dataset_features(folder)
        
        if len(meta_data) == 0:
            continue

        predictions = clf.predict(X_test)

        print(f"Guardando resultados en {OUTPUT_FILENAME}...")
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for i, data in enumerate(meta_data):
                pred_label = 'y' if predictions[i] == 1 else 'n'
                
                result_obj = {
                    "index": data['index'],
                    "has_fluency_mistakes": pred_label,
                    "has_factual_mistakes": "n" 
                }
                
                json.dump(result_obj, outfile)
                outfile.write('\n')
        
        print(f"Finalizado: {folder}.\n")

    print("="*60)
    print("Proceso completado con éxito.")

if __name__ == "__main__":
    main()