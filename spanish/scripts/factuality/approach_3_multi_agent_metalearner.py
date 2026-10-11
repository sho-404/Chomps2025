import os
import json
import numpy as np
from openai import AzureOpenAI
from sklearn.linear_model import LogisticRegression

# ==========================================
# CONFIGURATION & CREDENTIALS
# ==========================================

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "../../../../datasets/spanish")

# Folders
TRAIN_FOLDER = "train"
INFERENCE_FOLDERS = ["validation", "test"]

# Output filename
OUTPUT_FILENAME = "approach_3_answers_factuality.jsonl"

# The label we are trying to predict (Hallucinations = Factual Mistakes)
LABEL_KEY = "has_factual_mistakes" 

# Azure Config
ENDPOINT = "https://gpt-shroom.cognitiveservices.azure.com/"
SUBSCRIPTION_KEY = os.environ["AZURE_OPENAI_API_KEY"]  # set in the environment; never commit keys

# Judges Configuration (SPANISH CONFIGURATION)
JUDGES = [
    {
        "name": "gpt-5-mini",
        "deployment": "gpt-5-mini",
        "api_version": "2025-04-01-preview"
    },
    {
        "name": "gpt-4.1-mini-spanish",
        "deployment": "gpt-4-1-mini-2025-04-14-ft-spanish",
        "api_version": "2025-01-01-preview"
    },
    {
        "name": "gpt-4.1-nano-spanish",
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

def get_eval_prompt(user_prompt, context, model_answer):
    # SPANISH PROMPT
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

def query_judge(judge_config, eval_prompt):
    """Returns 1 for 'y' (mistake), 0 for 'n' (no mistake)."""
    client = judge_config["client"]
    deployment = judge_config["deployment"]
    
    try:
        response = client.chat.completions.create(
            messages=[
                # Spanish System Prompt
                {"role": "system", "content": "Eres un evaluador binario estricto. Solo respondes con 'y' o 'n'."},
                {"role": "user", "content": eval_prompt}
            ],
            # EXACT PARAMS PRESERVED
            max_completion_tokens=16384, 
            temperature=1.0,             
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment,
        )
        content = response.choices[0].message.content.strip().lower()
        
        if 'y' in content: return 1
        if 'n' in content: return 0
        
        return 1 # Fallback: assume mistake if unclear
    except Exception as e:
        print(f"Error {judge_config['name']}: {e}")
        return 1

def process_dataset_features(folder_name, is_training=False):
    """
    Runs the 3 judges on a dataset and returns:
    1. metadata (list of dicts with index, original data)
    2. features (matrix of shape [N_samples, 3_judges])
    """
    
    # Resolve file paths (SPANISH PREFIX: es_)
    folder_prefix = "valid" if folder_name == "validation" else folder_name
    data_file = f"es_{folder_prefix}_data_with_context.jsonl"
    path = os.path.join(DATASETS_DIR, folder_name, data_file)

    if not os.path.exists(path):
        print(f"File not found: {path}")
        return [], []

    print(f"--- Processing {folder_name} (Extracting Judge Features) ---")
    
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
            context = data.get('context') or ""

            eval_prompt = get_eval_prompt(prompt, context, output_text)
            
            # Gather votes from all 3 judges
            # 1 = 'y' (mistake), 0 = 'n' (ok)
            row_votes = []
            print(f"[{i+1}/{total}] ID: {idx} -> Judges:", end=" ")
            
            for j_conf in JUDGES:
                vote = query_judge(j_conf, eval_prompt)
                row_votes.append(vote)
                # Visual log: 1=Y, 0=N
                print("Y" if vote == 1 else "N", end="")
            
            print("") # Newline
            
            metadata.append(data)
            features.append(row_votes)

    return metadata, np.array(features)

def load_ground_truth(folder_name):
    """Loads the label file and returns a dict mapping index -> label (1 or 0)"""
    # SPANISH PREFIX: es_
    label_file = f"es_{folder_name}_label.jsonl"
    path = os.path.join(DATASETS_DIR, folder_name, label_file)
    
    gt_map = {}
    if not os.path.exists(path):
        print(f"Label file not found: {path}")
        return gt_map

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Map 'y' -> 1, 'n' -> 0
            val = 1 if data.get(LABEL_KEY) == 'y' else 0
            gt_map[data['index']] = val
    return gt_map

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Starting Factuality Evaluation (Approach 3: Meta-Learner) [SPANISH]...")
    print(f"Target Label: {LABEL_KEY}\n")

    # ---------------------------------------------------------
    # PHASE 1: TRAINING THE META-LEARNER
    # ---------------------------------------------------------
    print("=== PHASE 1: TRAINING ===")
    
    # 1. Get Features (Judge Votes) for Train Set
    train_meta, X_train = process_dataset_features(TRAIN_FOLDER, is_training=True)
    
    if len(train_meta) == 0:
        print("Training data empty or missing. Aborting.")
        return

    # 2. Get Ground Truth Labels
    gt_map = load_ground_truth(TRAIN_FOLDER)
    
    # 3. Align Features with Labels
    y_train = []
    final_X_train = []
    
    print("Aligning Training Data...")
    for i, meta in enumerate(train_meta):
        idx = meta['index']
        if idx in gt_map:
            final_X_train.append(X_train[i])
            y_train.append(gt_map[idx])
    
    final_X_train = np.array(final_X_train)
    y_train = np.array(y_train)

    print(f"Training samples aligned: {len(y_train)}")

    # 4. Train Model
    # We use Logistic Regression. It will learn weights for [Judge1, Judge2, Judge3]
    print("Training Meta-Learner (Logistic Regression)...")
    clf = LogisticRegression(random_state=42)
    clf.fit(final_X_train, y_train)

    # Show what it learned
    print(f"Model Coefficients (Weights): {clf.coef_}")
    print(f"Model Intercept: {clf.intercept_}")
    print("Interpretation: Higher weight = Judge is more trusted.")

    # ---------------------------------------------------------
    # PHASE 2: INFERENCE
    # ---------------------------------------------------------
    print("\n=== PHASE 2: INFERENCE ===")
    
    for folder in INFERENCE_FOLDERS:
        output_path = os.path.join(DATASETS_DIR, folder, OUTPUT_FILENAME)
        
        # 1. Get Features for Test/Val
        meta_data, X_test = process_dataset_features(folder)
        
        if len(meta_data) == 0:
            continue

        # 2. Predict using the trained meta-learner
        print(f"Predicting labels for {folder}...")
        predictions = clf.predict(X_test) # Returns 0s and 1s

        # 3. Write Results
        print(f"Writing results to {OUTPUT_FILENAME}...")
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for i, data in enumerate(meta_data):
                pred_label = 'y' if predictions[i] == 1 else 'n'
                
                result_obj = {
                    "index": data['index'],
                    "has_fluency_mistakes": "n", 
                    "has_factual_mistakes": pred_label
                }
                
                json.dump(result_obj, outfile)
                outfile.write('\n')
        
        print(f"Finished {folder}.\n")

    print("="*60)
    print("Approach 3 (Spanish) Completed Successfully.")

if __name__ == "__main__":
    main()