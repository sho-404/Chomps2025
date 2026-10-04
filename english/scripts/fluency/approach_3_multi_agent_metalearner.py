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
DATASETS_DIR = os.path.join(BASE_DIR, "../../../../datasets/english")

# Folders
TRAIN_FOLDER = "train"
INFERENCE_FOLDERS = ["validation", "test"]

# Output filename
OUTPUT_FILENAME = "approach_3_answers_fluency.jsonl"

# The label we are trying to predict (Fluency Mistakes)
LABEL_KEY = "has_fluency_mistakes" 

# Azure Config
ENDPOINT = "https://gpt-shroom.cognitiveservices.azure.com/"
SUBSCRIPTION_KEY = ""

# Judges Configuration
JUDGES = [
    {
        "name": "gpt-5-mini",
        "deployment": "gpt-5-mini",
        "api_version": "2025-04-01-preview"
    },
    {
        "name": "gpt-4.1-mini",
        "deployment": "gpt-4-1-mini-2025-04-14-ft-ee77b52ad26040dfa628828f1e88b44b",
        "api_version": "2024-12-01-preview"
    },
    {
        "name": "gpt-4.1-nano",
        "deployment": "1-nano-2025-04-14-clean_ft_english",
        "api_version": "2024-12-01-preview"
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

def get_fluency_eval_prompt(user_prompt, model_answer):
    """Keeps the prompt from single agent fluency approach as requested."""
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

def query_judge(judge_config, eval_prompt):
    """Returns 1 for 'y' (mistake), 0 for 'n' (no mistake)."""
    client = judge_config["client"]
    deployment = judge_config["deployment"]
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a strict binary evaluator. You only output 'y' or 'n'."},
                {"role": "user", "content": eval_prompt}
            ],
            max_completion_tokens=100, 
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
    """Runs the 3 judges on a dataset specifically for Fluency features."""
    folder_prefix = "valid" if folder_name == "validation" else folder_name
    data_file = f"en_{folder_prefix}_data_with_context.jsonl"
    path = os.path.join(DATASETS_DIR, folder_name, data_file)

    if not os.path.exists(path):
        print(f"File not found: {path}")
        return [], []

    print(f"--- Processing {folder_name} (Fluency Judge Features) ---")
    
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

            eval_prompt = get_fluency_eval_prompt(prompt, output_text)
            
            row_votes = []
            print(f"[{i+1}/{total}] ID: {idx} -> Fluency Judges:", end=" ")
            
            for j_conf in JUDGES:
                vote = query_judge(j_conf, eval_prompt)
                row_votes.append(vote)
                print("Y" if vote == 1 else "N", end="")
            
            print("") 
            metadata.append(data)
            features.append(row_votes)

    return metadata, np.array(features)

def load_ground_truth(folder_name):
    """Loads the label file mapping index -> fluency label (1 or 0)"""
    label_file = f"en_{folder_name}_label.jsonl"
    path = os.path.join(DATASETS_DIR, folder_name, label_file)
    
    gt_map = {}
    if not os.path.exists(path):
        print(f"Label file not found: {path}")
        return gt_map

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            val = 1 if data.get(LABEL_KEY) == 'y' else 0
            gt_map[data['index']] = val
    return gt_map

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Starting Fluency Evaluation (Approach 3: Meta-Learner)...")
    print(f"Target Label: {LABEL_KEY}\n")

    # ---------------------------------------------------------
    # PHASE 1: TRAINING THE META-LEARNER
    # ---------------------------------------------------------
    print("=== PHASE 1: TRAINING ON FLUENCY LABELS ===")
    
    train_meta, X_train = process_dataset_features(TRAIN_FOLDER)
    
    if len(train_meta) == 0:
        print("Training data empty or missing. Aborting.")
        return

    gt_map = load_ground_truth(TRAIN_FOLDER)
    
    y_train = []
    final_X_train = []
    
    for i, meta in enumerate(train_meta):
        idx = meta['index']
        if idx in gt_map:
            final_X_train.append(X_train[i])
            y_train.append(gt_map[idx])
    
    final_X_train = np.array(final_X_train)
    y_train = np.array(y_train)

    print(f"Training samples aligned: {len(y_train)}")

    # Train Logistic Regression specifically for Fluency detection patterns
    clf = LogisticRegression(random_state=42)
    clf.fit(final_X_train, y_train)

    print(f"Fluency weights: {clf.coef_}")
    print("Interpretation: Judges with higher weights are better at spotting word-salad/syntax issues.")

    # ---------------------------------------------------------
    # PHASE 2: INFERENCE
    # ---------------------------------------------------------
    print("\n=== PHASE 2: INFERENCE ===")
    
    for folder in INFERENCE_FOLDERS:
        output_path = os.path.join(DATASETS_DIR, folder, OUTPUT_FILENAME)
        meta_data, X_test = process_dataset_features(folder)
        
        if len(meta_data) == 0:
            continue

        predictions = clf.predict(X_test)

        print(f"Writing Fluency results to {OUTPUT_FILENAME}...")
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for i, data in enumerate(meta_data):
                pred_label = 'y' if predictions[i] == 1 else 'n'
                
                result_obj = {
                    "index": data['index'],
                    "has_fluency_mistakes": pred_label, # Calculated by Meta-Learner
                    "has_factual_mistakes": "n"        # Hardcoded
                }
                
                json.dump(result_obj, outfile)
                outfile.write('\n')
        
        print(f"Finished {folder}.\n")

    print("="*60)
    print("Approach 3 (Fluency) Completed Successfully.")

if __name__ == "__main__":
    main()