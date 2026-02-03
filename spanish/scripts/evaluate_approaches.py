import os
import json
import glob

def load_jsonl_map(filepath, key_name):
    """
    Loads a JSONL file and returns a dictionary mapping 'index' to the specified key.
    key_name: 'has_factual_mistakes' or 'has_fluency_mistakes'
    """
    data_map = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                data_map[entry['index']] = entry[key_name]
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Error loading {filepath}: {e}")
    return data_map

def calculate_metrics(prediction_file, ground_truth_map, key_name):
    """
    Calculates Accuracy, Precision, Recall, and F1.
    Positive class ('y') = mistake present.
    """
    tp = fp = tn = fn = 0
    
    with open(prediction_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pred_entry = json.loads(line)
            idx = pred_entry.get('index')
            pred_val = pred_entry.get(key_name)
            
            if idx in ground_truth_map:
                gt_val = ground_truth_map[idx]
                
                if pred_val == 'y' and gt_val == 'y':
                    tp += 1
                elif pred_val == 'y' and gt_val == 'n':
                    fp += 1
                elif pred_val == 'n' and gt_val == 'n':
                    tn += 1
                elif pred_val == 'n' and gt_val == 'y':
                    fn += 1

    total = tp + fp + tn + fn
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return accuracy, precision, recall, f1

def process_category(folder_path, label_path, category_type):
    """
    Handles evaluation for 'factuality' or 'fluency'.
    """
    key_map = {
        'factuality': 'has_factual_mistakes',
        'fluency': 'has_fluency_mistakes'
    }
    key = key_map[category_type]
    
    ground_truth = load_jsonl_map(label_path, key)
    # Searches for approach_*_answers_factuality.jsonl or approach_*_answers_fluency.jsonl
    search_pattern = os.path.join(folder_path, f"approach_*_answers_{category_type}.jsonl")
    approach_files = sorted(glob.glob(search_pattern))
    
    if not approach_files:
        return

    results_lines = ["RESULTS", ""]
    for app_file in approach_files:
        # Extract name (e.g., approach_1)
        approach_name = os.path.basename(app_file).split(f'_answers_{category_type}')[0]
        acc, prec, rec, f1 = calculate_metrics(app_file, ground_truth, key)
        
        results_lines.append(f"--- {approach_name} ---")
        results_lines.append(f"Accuracy:  {acc:.4f}")
        results_lines.append(f"Precision: {prec:.4f}")
        results_lines.append(f"Recall:    {rec:.4f}")
        results_lines.append(f"F1 Score:  {f1:.4f}")
        results_lines.append("")

    output_path = os.path.join(folder_path, f"{category_type}_results.txt")
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write('\n'.join(results_lines).strip())
    print(f"  -> Saved Spanish {category_type} results to {output_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, '..', 'datasets')
    
    # Updated to Spanish (es_) label files
    folders = {
        'test': 'es_test_label.jsonl',
        'train': 'es_train_label.jsonl',
        'validation': 'es_valid_label.jsonl'
    }

    for folder_name, label_filename in folders.items():
        folder_path = os.path.join(datasets_dir, folder_name)
        label_path = os.path.join(folder_path, label_filename)
        
        if not os.path.exists(label_path):
            print(f"Skipping Spanish {folder_name}: Label file missing ({label_path})")
            continue
            
        print(f"Processing Spanish folder: {folder_name}...")
        
        # Runs evaluation for both types
        process_category(folder_path, label_path, 'factuality')
        process_category(folder_path, label_path, 'fluency')

if __name__ == "__main__":
    main()