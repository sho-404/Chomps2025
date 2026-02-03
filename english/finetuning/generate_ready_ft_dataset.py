import json
import os

# Define file paths
input_file = 'finetune_dataset_raw.jsonl'
output_file = 'finetune_dataset_ready.jsonl'

def format_dataset():
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Processing {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        count = 0
        for line in infile:
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # Construct the User content clearly separating context, question, and answer
                user_content = (
                    f"Context: {data['context']}\n\n"
                    f"Question: {data['question']}\n\n"
                    f"Answer: {data['answer']}"
                )

                # Construct the Azure formatting object
                azure_entry = {
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are an evaluator reviewing an answer given by an LLM regarding a scientific paper question. Respond with 'y' if the answer has mistakes, or 'n' if there are no mistakes."
                        },
                        {
                            "role": "user", 
                            "content": user_content
                        },
                        {
                            "role": "assistant", 
                            "content": data['veredict']
                        }
                    ]
                }
                
                # Write to the new jsonl file
                outfile.write(json.dumps(azure_entry) + '\n')
                count += 1
                
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:50]}...")
            except KeyError as e:
                print(f"Skipping line missing key {e}: {line[:50]}...")

    print(f"Success! {count} lines formatted and saved to {output_file}")

if __name__ == "__main__":
    format_dataset()