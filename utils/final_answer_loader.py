import os
import json

OUTPUT_DIR = "./data/output/"

def extract_final_answers(file_content: str) -> list:
    final_answers = []
    for line in file_content.splitlines():
        if line.startswith("Final Answer:"):
            answer = line.replace("Final Answer:", "").strip()
            if answer.startswith("Final Answer:"):
                answer = answer.replace("Final Answer:", "").strip()
            final_answers.append(answer)
    return final_answers

def load_final_answers_by_dataset(dataset_name: str) -> dict:
    results = {}
    
    for file_name in os.listdir(OUTPUT_DIR):
        if file_name.startswith(f"{dataset_name}_") and file_name.endswith(".txt"):
            template_name = file_name.replace(f"{dataset_name}_", "").replace(".txt", "")
            file_path = os.path.join(OUTPUT_DIR, file_name)
            
            with open(file_path, "r") as f:
                file_content = f.read()
                answers = extract_final_answers(file_content)
                if answers:
                    results[template_name] = answers
    
    results_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")
    return results

if __name__ == "__main__":
    datasets = ["hotpot", "gsm8k"]
    for dataset in datasets:
        load_final_answers_by_dataset(dataset)