import os
import json

INPUT_DIR = "./data/input/"
OUTPUT_DIR = "./data/output/"
RESULTS_DIR = "./results/"


def load_dataset(dataset_name):
    input_file = os.path.join(INPUT_DIR, f"{dataset_name}.json")
    with open(input_file, "r") as f:
        return json.load(f)

def extract_final_answers(file_content):
    final_answers = []
    for line in file_content.splitlines():
        if line.startswith("Final Answer:"):
            answer = line.replace("Final Answer:", "").strip()
            if answer.startswith("Final Answer:"):
                answer = answer.replace("Final Answer:", "").strip()
            final_answers.append(answer)
    return final_answers

def load_final_answers_by_dataset(dataset_name):
    results = {}
    original_data = load_dataset(dataset_name)
    question_to_answer = {item["question"]: item["answer"] for item in original_data}
    
    for file_name in os.listdir(OUTPUT_DIR):
        if file_name.startswith(f"{dataset_name}_") and file_name.endswith(".txt"):
            template_name = file_name.replace(f"{dataset_name}_", "").replace(".txt", "")
            file_path = os.path.join(OUTPUT_DIR, file_name)
            
            with open(file_path, "r") as f:
                file_content = f.read()
                outputs = extract_final_answers(file_content)
                paired_results = []
                for i, output in enumerate(outputs):
                    if i < len(original_data):
                        question = original_data[i]["question"]
                        true_answer = question_to_answer[question]
                        paired_results.append({
                            "output": output,
                            "answer": true_answer
                        })
                
                if paired_results:
                    results[template_name] = paired_results

    results_file = os.path.join(RESULTS_DIR, f"{dataset_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Paired results saved to {results_file}")
    
    return results

if __name__ == "__main__":
    datasets = ["hotpot", "gsm8k"]
    for dataset in datasets:
        load_final_answers_by_dataset(dataset)