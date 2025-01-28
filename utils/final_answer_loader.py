import os
import json

INPUT_DIR = "./data/input/"
OUTPUT_DIR = "./data/output/"
RESULTS_DIR = "./results/"

def load_dataset(dataset_name):
    input_file = os.path.join(INPUT_DIR, f"{dataset_name}.json")
    with open(input_file, "r") as f:
        return json.load(f)

def extract_answers(section):
    detailed_output = None
    short_output = None
    
    # 에러 메시지나 불충분한 응답을 나타내는 패턴들
    error_patterns = [
        "I don't have enough information",
        "Invalid response format",
        "I encountered an error",
        "I encountered an unexpected error",
        "Unable to find",
        "I couldn't find",
        "No results found",
        "Insufficient information",
        "Not enough information",
        "Unknown",
        "Information not available",
        "Information unavailable",
        "Insufficient data",
        "Information not provided",
        "Unable to determine",
        "could not",
        "couldn't",
        "wasn't able",
        "was not able",
        "No detailed answer",
        "No answer",
        "unable to"
    ]

    lines = section.strip().split('\n')
    for i, line in enumerate(lines):
        if line.startswith("Final Answer:"):
            detailed_output = line.replace("Final Answer:", "").strip()
            # Final Answer가 에러 패턴을 포함하는지 검사
            if any(pattern.lower() in detailed_output.lower() for pattern in error_patterns):
                return detailed_output, "None"
            
            if i + 1 < len(lines) and lines[i + 1].startswith("Short Answer:"):
                short_output = lines[i + 1].replace("Short Answer:", "").strip()
                # Short Answer가 에러 패턴을 포함하는지 검사
                if any(pattern.lower() in short_output.lower() for pattern in error_patterns):
                    short_output = "None"

    detailed_output = detailed_output if detailed_output else "No detailed answer provided"
    short_output = short_output if short_output else "None"
    return detailed_output, short_output

def load_final_answers_by_dataset(dataset_name):
    results = {}
    original_data = load_dataset(dataset_name)
    question_to_answer = {item["question"]: item["answer"] for item in original_data}
    
    for file_name in os.listdir(OUTPUT_DIR):
        if file_name.startswith(f"{dataset_name}_") and file_name.endswith(".txt"):
            template_name = file_name.replace(f"{dataset_name}_", "").replace(".txt", "")
            file_path = os.path.join(OUTPUT_DIR, file_name)
            
            with open(file_path, "r") as f:
                content = f.read()
                paired_results = []
                current_question = ""
                current_section = []
                
                for line in content.split('\n'):
                    if line.startswith("Question"):
                        if current_section:
                            detailed_output, short_output = extract_answers('\n'.join(current_section))
                            if current_question in question_to_answer:
                                paired_results.append({
                                    "answer": question_to_answer[current_question],
                                    "short_output": short_output,
                                    "detailed_output": detailed_output
                                })
                        current_question = next((q for q in question_to_answer if q in line), "")
                        current_section = []
                    current_section.append(line)
                
                if current_section:
                    detailed_output, short_output = extract_answers('\n'.join(current_section))
                    if current_question in question_to_answer:
                        paired_results.append({
                            "answer": question_to_answer[current_question],
                            "short_output": short_output,
                            "detailed_output": detailed_output
                        })
                
                if paired_results:
                    results[template_name] = paired_results

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{dataset_name}_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    return results

if __name__ == "__main__":
    for dataset in ["hotpot", "gsm8k"]:
        load_final_answers_by_dataset(dataset)