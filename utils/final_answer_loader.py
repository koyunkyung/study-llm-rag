import os
import json

OUTPUT_DIR = "./data/output/"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "all_results.json")

def extract_final_answer(file_content: str) -> str:
    final_answers = []
    for line in file_content.splitlines():  # 모든 라인 탐색
        if line.startswith("Final Answer:"):
            final_answer = line.split("Final Answer:")[-1].strip()
            final_answers.append(final_answer)
    return final_answers


def load_all_final_answers(output_dir: str) -> dict:
    final_answers = {}

    for file_name in os.listdir(output_dir):
        if file_name.endswith("_trace.txt"):
            template_name = file_name.replace("_trace.txt", "")  # 템플릿 이름
            file_path = os.path.join(output_dir, file_name)

            with open(file_path, "r") as file:
                file_content = file.read()

            answers = extract_final_answer(file_content)
            if answers:
                final_answers[template_name] = answers

    return final_answers


def save_final_answers_to_json(final_answers: dict, results_file: str):
    with open(results_file, "w") as file:
        json.dump(final_answers, file, indent=4)
    print(f"All results saved to {results_file}")

if __name__ == "__main__":
    final_answers = load_all_final_answers(OUTPUT_DIR)
    save_final_answers_to_json(final_answers, RESULTS_FILE)
