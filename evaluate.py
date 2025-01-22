import json
import os

OUTPUT_DIR = "./data/output/"
RESULTS_DIR = "./data/results/"
ALL_RESULTS_FILE = os.path.join(OUTPUT_DIR, "all_results.json")

ACCURACY_FILE = os.path.join(RESULTS_DIR, "accuracy_scores.json")

# 각각의 프롬프트에 해당하는 Final Answer들 불러오기
def load_all_results(results_file: str) -> dict:
    with open(results_file, "r") as file:
        return json.load(file)



### TODO:모델 답변 평가 기준 정해야 함!!! ### 


def evaluate_accuracy(final_answers: dict, ground_truths: dict) -> dict:
    accuracy_scores = {}

    for template, answer in final_answers.items():
        ground_truth = ground_truths.get(template, "")
        accuracy_scores[template] = 1 if answer == ground_truth else 0

    return accuracy_scores



# 각각의 프롬프트에 해당하는 평가 점수 .json 파일로 저장 
def save_results(results: dict, file_path: str):
    with open(file_path, "w") as file:
        json.dump(results, file, indent=4)
    print(f"Results saved to {file_path}")

if __name__ == "__main__":
    # Load all results
    final_answers = load_all_results(ALL_RESULTS_FILE)

    # Define ground truths (replace with your actual ground truth answers)
    ground_truths = {
        "react": "Brazil has won the most FIFA World Cup titles."
        }

    # Evaluate accuracy
    accuracy_scores = evaluate_accuracy(final_answers, ground_truths)
    save_results(accuracy_scores, ACCURACY_FILE)