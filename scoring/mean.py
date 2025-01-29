import json
from collections import defaultdict

def calculate_average_scores(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
  
    scores_by_type = defaultdict(list)

    for item in data:
        inference_type = item["inference_type"]
        score = item["score"]
        scores_by_type[inference_type].append(score)

    average_scores = {}
    for inference_type, scores in scores_by_type.items():
        average_scores[inference_type] = sum(scores) / len(scores)

    return average_scores

def print_average_scores(file_path):
    try:
        average_scores = calculate_average_scores(file_path)
        print("\n추론 방식별 평균 스코어:")
        for inference_type, avg_score in average_scores.items():
            print(f"{inference_type}: {avg_score:.2f}")
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다. 경로를 확인하세요.")
    except json.JSONDecodeError:
        print("올바른 JSON 파일이 아닙니다. 파일 내용을 확인하세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")


if __name__ =="__main__":
    data_name = "gsm8k" #hotpot
    file_path = f"results/{data_name}_scoring_results.json" 
    print_average_scores(file_path)