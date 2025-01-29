import json
import re

file_path = "results/gsm8k_results_prev.json"  
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)  

processed_data = {}

def process_answer(entry):
    """'answer' 필드를 \n#### 기준으로 분리하여 detailed_answer와 answer 저장"""
    text = entry.get("answer", "")
    match = re.search(r"\n####\s*(.*)", text)  
    if match:
        answer = match.group(1).strip()  # #### 이후 부분 (단답형 답)
        detailed_answer = text[:match.start()].strip()  # #### 이전 부분 (설명)
    else:
        answer = text.strip()
        detailed_answer = text.strip()

    entry["detailed_answer"] = detailed_answer
    entry["answer"] = answer  # 기존 answer 필드 업데이트

    return entry

for key, entries in data.items():
    processed_data[key] = [process_answer(entry) for entry in entries]

# 결과 JSON 저장
output_path = "results/gsm8k_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=4, ensure_ascii=False)

print(f"변환 완료! 저장된 파일: {output_path}")