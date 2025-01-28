import openai
import os
import json
import random
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


with open('.key.txt', 'r', encoding='utf-8') as file:
    api_key = file.read().strip().replace('\ufeff', '')
    os.environ["OPENAI_API_KEY"] = api_key

openai.api_key = os.getenv("OPENAI_API_KEY")


model = ChatOpenAI(model='gpt-4', temperature=0)
output_parser = StrOutputParser()


def generate_template(output, answer):
    template = f"""
    당신은 아웃풋과 모범답안 두가지를 비교하여, 아웃풋이 모범답안과 얼마나 유사한지 평가해야 합니다.

    ### 아웃풋: "{output}"
    ### 답안: "{answer}"

    위의 아웃풋과 모범답안이 얼마나 유사한지를 비교하여 0점부터 5점 사이의 점수를 매겨주세요.
    예를 들어 "I don't have enough information to provide an answer to this query." 이런 아웃풋처럼 요구한 답변을 아무것도 생성하지 못하였다면 0점을 주세요.
    완벽하게 일치하지 않는다고 하여도 답안의 키워드가 정확하게 포함되어 있다면 높은 점수를 주세요.

    점수만 숫자로 출력하세요.
    """
    return template


def evaluate_similarity(data):
    all_items = []
    for inference_type, items in data.items():
        for item in items:
            item["inference_type"] = inference_type 
            all_items.append(item)
    random.shuffle(all_items)  


    results = []
    for item in all_items:
        output = item["output"]
        answer = item["answer"]
        inference_type = item["inference_type"]
        prompt = generate_template(output, answer)
        

        response = model.invoke(prompt) 
        response_text = response.content.strip() 
        try:
            score = float(response_text)
        except ValueError:
            score = 0.0 
        
        results.append({
            "output": output,
            "answer": answer,
            "score": score,
            "inference_type": inference_type
        })
        print(f"Processed: Output: {output}, Answer: {answer}, Score: {score}, Type: {inference_type}")
    return results


def save_results_to_file(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)


def main():
    input_file = "hotpot_results.json"
    output_file = "hotpot_scoring_results.json"
    

    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    results = evaluate_similarity(data)
    save_results_to_file(results, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
