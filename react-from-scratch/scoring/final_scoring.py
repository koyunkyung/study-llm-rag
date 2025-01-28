import os
import json
import openai
import random
from llm_scoring import *
from em_scoring import *
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

with open('.key.txt', 'r', encoding='utf-8') as file:
    api_key = file.read().strip().replace('\ufeff', '')
    os.environ["OPENAI_API_KEY"] = api_key

openai.api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model='gpt-4', temperature=0)
output_parser = StrOutputParser()

def random_shuffle(data):
    all_items = []
    for inference_type, items in data.items():
        for item in items:
            item["inference_type"] = inference_type 
            all_items.append(item)
    return random.shuffle(all_items) 

def evaluate_similarity(data):
    all_items = random_shuffle(data)
    results = []
    for item in all_items:
        output = item["output"]
        answer = item["answer"]
        inference_type = item["inference_type"]

        em_score = exact_match_score(output,answer)
        if em_score :
            results.append({
                "output": output,
                "answer": answer,
                "score": 5,
                "inference_type": inference_type
            })
        else :
            output, answer, score, inference_type = llm_similarity(item)
            results.append({
                "output": output,
                "answer": answer,
                "score": score,
                "inference_type": inference_type
            })
        print(f"Processed: Output: {output}, Answer: {answer}, Score: {score}, Type: {inference_type}")

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

if __name__ =="__main__":
    main()
