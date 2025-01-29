import os
import json
import openai
import random
from llm_scoring import *
from em_scoring import *
from mean import *
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from collections import defaultdict

with open("credentials/config.json", "r") as f:
    config = json.load(f)
openai.api_key = config.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY")

model = ChatOpenAI(model='gpt-4', temperature=0)
output_parser = StrOutputParser()

def random_shuffle(data):
    all_items = []
    for inference_type, items in data.items():
        for item in items:
            item["inference_type"] = inference_type 
            all_items.append(item)

    random.shuffle(all_items)
    return all_items

def evaluate_similarity(data,mode):
    all_items = random_shuffle(data)
    results = []
    em_count_dict = defaultdict(int)
    for item in all_items:
        output = item["short_output"]
        answer = item["answer"]
        inference_type = item["inference_type"]

        em_score = exact_match_score(output,answer)
        if em_score :
            score = 5
            em_count_dict[inference_type] += 1
        else :
            if mode == 1:
                output = item["detailed_output"]
            else:
                answer = item["detailed_answer"]
                output = item["detailed_output"]
            score = llm_similarity(output,answer,inference_type,mode)
        results.append({
            "output": output,
            "answer": answer,
            "score": score,
            "inference_type": inference_type
        })
        print(f"Processed: Output: {output}, Answer: {answer}, Score: {score}, Type: {inference_type}")

    return results, em_count_dict

def save_results_to_file(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

def main():
    data_name, mode = "hotpot", 1
    # data_name, mode = "gsm8k", 2
    input_file = f"results/{data_name}_results.json"
    output_file = f"results/{data_name}_scoring_results.json"
    em_count_output_file = f"results/{data_name}_em_count_results.json"

    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    results, em_count_dict = evaluate_similarity(data, mode)

    save_results_to_file(results, output_file)
    save_results_to_file(em_count_dict, em_count_output_file)
    print(f"Results saved to {output_file}")

    print_average_scores(output_file)

if __name__ =="__main__":
    main()