import openai
import os
import json
import random
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
with open("credentials/config.json", "r") as f:
    config = json.load(f)
openai.api_key = config.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY")

model = ChatOpenAI(model='gpt-4', temperature=0)
output_parser = StrOutputParser()

def generate_template_h(output, answer):
    template = f"""
    당신은 아웃풋과 답안 두가지를 비교하여, 아웃풋이 답안과 얼마나 유사한지 평가해야 합니다.

    ### 아웃풋: "{output}"
    ### 답안: "{answer}"

    위의 아웃풋과 답안이 얼마나 유사한지를 비교하여 0점부터 5점 사이의 점수를 매겨주세요.
    예를 들어 "I don't have enough information to provide an answer to this query." 이런 아웃풋처럼 요구한 답변을 아무것도 생성하지 못하였다면 0점을 주세요.
    완벽하게 일치하지 않는다고 하여도 답안의 키워드가 정확하게 포함되어 있다면 높은 점수를 주세요.
    아웃풋이 여러개여도 답안과 일치하는게 존재한다면 어느정도 점수를 주세요.

    점수만 숫자로 출력하세요.
    """
    return template

def generate_template_g(output, answer):
    template = f"""
    당신은 아웃풋과 답안 두가지를 비교하여, 아웃풋과 답안의 풀이 과정이 얼마나 유사한지 평가해야 합니다.

    ### 아웃풋: "{output}"
    ### 답안: "{answer}"

    위의 아웃풋과 답안이 얼마나 유사한지를 비교하여 0점부터 5점 사이의 점수를 매겨주세요.
    예를 들어 "I don't have enough information to provide an answer to this query." 이런 아웃풋처럼 요구한 답변을 아무것도 생성하지 못하였다면 0점을 주세요.
    결과가 정확히 일치하지 않아도 풀이 과정이 유사하다면 그에 상응하는 점수를 주세요 
    아웃풋이 여러개여도 답안과 일치하는게 존재한다면 어느정도 점수를 주세요

    점수만 숫자로 출력하세요.
    """
    return template

def llm_similarity(output,answer,inference_type, mode):
    if mode == 1:
        prompt = generate_template_h(output, answer)
    else:
        prompt = generate_template_g(output, answer)
        
    response = model.invoke(prompt) 
    response_text = response.content.strip() 
    try:
        score = float(response_text)
        return score #output, answer, score, inference_type
    except ValueError:
        score = 0.0 
        return score #output, answer, score, inference_type