from openai import OpenAI
import json
from typing import Optional
from config.logging import logger

try:
    with open("credentials/config.json", "r") as f:
        config = json.load(f)
    client = OpenAI(api_key=config.get("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY is missing in config.json")
except (FileNotFoundError, ValueError) as e:
    logger.error(f"Error loading OpenAI API key: {e}")
    raise RuntimeError("Failed to load OpenAI API key. Please check config.json.")

def generate(prompt: str) -> Optional[str]:
    try:
        logger.info("Generating response from OpenAI")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API Error: {e}")
        return None