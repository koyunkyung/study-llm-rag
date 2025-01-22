import openai
import json
from typing import Optional
from config.logging import logger

# Load API key from config.json
try:
    with open("credentials/config.json", "r") as f:
        config = json.load(f)
    openai.api_key = config.get("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY is missing in config.json")
except (FileNotFoundError, ValueError) as e:
    logger.error(f"Error loading OpenAI API key: {e}")
    raise RuntimeError("Failed to load OpenAI API key. Please check config.json.")

def generate(prompt: str) -> Optional[str]:
    """
    Generates a response from OpenAI GPT model.

    Args:
        prompt (str): The input prompt for the model.

    Returns:
        Optional[str]: The generated response, or None if an error occurs.
    """
    try:
        logger.info("Generating response from OpenAI")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response['choices'][0]['message']['content']
    except openai.error.InvalidRequestError as e:
        logger.error(f"Invalid Request Error: {e}")
        return None
    except openai.error.AuthenticationError as e:
        logger.error(f"Authentication Error: {e}")
        return None
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API Error: {e}")
        return None
