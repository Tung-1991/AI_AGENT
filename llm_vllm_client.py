import requests
import json
import logging
import os

VLLM_ENDPOINT = "http://localhost:30888/v1/chat/completions"
VLLM_MODEL_NAME = "llama-3.3" 

logging.info(f"vLLM Client configured for endpoint: {VLLM_ENDPOINT}")
logging.info(f"Requesting model: {VLLM_MODEL_NAME}")

def call_llm(messages: list, temperature: float = 0.4, max_tokens: int = 4096) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9
    }

    try:
        response = requests.post(
            VLLM_ENDPOINT,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=300
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Connection error calling vLLM API ({VLLM_ENDPOINT}): {e}")
        return "Error: Could not connect to the large language model (vLLM)."
    except Exception as e:
        logging.error(f"An unexpected error occurred when calling vLLM: {e}")
        return "Error: An unexpected issue occurred while processing the request with vLLM."
