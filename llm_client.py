# FILE: llm_client.py

import requests
import json
import logging
import os

LOCAL_API_ENDPOINT = "http://localhost:8000/v1/chat/completions"
LOCAL_MODEL_NAME = "llama-3-13b-instruct.Q4_K_M.gguf"

logging.info(f"Local Llama.cpp Client configured for endpoint: {LOCAL_API_ENDPOINT}")
logging.info(f"Requesting model: {LOCAL_MODEL_NAME}")

def call_llm(messages: list, temperature: float = 0.4, max_tokens: int = 4096) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": LOCAL_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9
    }

    try:
        response = requests.post(
            LOCAL_API_ENDPOINT,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=180
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Connection error calling Llama.cpp API ({LOCAL_API_ENDPOINT}): {e}")
        return "Lỗi: Không thể kết nối tới server Llama.cpp local."
    except Exception as e:
        logging.error(f"An unexpected error occurred when calling Llama.cpp: {e}")
        return "Lỗi: Đã có sự cố không mong muốn xảy ra với server Llama.cpp."
