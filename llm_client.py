# FILE: llm_client.py
import os
import requests
import json
import logging

LLM_API_ENDPOINT = os.getenv("LLM_API_ENDPOINT", "http://localhost:8000/v1/chat/completions")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama-3-13b-instruct.Q4_K_M.gguf") 

def call_llm(messages: list, temperature: float = 0.4, max_tokens: int = 2048) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        response = requests.post(
            LLM_API_ENDPOINT,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=180
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"Lỗi khi gọi LLM: {e}")
        return "Xin lỗi, tôi đang gặp sự cố khi kết nối tới mô hình ngôn ngữ."
