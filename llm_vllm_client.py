
# FILE: llm_vllm_client.py

import requests
import json
import logging
import os

# Cấu hình cứng cho môi trường A100 K8s
VLLM_ENDPOINT = "http://localhost:30888/v1/chat/completions"
# THAY THẾ BẰNG TÊN MODEL THỰC TẾ TRÊN vLLM CỦA BẠN
VLLM_MODEL_NAME = "meta-llama/Llama-3-70B-Instruct" 

logging.info(f"✅ vLLM Client được cấu hình để gọi tới: {VLLM_ENDPOINT}")
logging.info(f"✅ Sẽ yêu cầu model: {VLLM_MODEL_NAME}")

def call_llm(messages: list, temperature: float = 0.4, max_tokens: int = 4096) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
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
    except Exception as e:
        logging.error(f"Lỗi khi gọi vLLM: {e}")
        return "Xin lỗi, tôi đang gặp sự cố khi kết nối tới mô hình ngôn ngữ lớn (vLLM)."
