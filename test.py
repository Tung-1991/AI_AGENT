# test.py
from llama_cpp import Llama
import os

llm = Llama(
    model_path=os.path.expanduser(
        "~/AIagent/models/llm/llama-3-13b-instruct.Q4_K_M.gguf"
    ),
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False          # tắt log
)

messages = [
    {"role": "system",
     "content": "Tôi tên là tùng làm ở Vietinbank, thích kiếm tiền đầu tư COIN và quen nhiều cô gái xinh đẹp, bạn có lời nào cho tôi về công việc hay kiểm tiền ko."},
    {"role": "user", "content": "Xin chào!"}
]

resp = llm.create_chat_completion(
    messages=messages,
    max_tokens=1280,
    temperature=0.4,
    top_p=0.9,
    top_k=40
)

print(resp["choices"][0]["message"]["content"].strip())
