# new.py - Bản tối ưu hoá hoàn chỉnh: RAG + Entity Memory + Auto Entity Buffer + LLM client + Session Memory + Auto-Summary

import os
import json
import requests
import subprocess
from datetime import datetime
from collections import Counter
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ===================== CONFIG =====================
TMP_DIR = os.path.join(os.getcwd(), "tmp")
HISTORY_DIR = os.path.join(TMP_DIR, "chat_history")
ENTITY_DB = os.path.join(TMP_DIR, "entities_all.json")
ENTITY_TMP = os.path.join(TMP_DIR, "entities.json")
AUTO_ENTITY_DIR = os.path.join(TMP_DIR, "auto")

MODEL_NAME = "llama-3.3"  
PORT = 30088

CTX_LIMIT = 8192 - 1024
TOKEN_PER_CHAR = 4
ENTITY_THRESHOLD = 3
MAX_ENTITIES = 5
KEYWORDS = ["nginxA", "Hoàng"]
AUTO_ENTITY_LIMIT = 10000

# ===================== GLOBAL STATE =====================
HISTORY = []
SESSION_NAME = None
embed_model, faiss_index, chunk_map = None, None, {}

# ===================== EMBED + FAISS =====================
def load_vector_engine():
    global embed_model, faiss_index, chunk_map
    try:
        embed_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder", device="cuda")
        faiss_index = faiss.read_index("./data/processed/faiss.index")
        with open("./data/processed/chunk_map.json", "r", encoding="utf-8") as f:
            chunk_map = json.load(f)
        print("✅ FAISS và embedding model đã sẵn sàng.")
    except Exception as e:
        print("❌ Lỗi khi load FAISS hoặc embedding model:", e)

def retrieve_context(query, top_k=3):
    if not embed_model or not faiss_index or not chunk_map:
        return ""
    vec = embed_model.encode([query])
    faiss.normalize_L2(vec)
    scores, ids = faiss_index.search(np.array(vec).astype("float32"), top_k)

    chunks, seen = [], set()
    for i, score in zip(ids[0], scores[0]):
        idx = str(i)
        if idx in chunk_map:
            chunk = chunk_map[idx]["chunk"].strip()
            if chunk not in seen and len(chunk) > 20 and not re.fullmatch(r"[\d\s\W]+", chunk):
                seen.add(chunk)
                chunks.append(chunk)
    return "\n".join(chunks)

# ===================== ENTITY MEMORY =====================
def extract_all_phrases(history):
    counter = Counter()
    for msg in history:
        if msg["role"] == "user":
            phrases = re.findall(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)', msg["content"])
            for ph in phrases:
                counter[ph] += 1
    return counter

def update_entity_memory(hot_entities):
    os.makedirs(TMP_DIR, exist_ok=True)
    with open(ENTITY_TMP, "w", encoding="utf-8") as f:
        json.dump(hot_entities, f, ensure_ascii=False, indent=2)

    if os.path.exists(ENTITY_DB):
        with open(ENTITY_DB, "r", encoding="utf-8") as f:
            all_ents = json.load(f)
    else:
        all_ents = {}

    for k, v in hot_entities.items():
        all_ents[k] = all_ents.get(k, 0) + v

    with open(ENTITY_DB, "w", encoding="utf-8") as f:
        json.dump(all_ents, f, ensure_ascii=False, indent=2)

def save_warm_entities(all_phrases):
    os.makedirs(AUTO_ENTITY_DIR, exist_ok=True)
    warm = {k: v for k, v in all_phrases.items() if 0 < v < ENTITY_THRESHOLD and k not in KEYWORDS}
    if warm:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(AUTO_ENTITY_DIR, f"auto_entity_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(warm, f, ensure_ascii=False, indent=2)
        print(f"🧾 Ghi {len(warm)} entity sơ bộ vào {path}")

def top_entities(n=MAX_ENTITIES):
    if not os.path.exists(ENTITY_DB):
        return ""
    with open(ENTITY_DB, "r", encoding="utf-8") as f:
        data = json.load(f)
    top = sorted(data.items(), key=lambda x: x[1], reverse=True)[:n]
    return "; ".join([f"{k} (x{v})" for k, v in top])

# ===================== LLM CLIENT =====================
def estimate_tokens(msgs):
    return sum(len(m["content"]) // TOKEN_PER_CHAR for m in msgs)

def trim_context(system_prompt):
    while estimate_tokens([system_prompt] + HISTORY) > CTX_LIMIT:
        for i, msg in enumerate(HISTORY):
            if msg["role"] == "user" and not any(k.lower() in msg["content"].lower() for k in KEYWORDS):
                del HISTORY[i]
                break
        else:
            HISTORY.pop(0)

def build_prompt(user_input):
    rag_text = retrieve_context(user_input)
    entity_text = top_entities()
    return {
        "role": "system",
        "content": (
            "Bạn là một trợ lý AI kỹ thuật. Bạn đang hỗ trợ một người dùng tên là Nguyễn Thanh Tùng, hiện là Systems Administrator tại CTG, có chuyên môn về vận hành hạ tầng và đầu tư crypto."
            " Luôn trả lời ngắn gọn, có ngữ cảnh, và không được nhận mình là người dùng."
            f"\n\n### Kiến thức kỹ thuật:\n{rag_text}"
            f"\n\n### Ngữ cảnh liên quan:\n{entity_text}"
        )
    }

def summarize_session(history):
    if not history:
        return ""
    content = "\n".join([f"{m['role']}: {m['content']}" for m in history if m["role"] in ("user", "assistant")][-10:])
    payload = {
        #"model": os.path.basename(MODEL_PATH),
        "model": MODEL_NAME,
		"messages": [
            {"role": "system", "content": "Tóm tắt ngắn gọn nội dung cuộc hội thoại sau (5 dòng hoặc ít hơn, rõ ràng, súc tích):"},
            {"role": "user", "content": content}
        ],
        "max_tokens": 200,
        "temperature": 0.3,
    }
    try:
        res = requests.post(
            f"http://localhost:{PORT}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )
        return res.json()["choices"][0]["message"]["content"].strip()
    except:
        return "Không tóm tắt được."

def call_llm(user_input):
    global HISTORY
    user_input = user_input.encode("utf-8", "ignore").decode("utf-8")
    HISTORY.append({"role": "user", "content": user_input})
    sys_prompt = build_prompt(user_input)
    trim_context(sys_prompt)

    payload = {
        #"model": os.path.basename(MODEL_PATH),
        "model": MODEL_NAME,
		"messages": [sys_prompt] + HISTORY,
        "max_tokens": 1024,
        "temperature": 0.4,
        "top_p": 0.9,
        "top_k": 40
    }

    try:
        res = requests.post(
            f"http://localhost:{PORT}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )
        res.raise_for_status()
        reply = res.json()["choices"][0]["message"]["content"].strip()
        HISTORY.append({"role": "assistant", "content": reply})

        all_phrases = extract_all_phrases(HISTORY)
        hot = {k: v for k, v in all_phrases.items() if v >= ENTITY_THRESHOLD and k not in KEYWORDS}
        update_entity_memory(hot)
        save_warm_entities(all_phrases)

        print(f"📊 Tokens hiện tại: ~{estimate_tokens([sys_prompt] + HISTORY)} / 8192")
        return reply
    except Exception as e:
        print("❌ Lỗi khi gọi LLM:", e)
        return "Xin lỗi, tôi đang gặp sự cố khi gọi mô hình."

# ===================== SESSION =====================
def save_history():
    if SESSION_NAME:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        filename = f"{SESSION_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(HISTORY_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(HISTORY, f, ensure_ascii=False, indent=2)

        summary = summarize_session(HISTORY)
        with open(path + ".summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)

        print(f"📁 Đã lưu lịch sử tại: {path}")
        print("📝 Tóm tắt session:")
        print(summary)

def select_history_file():
    if not os.path.exists(HISTORY_DIR):
        return None
    files = sorted(f for f in os.listdir(HISTORY_DIR) if f.endswith(".json"))
    if not files:
        return None

    print("📂 Danh sách file lịch sử:")
    for i, f in enumerate(files):
        print(f"  [{i}] {f}")
        summary_path = os.path.join(HISTORY_DIR, f + ".summary.txt")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as sf:
                print("     📝", sf.read().strip())

    choice = input("🔢 Chọn số file để load (hoặc Enter bỏ qua): ").strip()
    if not choice:
        return None
    try:
        idx = int(choice)
        with open(os.path.join(HISTORY_DIR, files[idx]), "r", encoding="utf-8") as f:
            print(f"🔁 Đang load lại session: {files[idx]}")
            return json.load(f)
    except:
        print("⚠️ Không đọc được file. Bỏ qua.")
        return None



# ===================== MAIN =====================
if __name__ == "__main__":
    load_vector_engine()
    SESSION_NAME = input("📋 Đặt tên cho phiên chat này: ").strip().replace(" ", "_")

    old_history = select_history_file()
    if old_history:
        use_summary = input("📋 Dùng tóm tắt thay vì full? (Y/n): ").strip().lower()
        if use_summary != "n":
            HISTORY = [{"role": "system", "content": "Tóm tắt lịch sử: " + " ".join(m["content"] for m in old_history if m["role"] == "user")[:300]}]
        else:
            HISTORY = old_history
        print(f"📈 Token ban đầu: ~{estimate_tokens(HISTORY)} / 8192")

    print("💬 Gõ 'exit' hoặc Ctrl+C để thoát.")
    while True:
        try:
            user_input = input("🧑‍💼 Ngài hỏi: ").strip()
            if user_input.lower() in ("exit", "quit"):
                save_history()
                break
            print("🧠 AI trả lời:", call_llm(user_input))
        except KeyboardInterrupt:
            print("\n⛔️ Kết thúc bởi người dùng. Đang lưu...")
            save_history()
            break
