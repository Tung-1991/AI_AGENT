import os
import json
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from tqdm import tqdm

# ============ CẤU HÌNH ============
DATA_DIR = "./data/RAG"
PROCESSED_DIR = "./data/processed"
EMBED_MODEL_DIR = "/home/tungn/AIagent/models/embed"

CHUNK_SIZE = 500  # số từ mỗi chunk
CHUNK_OVERLAP = 50  # số từ trùng giữa các chunk

# ============ LOAD MODEL ============
print("📦 Đang load model embedding từ GPU...")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_DIR)
model = AutoModel.from_pretrained(EMBED_MODEL_DIR).cuda()
print("✅ Đã load model.")


# ============ TEXT UTILS ============
def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks


# ============ EMBED FUNC ============
def embed_texts(texts):
    embeddings = []
    batch_size = 8
    for i in tqdm(range(0, len(texts), batch_size), desc="🔍 Embedding chunks"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state[:, 0]
        embeddings.append(outputs.cpu().numpy())
    return np.vstack(embeddings)


# ============ CHẠY TOÀN BỘ ============
if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_chunks = []
    source_map = {}

    print("📂 Đang quét thư mục:", DATA_DIR)
    file_list = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

    for fname in file_list:
        fpath = os.path.join(DATA_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        chunks = split_text_into_chunks(text)
        print(f"🧩 {fname}: {len(chunks)} chunks")
        for c in chunks:
            source_map[len(all_chunks)] = {"chunk": c, "source": fname}
            all_chunks.append(c)

    print(f"🧠 Tổng số chunk: {len(all_chunks)}")
    print("🚀 Đang tạo embedding (GPU)...")
    import numpy as np
    embeds = embed_texts(all_chunks)

    print("📦 Đang lưu FAISS index...")
    index = faiss.IndexFlatL2(embeds.shape[1])
    index.add(embeds)

    faiss.write_index(index, os.path.join(PROCESSED_DIR, "faiss.index"))

    with open(os.path.join(PROCESSED_DIR, "chunk_map.json"), "w", encoding="utf-8") as f:
        json.dump(source_map, f, ensure_ascii=False, indent=2)

    print("✅ Hoàn tất! Lưu tại:", PROCESSED_DIR)
import os
import json
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from tqdm import tqdm

# ============ CẤU HÌNH ============
DATA_DIR = "./data/RAG"
PROCESSED_DIR = "./data/processed"
EMBED_MODEL_DIR = "/home/tungn/AIagent/models/embed"

CHUNK_SIZE = 500  # số từ mỗi chunk
CHUNK_OVERLAP = 50  # số từ trùng giữa các chunk

# ============ LOAD MODEL ============
print("📦 Đang load model embedding từ GPU...")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_DIR)
model = AutoModel.from_pretrained(EMBED_MODEL_DIR).cuda()
print("✅ Đã load model.")


# ============ TEXT UTILS ============
def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks


# ============ EMBED FUNC ============
def embed_texts(texts):
    embeddings = []
    batch_size = 8
    for i in tqdm(range(0, len(texts), batch_size), desc="🔍 Embedding chunks"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state[:, 0]
        embeddings.append(outputs.cpu().numpy())
    return np.vstack(embeddings)


# ============ CHẠY TOÀN BỘ ============
if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_chunks = []
    source_map = {}

    print("📂 Đang quét thư mục:", DATA_DIR)
    file_list = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

    for fname in file_list:
        fpath = os.path.join(DATA_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        chunks = split_text_into_chunks(text)
        print(f"🧩 {fname}: {len(chunks)} chunks")
        for c in chunks:
            source_map[len(all_chunks)] = {"chunk": c, "source": fname}
            all_chunks.append(c)

    print(f"🧠 Tổng số chunk: {len(all_chunks)}")
    print("🚀 Đang tạo embedding (GPU)...")
    import numpy as np
    embeds = embed_texts(all_chunks)

    print("📦 Đang lưu FAISS index...")
    index = faiss.IndexFlatL2(embeds.shape[1])
    index.add(embeds)

    faiss.write_index(index, os.path.join(PROCESSED_DIR, "faiss.index"))

    with open(os.path.join(PROCESSED_DIR, "chunk_map.json"), "w", encoding="utf-8") as f:
        json.dump(source_map, f, ensure_ascii=False, indent=2)

    print("✅ Hoàn tất! Lưu tại:", PROCESSED_DIR)
