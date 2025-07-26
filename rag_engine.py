# FILE: rag_engine.py
# VERSION: Hoàn chỉnh - Thêm tính năng tự động thông báo cho Alert-API

import os
import json
import hashlib
import shutil
import logging
from datetime import datetime
import argparse
import sys
import requests # <-- THÊM MỚI: Import thư viện requests

import numpy as np
import faiss
import torch
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain_text_splitters import TokenTextSplitter

# --- Cấu hình Logging và Thư mục ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_rag = os.path.join(LOG_DIR, f"rag_engine_{datetime.now().strftime('%Y-%m-%d')}.log")
handler = logging.FileHandler(log_file_rag, mode='a', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logging.basicConfig(handlers=[handler], level=logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAG_PENDING_DIR = os.path.join(DATA_DIR, "rag_pending")
RAG_SOURCE_DIR = os.path.join(DATA_DIR, "rag_source")
RAG_PROCESSED_DIR = os.path.join(DATA_DIR, "rag_processed")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")

FAISS_VI_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_vi.index")
FAISS_EN_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_en.index")
CHUNK_MAP_PATH = os.path.join(VECTOR_STORE_DIR, "chunk_map.json")
PROCESSED_CHUNK_HASHES_PATH = os.path.join(VECTOR_STORE_DIR, "processed_chunk_hashes.json")

EMBED_VI_MODEL_PATH = os.path.join(BASE_DIR, "models", "embed_vi")
EMBED_EN_MODEL_PATH = os.path.join(BASE_DIR, "models", "embed_en")

tokenizer = None
text_splitter = None

try:
    import pypdf
    import docx
    from bs4 import BeautifulSoup
    import markdown_it
except ImportError as e:
    logging.error(f"❌ Missing file processing library: {e}. Please run 'pip install pypdf python-docx beautifulsoup4 markdown-it-py'")
    sys.exit(1)

def load_processed_chunk_hashes():
    if os.path.exists(PROCESSED_CHUNK_HASHES_PATH):
        with open(PROCESSED_CHUNK_HASHES_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def extract_knowledge_from_file(filepath):
    _, extension = os.path.splitext(filepath.lower())
    knowledge_unit = {"metadata": {}, "content": "", "actionable_plan": {}}
    try:
        if extension in ['.yml', '.yaml']:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            knowledge_unit["metadata"] = data.get('metadata', {})
            knowledge_unit["content"] = data.get('content', '')
            knowledge_unit["actionable_plan"] = data.get('actionable_plan', {})
            if not knowledge_unit["actionable_plan"] and 'suggestions' in data:
                knowledge_unit["actionable_plan"] = data.get('suggestions', {})
        else:
            knowledge_unit["metadata"] = { "system_code": "GENERAL", "tags": [extension.replace('.', '')], "overwrite_existing": False, "title": os.path.basename(filepath), "version": "1.0" }
            if extension == '.json':
                with open(filepath, 'r', encoding='utf-8') as f: j_data = json.load(f)
                knowledge_unit["content"] = "\n".join([f"{msg['role']}: {msg['content']}" for msg in j_data.get('messages', [])]) if 'messages' in j_data else json.dumps(j_data, ensure_ascii=False, indent=2)
            elif extension == '.pdf':
                reader = pypdf.PdfReader(filepath)
                knowledge_unit["content"] = "".join([page.extract_text() or "" for page in reader.pages])
            elif extension in ['.txt', '.md']:
                with open(filepath, 'r', encoding="utf-8") as f: text = f.read()
                knowledge_unit["content"] = BeautifulSoup(markdown_it.MarkdownIt().render(text), 'html.parser').get_text() if extension == '.md' else text
            elif extension == '.docx':
                doc = docx.Document(filepath)
                knowledge_unit["content"] = "\n".join([para.text for para in doc.paragraphs])
            else:
                logging.warning(f"Unsupported file format: {filepath}. Skipping.")
                return None
        knowledge_unit["metadata"]['source_file'] = os.path.basename(filepath)
    except Exception as e:
        logging.error(f"Error processing file {filepath}: {e}", exc_info=True)
        return None
    return knowledge_unit

def handle_overwrite_bilingual(faiss_indexes, chunk_map, content_to_overwrite, embed_models, distance_threshold=1.0):
    if faiss_indexes["en"].ntotal == 0: return faiss_indexes, chunk_map, set()
    logging.info("Starting Overwrite process (based on English model for similarity detection)...")
    target_vector = embed_models["en"].encode([content_to_overwrite], convert_to_tensor=True).cpu().numpy()
    k = min(100, faiss_indexes["en"].ntotal)
    distances, indices = faiss_indexes["en"].search(target_vector, k)
    ids_to_remove = set()
    for i, idx in enumerate(indices[0]):
        if distances[0][i] < distance_threshold: ids_to_remove.add(idx)
    if not ids_to_remove:
        logging.info("No sufficiently similar old chunks found for overwriting.")
        return faiss_indexes, chunk_map, set()
    logging.warning(f"Detected {len(ids_to_remove)} old chunks to be OVERWRITTEN across both indexes.")
    id_array = np.array(list(ids_to_remove))
    faiss_indexes["vi"].remove_ids(id_array)
    faiss_indexes["en"].remove_ids(id_array)
    hashes_to_remove = set()
    new_chunk_map = {}
    for idx_str, chunk_data in chunk_map.items():
        if int(idx_str) not in ids_to_remove:
            new_chunk_map[idx_str] = chunk_data
        else:
            text_hash = hashlib.sha256(chunk_data['content'].encode('utf-8')).hexdigest()
            hashes_to_remove.add(text_hash)
    logging.info(f"Removed {len(ids_to_remove)} chunks from FAISS indexes and chunk map.")
    return faiss_indexes, new_chunk_map, hashes_to_remove

def run_rag_builder():
    logging.info("================ STARTING RAG BUILDER (BILINGUAL MODE) ===============")
    for path in [RAG_PENDING_DIR, RAG_SOURCE_DIR, RAG_PROCESSED_DIR, VECTOR_STORE_DIR]: os.makedirs(path, exist_ok=True)
    global tokenizer, text_splitter
    embed_models = {}
    try:
        logging.info("Loading Vietnamese embedding model..."); embed_models["vi"] = SentenceTransformer(EMBED_VI_MODEL_PATH)
        logging.info("Loading English/Multilingual embedding model..."); embed_models["en"] = SentenceTransformer(EMBED_EN_MODEL_PATH)
    except Exception as e: logging.error(f"❌ Error loading embedding models: {e}. Please check model paths.", exc_info=True); return
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMBED_EN_MODEL_PATH)
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50, length_function=lambda text: len(tokenizer.encode(text, add_special_tokens=False)))
    except Exception as e: logging.error(f"❌ Error initializing tokenizer or text splitter: {e}", exc_info=True); return
    processed_chunk_hashes = load_processed_chunk_hashes()
    faiss_indexes = {}
    for lang in ["vi", "en"]:
        path = FAISS_VI_INDEX_PATH if lang == "vi" else FAISS_EN_INDEX_PATH
        if os.path.exists(path):
            try: faiss_indexes[lang] = faiss.read_index(path); logging.info(f"Loaded existing FAISS index for {lang.upper()} from {path}")
            except Exception as e: logging.warning(f"Could not load FAISS index for {lang.upper()} from {path}. Initializing new index. Error: {e}"); dim = embed_models[lang].get_sentence_embedding_dimension(); faiss_indexes[lang] = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        else: dim = embed_models[lang].get_sentence_embedding_dimension(); faiss_indexes[lang] = faiss.IndexIDMap(faiss.IndexFlatL2(dim)); logging.info(f"Initialized new FAISS index for {lang.upper()} at {path}")
    chunk_map = {}
    if os.path.exists(CHUNK_MAP_PATH):
        try:
            with open(CHUNK_MAP_PATH, "r", encoding="utf-8") as f: chunk_map = json.load(f)
            logging.info(f"Loaded existing chunk map from {CHUNK_MAP_PATH}")
        except Exception as e: logging.warning(f"Could not load chunk map from {CHUNK_MAP_PATH}. Initializing new chunk map. Error: {e}"); chunk_map = {}
    source_files = [os.path.join(root, file) for dir_path in [RAG_PENDING_DIR, RAG_SOURCE_DIR] for root, _, files in os.walk(dir_path) for file in files if file.lower().endswith(('.yml', '.yaml', '.txt', '.pdf', '.docx', '.json', '.md'))]
    if not source_files: logging.info("✅ No new knowledge files to process."); return
    all_new_chunks_data = []; processed_files_paths = []
    for filepath in tqdm(source_files, desc="🔎 Extracting and processing knowledge files"):
        knowledge_unit = extract_knowledge_from_file(filepath)
        if not knowledge_unit or not knowledge_unit.get("content"): continue
        metadata = knowledge_unit["metadata"]; content = knowledge_unit["content"]; actionable_plan = knowledge_unit.get("actionable_plan", {})
        if metadata.get('overwrite_existing', False):
            logging.info(f"Processing overwrite for file: {os.path.basename(filepath)}")
            faiss_indexes, chunk_map, hashes_to_remove = handle_overwrite_bilingual(faiss_indexes, chunk_map, content, embed_models)
            processed_chunk_hashes -= hashes_to_remove
        chunks = text_splitter.split_text(content)
        for chunk_text in chunks:
            chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
            if chunk_hash not in processed_chunk_hashes:
                processed_chunk_hashes.add(chunk_hash)
                all_new_chunks_data.append({"metadata": metadata, "content": chunk_text, "actionable_plan": actionable_plan})
        processed_files_paths.append(filepath)
    if all_new_chunks_data:
        logging.info(f"🧠 Found {len(all_new_chunks_data)} new chunks. Starting bilingual embedding...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        chunk_contents = [c['content'] for c in all_new_chunks_data]
        embeddings_vi = embed_models["vi"].encode(chunk_contents, device=device, convert_to_numpy=True, show_progress_bar=True)
        embeddings_en = embed_models["en"].encode(chunk_contents, device=device, convert_to_numpy=True, show_progress_bar=True)
        start_id = max(map(int, chunk_map.keys())) + 1 if chunk_map else 0
        new_ids = np.arange(start_id, start_id + len(all_new_chunks_data))
        faiss_indexes["vi"].add_with_ids(embeddings_vi.astype('float32'), new_ids)
        faiss_indexes["en"].add_with_ids(embeddings_en.astype('float32'), new_ids)
        for i, chunk_package in enumerate(all_new_chunks_data): chunk_map[str(new_ids[i])] = chunk_package
        logging.info(f"✅ Added {len(all_new_chunks_data)} new chunks to both VectorDBs.")
    else: logging.info("✅ No new chunks were added.")
    try:
        faiss.write_index(faiss_indexes["vi"], FAISS_VI_INDEX_PATH); faiss.write_index(faiss_indexes["en"], FAISS_EN_INDEX_PATH)
        with open(CHUNK_MAP_PATH, "w", encoding="utf-8") as f: json.dump(chunk_map, f, ensure_ascii=False, indent=2)
        with open(PROCESSED_CHUNK_HASHES_PATH, "w", encoding="utf-8") as f: json.dump(list(processed_chunk_hashes), f)
        logging.info("💾 Successfully saved both VectorDBs, Chunk Map, and Hashes.")
    except Exception as e: logging.error(f"❌ Error saving VectorDBs or related files: {e}", exc_info=True)
    for path in processed_files_paths:
        try:
            relative_path = os.path.relpath(os.path.dirname(path), DATA_DIR); dest_dir = os.path.join(RAG_PROCESSED_DIR, relative_path)
            os.makedirs(dest_dir, exist_ok=True); shutil.move(path, os.path.join(dest_dir, os.path.basename(path)))
        except Exception as e: logging.error(f"Error moving file {path}: {e}", exc_info=True)
    
    logging.info(f"🎉 Completed! Processed {len(processed_files_paths)} files.")

    # =================================================================================
    # === BƯỚC CẢI TIẾN: TỰ ĐỘNG THÔNG BÁO CHO ALERT-API ĐỂ LÀM MỚI CACHE TRI THỨC ===
    # =================================================================================
    # Sau khi cập nhật xong VectorDB, ta cần báo cho alert-api xóa cache cũ đi
    # để nó đọc lại các file .yml mới nhất ở lần xử lý sự cố tiếp theo.
    try:
        # URL này trỏ đến service alert-api trong cùng network Docker
        alert_api_reload_url = "http://alert-api:5001/reload" 
        logging.info(f"Notifying Alert-API to reload its knowledge cache at {alert_api_reload_url}...")
        response = requests.post(alert_api_reload_url, timeout=10)
        if response.status_code == 200:
            logging.info("✅ Successfully triggered Alert-API cache reload.")
        else:
            logging.warning(f"⚠️ Failed to trigger Alert-API cache reload. Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        # Lỗi ở bước này không làm dừng chương trình, chỉ ghi lại cảnh báo
        logging.error(f"❌ Could not notify Alert-API to reload cache. This is not a critical error, but you may need to restart the alert-api service manually to load new knowledge. Error: {e}")
    # =================================================================================

def search_knowledge_base(query_text, index, model, chunk_map, k=5, metadata_filter=None):
    if index.ntotal == 0:
        logging.info("VectorDB is empty. Cannot perform search.")
        return np.array([]), np.array([])

    query_embedding = model.encode([query_text], convert_to_numpy=True).astype('float32')

    if metadata_filter and isinstance(metadata_filter, dict):
        key_path, filter_value = next(iter(metadata_filter.items()))

        if key_path == "hosts.hostname":
            valid_ids = []
            for idx_str, chunk_data in chunk_map.items():
                hosts = chunk_data.get("metadata", {}).get("hosts", [])
                if isinstance(hosts, list):
                    for host in hosts:
                        if isinstance(host, dict) and host.get("hostname") == filter_value:
                            valid_ids.append(int(idx_str))
                            break
            
            if valid_ids:
                logging.info(f"Applying filter, searching within {len(valid_ids)} chunks for hostname '{filter_value}'.")
                id_selector = faiss.IDSelectorArray(np.array(valid_ids, dtype='int64'))
                params = faiss.SearchParameters()
                params.sel = id_selector
                return index.search(query_embedding, k, params=params)
            else:
                logging.warning(f"Filter provided for hostname '{filter_value}', but no matching documents found. Returning empty result.")
                return np.array([[]]), np.array([[]])

    logging.info("No valid filter applied. Performing a general search.")
    return index.search(query_embedding, k)

def run_rag_debugger():
    logging.info("================ STARTING RAG DEBUGGER (BILINGUAL MODE) ===============")
    if not os.path.exists(FAISS_VI_INDEX_PATH) or not os.path.exists(FAISS_EN_INDEX_PATH) or not os.path.exists(CHUNK_MAP_PATH):
        logging.warning("⚠️ One or more VectorDBs or Chunk Map do not exist. Please run BUILD mode first.")
        return

    embed_models = {}; faiss_indexes = {}; chunk_map = {}
    try:
        logging.info("Loading Vietnamese embedding model for debugger..."); embed_models["vi"] = SentenceTransformer(EMBED_VI_MODEL_PATH)
        logging.info("Loading English/Multilingual embedding model for debugger..."); embed_models["en"] = SentenceTransformer(EMBED_EN_MODEL_PATH)
        faiss_indexes["vi"] = faiss.read_index(FAISS_VI_INDEX_PATH)
        faiss_indexes["en"] = faiss.read_index(FAISS_EN_INDEX_PATH)
        with open(CHUNK_MAP_PATH, "r", encoding="utf-8") as f: chunk_map = json.load(f)
    except Exception as e:
        logging.error(f"❌ Error loading models or VectorDBs for debugger: {e}", exc_info=True)
        return

    logging.info(f"Total chunks in Vietnamese VectorDB: {faiss_indexes['vi'].ntotal}")
    logging.info(f"Total chunks in English VectorDB: {faiss_indexes['en'].ntotal}")

    while True:
        query = input("\nEnter your query (or 'quit' to exit): ").strip()
        if query.lower() == 'quit': break
        if not query: continue

        lang_choice = input("Which language model to use for query? (vi/en, default: en): ").strip().lower()
        selected_lang = "en" if lang_choice not in ["vi", "en"] else lang_choice

        filter_hostname = input("Enter hostname to filter by (or press Enter for none): ").strip()
        test_filter = {"hosts.hostname": filter_hostname} if filter_hostname else None

        try:
            selected_model = embed_models[selected_lang]
            selected_index = faiss_indexes[selected_lang]

            distances, indices = search_knowledge_base(
                query, selected_index, selected_model, chunk_map, k=5, metadata_filter=test_filter
            )

            logging.info(f"\n⚡ Search results for query: '{query}' (using {selected_lang.upper()} model)")
            if indices.size == 0 or len(indices[0]) == 0:
                logging.info("No results found.")
                continue

            for i, idx in enumerate(indices[0]):
                if str(idx) in chunk_map:
                    chunk_package = chunk_map[str(idx)]
                    metadata = chunk_package.get("metadata", {})
                    actionable_plan = chunk_package.get("actionable_plan", {})
                    logging.info(f"\n--- Chunk {i+1} (ID: {idx}, L2 Distance: {distances[0][i]:.4f}) ---")
                    logging.info(f"  Source File: {metadata.get('source_file', 'N/A')}")
                    logging.info(f"  Title: {metadata.get('title', 'N/A')}")
                    logging.info(f"  Content (Preview): {chunk_package.get('content', '')[:200]}...")
                    logging.info(f"  Actionable Plan: {actionable_plan.get('alert_summary', 'None')}")
                else:
                    logging.warning(f"Chunk ID {idx} exists in {selected_lang.upper()} FAISS index but not in chunk_map. Data might be out of sync.")
        except Exception as e:
            logging.error(f"Error during debugging: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Engine for building and debugging the knowledge base.")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode to search and inspect VectorDB.")
    args = parser.parse_args()
    if args.debug:
        run_rag_debugger()
    else:
        run_rag_builder()
