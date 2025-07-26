# FILE: app.py
# VERSION: 5.0 - MERGED - Best of Home (Advanced RAG) & A100 (Detailed Logging)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['TIKTOKEN_CACHE_DIR'] = os.path.join(BASE_DIR, 'data', 'tiktoken_cache')
import json
import hashlib
import re # <-- Cần cho tìm kiếm
from datetime import datetime, timedelta
import secrets
import logging
from logging.handlers import TimedRotatingFileHandler
import platform
import subprocess
import time
import pytz
import yaml

from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from waitress import serve
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ===================== LỰA CHỌN LLM CLIENT (FIX CỨNG) =====================
# Đặt thành True khi chạy trên A100 (dùng vLLM).
# Đặt thành False khi chạy ở máy nhà (dùng llama-cpp).
IS_PRODUCTION_ENVIRONMENT = False
# ---------------------------------------------------------------------

if IS_PRODUCTION_ENVIRONMENT:
    try:
        from llm_vllm_client import call_llm
        logging.info("🚀 [PRODUCTION MODE] Đã tải vLLM client.")
    except ImportError:
        logging.error("❌ Lỗi: Chế độ là 'production' nhưng không tìm thấy file llm_vllm_client.py.")
        exit()
else:
    try:
        from llm_client import call_llm
        logging.info("🚀 [LOCAL MODE] Đã tải llama-cpp client.")
    except ImportError:
        logging.error("❌ Lỗi: Không tìm thấy file llm_client.py.")
        exit()

# ===================== CẤU HÌNH LOGGING =====================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_app = os.path.join(LOG_DIR, "app.log")
handler = TimedRotatingFileHandler(log_file_app, when="midnight", interval=1, backupCount=14)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s [in %(pathname)s:%(lineno)d]')
handler.setFormatter(formatter)
logging.basicConfig(handlers=[handler, logging.StreamHandler()], level=logging.INFO, force=True)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('waitress').disabled = True

# ===================== KHỞI TẠO FLASK APP =====================
app = Flask(__name__, template_folder='webchat/templates', static_folder='webchat/static')
app.secret_key = secrets.token_hex(24)

# ===================== CONFIGURATION & GLOBAL STATE =====================
APP_PORT = 5005
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
USERS_FILE = os.path.join(DATA_DIR, "users", "users.json")
HISTORY_BASE_DIR = os.path.join(DATA_DIR, "chat_histories")
RAG_PENDING_DIR = os.path.join(DATA_DIR, "rag_pending")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
FAISS_VI_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_vi.index")
FAISS_EN_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_en.index")
CHUNK_MAP_PATH = os.path.join(VECTOR_STORE_DIR, "chunk_map.json")
EMBED_VI_MODEL_PATH = os.path.join(BASE_DIR, "models", "embed_vi")
EMBED_EN_MODEL_PATH = os.path.join(BASE_DIR, "models", "embed_en")

embed_models = {}
faiss_indexes = {}
chunk_map = {}

TOKEN_PER_CHAR = 4
CONTEXT_LIMIT = 7000
VIETNAM_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

# Custom YAML representer for multi-line strings
def str_presenter(dumper, data):
  if len(data.splitlines()) > 1:
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
  return dumper.represent_scalar('tag:yaml.org,2002:str', data)
yaml.add_representer(str, str_presenter)

# ===================== HÀM HỖ TRỢ HỆ THỐNG & BẢO MẬT =====================
def kill_process_on_port(port):
    logging.info(f"Checking and stopping old processes on port {port}...")
    try:
        system = platform.system()
        cmd = ""
        if system == "Windows":
            find_cmd = f"netstat -aon | findstr :{port}"
            output = subprocess.check_output(find_cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
            if "LISTENING" in output:
                pid = output.strip().split()[-1]
                cmd = f"taskkill /F /PID {pid}"
        elif system in ["Linux", "Darwin"]:
            find_cmd = f"lsof -t -i:{port}"
            pids_output = subprocess.check_output(find_cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
            if pids_output:
                cmd = f"kill -9 {pids_output}"
        if cmd:
            subprocess.run(cmd, shell=True, check=True)
            logging.info(f"✅ Successfully stopped old process on port {port}.")
    except Exception:
        pass

def hash_password(password, salt):
    return hashlib.sha256(salt.encode() + password.encode()).hexdigest()

def check_password(password, salt, stored_hash):
    return hash_password(password, salt) == stored_hash

def sanitize_filename(name):
    return "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).replace(':', '_').replace('/', '_').replace('\\', '_').strip()

# ===================== HÀM LOGIC CỐT LÕI & RAG (HYBRID SEARCH VERSION) =====================
def load_vector_engine():
    global embed_models, faiss_indexes, chunk_map
    logging.info("Loading Vector Engine (Bilingual Mode)...")
    try:
        if os.path.exists(EMBED_VI_MODEL_PATH):
            logging.info("Loading Vietnamese embedding model...")
            embed_models["vi"] = SentenceTransformer(EMBED_VI_MODEL_PATH)
        if os.path.exists(EMBED_EN_MODEL_PATH):
            logging.info("Loading English/Multilingual embedding model...")
            embed_models["en"] = SentenceTransformer(EMBED_EN_MODEL_PATH)
        logging.info("✅ Successfully loaded embedding models.")

        if os.path.exists(FAISS_VI_INDEX_PATH):
            try:
                faiss_indexes["vi"] = faiss.read_index(FAISS_VI_INDEX_PATH)
                logging.info(f"Loaded Vietnamese FAISS index ({faiss_indexes['vi'].ntotal} vectors).")
            except Exception as e:
                logging.warning(f"Could not load Vietnamese FAISS index: {e}. RAG for VI may be limited.")
                faiss_indexes["vi"] = None
        else:
            logging.warning("⚠️ Vietnamese FAISS index not found. RAG for VI will be limited.")
            faiss_indexes["vi"] = None

        if os.path.exists(FAISS_EN_INDEX_PATH):
            try:
                faiss_indexes["en"] = faiss.read_index(FAISS_EN_INDEX_PATH)
                logging.info(f"Loaded English FAISS index ({faiss_indexes['en'].ntotal} vectors).")
            except Exception as e:
                logging.warning(f"Could not load English FAISS index: {e}. RAG for EN may be limited.")
                faiss_indexes["en"] = None
        else:
            logging.warning("⚠️ English FAISS index not found. RAG for EN will be limited.")
            faiss_indexes["en"] = None

        if os.path.exists(CHUNK_MAP_PATH):
            with open(CHUNK_MAP_PATH, "r", encoding="utf-8") as f:
                chunk_map = json.load(f)
            logging.info(f"✅ Successfully loaded chunk map ({len(chunk_map)} entries).")
        else:
            logging.warning("⚠️ Chunk map not found. RAG functionality will be limited.")
            chunk_map = {}
    except Exception as e:
        logging.error(f"❌ Critical error loading bilingual vector engine: {e}", exc_info=True)
        embed_models.clear(); faiss_indexes.clear(); chunk_map.clear()

def retrieve_context(query, top_k=3):
    if not chunk_map: return []
    final_results = {}
    
    # --- Vector Search with RRF ---
    if embed_models and faiss_indexes:
        try:
            results = {}
            for lang_code, model in embed_models.items():
                index = faiss_indexes.get(lang_code)
                if not index or index.ntotal == 0: continue
                query_vector = model.encode([query])
                distances, indices = index.search(np.array(query_vector).astype('float32'), top_k)
                results[lang_code] = list(zip(indices[0], distances[0]))
            
            fused_scores = {}
            k_rrf = 60
            for lang_code, lang_results in results.items():
                for rank, (doc_id, score) in enumerate(lang_results):
                    doc_id_str = str(doc_id)
                    if doc_id_str not in fused_scores: fused_scores[doc_id_str] = 0
                    fused_scores[doc_id_str] += 1 / (k_rrf + rank + 1)
            
            reranked_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

            for doc_id_str, score in reranked_results[:5]:
                chunk_package = chunk_map.get(doc_id_str)
                if not chunk_package: continue
                source_file = chunk_package.get("metadata", {}).get("source_file", doc_id_str)
                if source_file not in final_results:
                    final_results[source_file] = chunk_package
        except Exception as e:
            logging.error(f"Lỗi khi truy xuất vector search: {e}")

    # --- Fallback Keyword/IP Search ---
    try:
        search_terms = {query.lower()}
        found_ips = re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', query)
        if found_ips:
            search_terms.update(found_ips)
        
        for doc_id_str, chunk_package in chunk_map.items():
            source_file = chunk_package.get("metadata", {}).get("source_file", doc_id_str)
            if source_file in final_results: continue # Avoid re-adding

            full_text_search = json.dumps(chunk_package, ensure_ascii=False).lower()
            
            for term in search_terms:
                if term in full_text_search:
                    final_results[source_file] = chunk_package
                    break 
    except Exception as e:
        logging.error(f"Lỗi khi truy xuất fallback search: {e}")

    logging.info(f"Retrieved {len(final_results)} RAG documents using Hybrid Search.")
    return list(final_results.values())

def get_ai_response(user_input, chat_history):
    retrieved_data = retrieve_context(user_input)
    is_vietnamese = any(char in user_input for char in "àáãạảăằắẵặẳâầấẫậẩđèéẹẻẽêềếễệểìíịỉĩòóọỏõôồốỗộổơờớỡợởùúụủũưừứữựửỳýỵỷỹ")

    rag_context_str = ""
    if retrieved_data:
        context_parts = []
        # Format for Vietnamese: More readable, human-friendly
        if is_vietnamese:
            for item in retrieved_data:
                meta = item.get("metadata", {})
                content = item.get("content", "N/A")
                part = f"""--- Tài liệu: {meta.get("title", "Không rõ")} ---
- Mã hệ thống: {meta.get("system_code", "Không rõ")}
- Tags: {', '.join(meta.get("tags", []))}"""
                if 'hosts' in meta and meta['hosts']:
                    part += "\n- Các máy chủ liên quan:\n"
                    for host in meta['hosts']:
                        part += f"  - Hostname: {host.get('hostname', 'N/A')}, IP: {host.get('ip', 'N/A')}\n"
                part += f"\n- Nội dung: {content}"
                context_parts.append(part)
        # Format for English: YAML-like, structured for technical models
        else:
            for item in retrieved_data:
                metadata_str = yaml.dump(item.get("metadata", {}), allow_unicode=True, indent=2, sort_keys=False)
                content = item.get("content", "N/A")
                part = f"---\n{metadata_str}content: |\n  {content}\n---"
                context_parts.append(part)
        rag_context_str = "\n\n".join(context_parts)
    else:
        rag_context_str = "Không tìm thấy tài liệu nào liên quan."

    # Create dynamic prompt based on language
    if is_vietnamese:
        system_prompt = f"""Mày là một trợ lý AI kỹ thuật chuyên nghiệp. Nhiệm vụ của mày là đọc kỹ <tài_liệu> dưới đây và trả lời câu hỏi của người dùng một cách chính xác.
Câu trả lời có thể nằm trong phần 'metadata' (như mã hệ thống, hosts, ip) hoặc trong phần 'nội dung'. Hãy suy luận để trả lời.
TRẢ LỜI BẰNG TIẾNG VIỆT. Nếu không tìm thấy, hãy nói là không tìm thấy.

<tài_liệu>
{rag_context_str}
</tài_liệu>
"""
    else:
        system_prompt = f"""You are a professional technical AI assistant. Your task is to carefully read the <document> below and accurately answer the user's question.
The answer might be in the 'metadata' (like system_code, hosts, ip) or in the 'content'. Use logical reasoning to answer.
ANSWER IN ENGLISH. If you cannot find the answer, say so.

<document>
{rag_context_str}
</document>
"""

    # Context window management
    temp_history = list(chat_history)
    messages = [{"role": "system", "content": system_prompt}] + temp_history + [{"role": "user", "content": user_input}]
    
    current_tokens = sum(len(m.get("content", "")) // TOKEN_PER_CHAR for m in messages)
    while current_tokens > CONTEXT_LIMIT and len(temp_history) > 1:
        temp_history.pop(0); temp_history.pop(0)
        messages = [{"role": "system", "content": system_prompt}] + temp_history + [{"role": "user", "content": user_input}]
        current_tokens = sum(len(m.get("content", "")) // TOKEN_PER_CHAR for m in messages)

    if 'username' in session:
        session['chat_history'] = temp_history
        
    reply = call_llm(messages)
    return reply, sum(len(m.get("content", "")) // TOKEN_PER_CHAR for m in messages + [{"role": "assistant", "content": reply}])

# ===================== FLASK ROUTES =====================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session: return redirect(url_for('chat_ui'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f: users = json.load(f)
            user_data = next((u for u in users if u['username'] == username), None)
            if user_data and check_password(password, user_data.get('salt', ''), user_data.get('password_hash', '')):
                session['username'] = username
                session.permanent = True
                app.permanent_session_lifetime = timedelta(days=7)
                logging.info(f"User '{username}' logged in successfully.")
                return redirect(url_for('chat_ui'))
            else:
                return render_template('login.html', error="Invalid username or password.")
        except Exception as e:
            logging.error(f"Login error: {e}", exc_info=True)
            return render_template('login.html', error="System error during login.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    username = session.get('username', 'unknown')
    session.clear()
    logging.info(f"User '{username}' logged out.")
    return redirect(url_for('login'))

@app.route('/')
def chat_ui():
    if 'username' not in session: return redirect(url_for('login'))
    return render_template('chat.html', username=session['username'], session_id=session.get('active_session_id', 'New Session'))

@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.json['message']
    chat_history = session.get('chat_history', []) 
    reply_content, token_count = get_ai_response(user_message, chat_history)
    
    if 'username' in session:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": reply_content})
        session['chat_history'] = chat_history
        
    return jsonify({"reply": reply_content, "token_count": token_count})

@app.route('/new_session', methods=['POST'])
def new_session_api():
    if 'username' not in session: return jsonify({"error": "Not logged in"}), 401
    session['chat_history'] = []
    session['active_session_id'] = None
    return jsonify({"status": "success"})

@app.route('/save_session', methods=['POST'])
def save_session_api():
    if 'username' not in session: return jsonify({"error": "Not logged in"}), 401
    username, chat_history = session['username'], session.get('chat_history', [])
    if not chat_history: return jsonify({"error": "No content to save."}), 400
    active_id = session.get('active_session_id')
    input_name = request.json.get('session_name', '').strip()
    filename_base = sanitize_filename(input_name) if input_name else (active_id if active_id and active_id != 'New Session' else datetime.now(VIETNAM_TZ).strftime('%Y%m%d_%H%M%S'))
    session['active_session_id'] = filename_base
    user_path = os.path.join(HISTORY_BASE_DIR, username); os.makedirs(user_path, exist_ok=True)
    filepath = os.path.join(user_path, f"{filename_base}.json")
    current_time = datetime.now(VIETNAM_TZ).isoformat()
    data = {'updated_at': current_time, 'messages': chat_history}
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            try: data['created_at'] = json.load(f).get('created_at', current_time)
            except json.JSONDecodeError: data['created_at'] = current_time
    else: data['created_at'] = current_time
    with open(filepath, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "success", "new_session_id": filename_base})

@app.route('/history/list', methods=['GET'])
def list_history_api():
    if 'username' not in session: return jsonify({"error": "Not logged in"}), 401
    user_path = os.path.join(HISTORY_BASE_DIR, session['username'])
    sessions = []
    if os.path.exists(user_path):
        for f_name in os.listdir(user_path):
            if f_name.endswith(".json"):
                try:
                    with open(os.path.join(user_path, f_name), 'r', encoding='utf-8') as f: data = json.load(f)
                    sessions.append({"session_id": os.path.splitext(f_name)[0], "created_at": data.get("created_at"), "updated_at": data.get("updated_at")})
                except Exception: continue
    sessions.sort(key=lambda x: x.get('updated_at') or x.get('created_at') or '', reverse=True)
    return jsonify(sessions)

@app.route('/history/load', methods=['POST'])
def load_history_api():
    if 'username' not in session: return jsonify({"error": "Not logged in"}), 401
    session_id = request.json['session_id']
    filepath = os.path.join(HISTORY_BASE_DIR, session['username'], f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f: data = json.load(f)
        session['chat_history'] = data.get('messages', [])
        session['active_session_id'] = session_id
        return jsonify({"status": "success", "history": session['chat_history'], "session_id": session_id})
    return jsonify({"error": "Not found"}), 404

@app.route('/history/delete', methods=['POST'])
def delete_history_api():
    if 'username' not in session: return jsonify({"error": "Not logged in"}), 401
    filepath = os.path.join(HISTORY_BASE_DIR, session['username'], f"{request.json['session_id']}.json")
    if os.path.exists(filepath): os.remove(filepath)
    return jsonify({"status": "success"})

@app.route('/history/rename', methods=['POST'])
def rename_history_api():
    if 'username' not in session: return jsonify({"error": "Not logged in"}), 401
    old_id, new_name = request.json['old_id'], request.json['new_name']
    user_path = os.path.join(HISTORY_BASE_DIR, session['username'])
    sanitized_name = sanitize_filename(new_name)
    if not sanitized_name: return jsonify({"error": "Invalid name"}), 400
    old_fp, new_fp = os.path.join(user_path, f"{old_id}.json"), os.path.join(user_path, f"{sanitized_name}.json")
    if os.path.exists(new_fp): return jsonify({"error": "Name exists"}), 409
    if os.path.exists(old_fp):
        os.rename(old_fp, new_fp)
        if session.get('active_session_id') == old_id: session['active_session_id'] = sanitized_name
        return jsonify({"status": "success"})
    return jsonify({"error": "Not found"}), 404

@app.route('/prepare_rag_submission', methods=['POST'])
def prepare_rag_submission_api():
    if 'username' not in session: return jsonify({"error": "Not logged in"}), 401
    chat_history = session.get('chat_history', [])
    if len(chat_history) < 2: return jsonify({"error": "Insufficient content"}), 400
    full_conversation = "\n".join([f"{'User' if msg['role'] == 'user' else 'AI'}: {msg['content']}" for msg in chat_history])
    
    # === STEP 1: Extract structured data (Metadata & Action Plan) ===
    structured_data_prompt = f"""
TASK: Extract structured metadata and an actionable plan from a technical conversation.
INPUT: A conversation.
OUTPUT: A single, raw JSON object. Do NOT provide any explanation or surrounding text.

CONVERSATION:
---
{full_conversation}
---

REQUIRED JSON STRUCTURE:
{{
  "metadata": {{
    "title": "A short, descriptive title for the knowledge document.",
    "system_code": "The relevant system code (e.g., WEBSV_HA).",
    "tags": ["tag1", "tag2"],
    "version": "1.0",
    "related_docs": ["other_doc.yml"],
    "hosts": [{{ "hostname": "host1", "ip": "ip1" }}]
  }},
  "actionable_plan": {{
    "alert_summary": "A one-sentence summary for alert notifications.",
    "investigation_steps": [
      {{ "name": "check_connections", "description": "Check active connections.", "command": "ss -tan" }}
    ],
    "remediation_playbooks": [
      {{ "name": "safe_restart", "description": "Safe restart of the service.", "type": "ssh_command", "target": "sudo systemctl restart service", "allow_automation": true }}
    ]
  }}
}}

JSON_OUTPUT:
"""
    messages_step1 = [{"role": "user", "content": structured_data_prompt}]
    response_step1 = call_llm(messages_step1)
    
    structured_data = {}
    try:
        json_start = response_step1.find('{')
        json_end = response_step1.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            structured_data = json.loads(response_step1[json_start:json_end])
        else:
            raise json.JSONDecodeError("No valid JSON found", response_step1, 0)
    except json.JSONDecodeError:
        logging.error("JSON parsing error in RAG prep step 1. Using fallback structure.")
        structured_data = {"metadata": {}, "actionable_plan": {}}

    context_title = structured_data.get("metadata", {}).get("title", "N/A")
    context_tags = ", ".join(structured_data.get("metadata", {}).get("tags", []))

    # === STEP 2: Generate content based on available data ===
    content_generation_prompt = f"""
TASK: You are a senior engineer writing a technical knowledge base article.
CONTEXT: The article is titled '{context_title}' and is about '{context_tags}'.
INPUT: A raw conversation.
OUTPUT: A clear, well-structured technical summary in Vietnamese. Explain the 'why' (root cause) and the 'how' (long-term solution). Do NOT include greetings, titles, or any other text, just the summary content itself.

CONVERSATION:
---
{full_conversation}
---

SUMMARY_CONTENT:
"""
    messages_step2 = [{"role": "user", "content": content_generation_prompt}]
    generated_content = call_llm(messages_step2).strip()

    final_data = {
        "metadata": structured_data.get("metadata", {}),
        "actionable_plan": structured_data.get("actionable_plan", {}),
        "content": generated_content
    }
    logging.info("Prepared RAG draft using 2-step method.")
    return jsonify(final_data)

@app.route('/submit_rag_knowledge', methods=['POST'])
def submit_rag_knowledge_api():
    if 'username' not in session: return jsonify({"error": "Not logged in"}), 401
    knowledge_unit = request.json
    metadata = knowledge_unit.get('metadata', {})
    content = knowledge_unit.get('content', '').strip()
    if not content or not metadata.get('system_code') or not metadata.get('title'):
        return jsonify({"error": "Missing required fields."}), 400

    metadata['curator'] = session['username']
    metadata['source_session_id'] = session.get('active_session_id', 'unknown')
    metadata['curated_at'] = datetime.now(VIETNAM_TZ).isoformat()
    knowledge_unit['metadata'] = metadata

    rag_pending_user_dir = os.path.join(RAG_PENDING_DIR, session['username'])
    os.makedirs(rag_pending_user_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    sanitized_title = sanitize_filename(metadata.get('title', 'untitled'))
    filename = f"{sanitized_title[:50]}_{timestamp}.yml"
    filepath = os.path.join(rag_pending_user_dir, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(knowledge_unit, f, allow_unicode=True, sort_keys=False, indent=2)
        logging.info(f"User '{session['username']}' contributed RAG knowledge: {filename}")
        return jsonify({"status": "success"})
    except Exception as e:
        logging.error(f"Error saving RAG knowledge: {e}", exc_info=True)
        return jsonify({"error": "System error saving file."}), 500

# ===================== HÀM KHỞI TẠO ỨNG DỤNG =====================
def initialize_app():
    logging.info("--- Starting application initialization process ---")
    for path in [os.path.dirname(USERS_FILE), HISTORY_BASE_DIR, RAG_PENDING_DIR, VECTOR_STORE_DIR]:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Checked/created directory: {path}")

    if not os.path.exists(USERS_FILE):
        logging.warning(f"⚠️ User file at '{USERS_FILE}' does not exist. Creating default file...")
        default_username, default_password = "admin", secrets.token_urlsafe(12)
        salt = secrets.token_hex(16)
        password_hash = hash_password(default_password, salt)
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump([{"username": default_username, "password_hash": password_hash, "salt": salt}], f, indent=2)
        logging.info("✅ Successfully created users.json file.")
        print("="*60, "\n!!! FIRST-TIME LOGIN INFORMATION !!!", f"\n    Username: {default_username}", f"\n    Password: {default_password}", "\n" + "="*60)
        
    load_vector_engine()
    logging.info("--- Initialization process complete ---")

# ===================== KHỞI ĐỘNG ỨNG DỤNG =====================
if __name__ == '__main__':
    kill_process_on_port(APP_PORT)
    time.sleep(1)
    initialize_app()
    print(f"\n🚀 AI Agent is ready! Access at: http://127.0.0.1:{APP_PORT}")
    serve(app, host='0.0.0.0', port=APP_PORT, threads=16)
