# FILE: app.py
import os
import json
import hashlib
import shutil
from datetime import datetime
import secrets
import logging
from logging.handlers import TimedRotatingFileHandler
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from waitress import serve
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llm_client import call_llm
import subprocess # Thêm import này
import platform   # Thêm import này
import time       # Thêm import này

# ===================== Cấu hình Logging =====================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_app = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y-%m-%d')}.log")
handler = TimedRotatingFileHandler(log_file_app, when="midnight", interval=1, backupCount=7)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s [in %(pathname)s:%(lineno)d]')
handler.setFormatter(formatter)
logging.basicConfig(handlers=[handler], level=logging.INFO)

# ===================== KHỞI TẠO FLASK APP =====================
app = Flask(__name__, template_folder='webchat/templates', static_folder='webchat/static')
app.secret_key = secrets.token_hex(16)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('waitress').disabled = True

# ===================== CONFIG & GLOBAL STATE =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
USERS_FILE = os.path.join(DATA_DIR, "users", "users.json")
HISTORY_BASE_DIR = os.path.join(DATA_DIR, "chat_histories")
RAG_PENDING_DIR = os.path.join(DATA_DIR, "rag_pending")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss.index")
CHUNK_MAP_PATH = os.path.join(VECTOR_STORE_DIR, "chunk_map.json")

embed_model, faiss_index, chunk_map = None, None, {}
TOKEN_PER_CHAR = 4
CONTEXT_LIMIT = 7000 # Ngưỡng an toàn cho 8k context

# ===================== CÁC HÀM HỖ TRỢ HỆ THỐNG =====================
def kill_process_on_port(port):
    """
    Kiểm tra và dừng tiến trình đang sử dụng một cổng cụ thể.
    Hỗ trợ cả Windows và Linux/macOS.
    """
    logging.info(f"Kiểm tra và dừng tiến trình cũ trên port {port}...")
    try:
        if platform.system() == "Windows":
            command = f"netstat -aon | findstr :{port}"
            output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.DEVNULL)
            if "LISTENING" in output:
                # Lấy PID từ dòng output
                pid = output.strip().split()[-1]
                subprocess.run(f"taskkill /F /PID {pid}", shell=True, check=True)
                logging.info(f"✅ Đã dừng tiến trình PID {pid} trên port {port}.")
                print(f"✅ Đã dừng tiến trình PID {pid} trên port {port}.")
        else: # Linux & macOS
            command = f"lsof -t -i:{port}"
            pids_output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
            pids = pids_output.split('\n') if pids_output else []
            for pid in filter(None, pids):
                subprocess.run(f"kill -9 {pid}", shell=True, check=True)
                logging.info(f"✅ Đã dừng tiến trình PID {pid} trên port {port}.")
                print(f"✅ Đã dừng tiến trình PID {pid} trên port {port}.")
        if not pids and platform.system() != "Windows" and "LISTENING" not in output: # Check for no active processes found
            logging.info(f"✅ Port {port} đang rảnh.")
            print(f"✅ Port {port} đang rảnh.")

    except subprocess.CalledProcessError as e:
        # Lệnh không thành công (có thể do không tìm thấy tiến trình)
        logging.info(f"✅ Port {port} đang rảnh hoặc không tìm thấy tiến trình cần dừng.")
        print(f"✅ Port {port} đang rảnh hoặc không tìm thấy tiến trình cần dừng.")
    except FileNotFoundError:
        logging.warning(f"Lệnh 'netstat' (Windows) hoặc 'lsof' (Linux/macOS) không tìm thấy. Không thể kiểm tra port {port}.")
        print(f"🟡 Lệnh 'netstat' hoặc 'lsof' không tìm thấy. Không thể kiểm tra port {port}.")
    except Exception as e:
        logging.error(f"❌ Lỗi khi dọn dẹp port {port}: {e}")
        print(f"❌ Lỗi khi dọn dẹp port {port}: {e}")

# ===================== CÁC HÀM LOGIC CỐT LÕI =====================
def sanitize_string(text: str) -> str: return text.encode('utf-8', 'replace').decode('utf-8')
def hash_password(password: str) -> str: return hashlib.sha256(password.encode()).hexdigest()
def count_tokens_simple(messages: list) -> int: return sum(len(m.get("content", "")) // TOKEN_PER_CHAR for m in messages)

def load_vector_engine():
    global embed_model, faiss_index, chunk_map
    logging.info("Đang tải SentenceTransformer model...")
    try:
        # Khởi tạo mô hình chỉ một lần
        embed_model = SentenceTransformer('keepitreal/vietnamese-sbert')
        logging.info("Đã tải SentenceTransformer model.")

        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNK_MAP_PATH):
            logging.info("Đang tải FAISS index và chunk map...")
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CHUNK_MAP_PATH, "r", encoding="utf-8") as f:
                chunk_map = json.load(f)
            logging.info("Đã tải FAISS index và chunk map.")
        else:
            logging.warning("FAISS index hoặc chunk map không tìm thấy. RAG sẽ không hoạt động.")
            faiss_index = None
            chunk_map = {}
    except Exception as e:
        logging.error(f"Lỗi khi tải vector engine: {e}")
        embed_model = None
        faiss_index = None
        chunk_map = {}

def retrieve_context(query, top_k=3):
    if not embed_model or not faiss_index or not chunk_map:
        logging.warning("Vector engine chưa được tải, bỏ qua RAG.")
        return "Không có ngữ cảnh RAG."

    try:
        query_vector = embed_model.encode([query])
        D, I = faiss_index.search(np.array(query_vector).astype('float32'), top_k)
        context = [chunk_map[str(idx)] for idx in I[0] if str(idx) in chunk_map]
        logging.info(f"Đã truy xuất ngữ cảnh RAG: {context}")
        return "\n".join(context)
    except Exception as e:
        logging.error(f"Lỗi khi truy xuất ngữ cảnh RAG: {e}")
        return "Lỗi khi truy xuất ngữ cảnh RAG."

def get_ai_response(user_input, chat_history):
    user_input = sanitize_string(user_input)
    rag_context = retrieve_context(user_input)
    system_prompt = f"Bạn là một trợ lý AI kỹ thuật có tên là AI Agent. Bạn phải trả lời các câu hỏi dựa trên các tài liệu đã được cung cấp (nếu có) và không được bịa đặt thông tin. Nếu bạn không biết, hãy nói rằng bạn không biết. Luôn giữ câu trả lời ngắn gọn và đi thẳng vào vấn đề.\n\n### Ngữ cảnh từ RAG:\n{rag_context}"

    # === LOGIC CẮT GIẢM TOKEN ===
    temp_history = list(chat_history) # Tạo bản sao để xử lý
    messages = [{"role": "system", "content": system_prompt}] + temp_history + [{"role": "user", "content": user_input}]

    current_tokens = count_tokens_simple(messages)

    while current_tokens > CONTEXT_LIMIT and len(temp_history) > 1:
        # Xóa cặp tin nhắn cũ nhất (user và assistant)
        temp_history.pop(0)
        temp_history.pop(0)
        messages = [{"role": "system", "content": system_prompt}] + temp_history + [{"role": "user", "content": user_input}]
        current_tokens = count_tokens_simple(messages)
        logging.info(f"Context trimmed. New token count: {current_tokens}")

    # Cập nhật lại chat_history trong session sau khi đã cắt giảm
    session['chat_history'] = temp_history

    reply = call_llm(messages)
    clean_reply = sanitize_string(reply)

    # Tính toán lại token count sau khi có phản hồi để chính xác hơn
    final_messages = messages + [{"role": "assistant", "content": clean_reply}]
    token_count = count_tokens_simple(final_messages)

    return clean_reply, token_count

# ===================== FLASK ROUTES (Phần Web) =====================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('chat_ui'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            with open(USERS_FILE, "r") as f:
                users = json.load(f)
            password_hash = hash_password(password)
            user_data = next((u for u in users if u['username'] == username and u['password_hash'] == password_hash), None)
            if user_data:
                session['username'] = username
                session['active_session_id'] = None # Phiên mới, chưa có ID
                session['chat_history'] = []
                logging.info(f"User '{username}' logged in.")
                return redirect(url_for('chat_ui'))
            else:
                return render_template('login.html', error="Sai tên đăng nhập hoặc mật khẩu")
        except FileNotFoundError:
            return render_template('login.html', error="Lỗi hệ thống: Không tìm thấy file người dùng.")
        except Exception as e:
            logging.error(f"Lỗi đăng nhập: {e}")
            return render_template('login.html', error="Lỗi hệ thống.")
    return render_template('login.html')

@app.route('/')
def chat_ui():
    if 'username' not in session: return redirect(url_for('login'))
    return render_template('chat.html',
                           username=session['username'],
                           session_id=session.get('active_session_id', 'Phiên Mới'))

@app.route('/ask', methods=['POST'])
def ask():
    if 'username' not in session: return jsonify({"error": "Chưa đăng nhập"}), 401
    user_message = request.json['message']
    
    # Lấy lịch sử chat hiện tại từ session, nếu không có thì là list rỗng
    chat_history = session.get('chat_history', [])

    reply_content, token_count = get_ai_response(user_message, chat_history)

    # Cập nhật lịch sử chat trong session
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": reply_content})
    session['chat_history'] = chat_history

    logging.info(f"User '{session['username']}' asked: '{user_message}' | AI replied: '{reply_content[:50]}...' (Tokens: {token_count})")
    return jsonify({"reply": reply_content, "token_count": token_count})

@app.route('/logout')
def logout():
    username = session.pop('username', None)
    if username:
        logging.info(f"User '{username}' logged out.")
    session.clear()
    return redirect(url_for('login'))

@app.route('/new_session', methods=['POST'])
def new_session_api():
    if 'username' not in session: return jsonify({"error": "Chưa đăng nhập"}), 401
    session['chat_history'] = []
    session['active_session_id'] = None # Reset ID của phiên đang hoạt động
    logging.info(f"User '{session['username']}' started a new session.")
    return jsonify({"status": "success"})

@app.route('/save_session', methods=['POST'])
def save_session_api():
    if 'username' not in session: return jsonify({"error": "Chưa đăng nhập"}), 401

    username = session['username']
    chat_history = session.get('chat_history', [])
    active_session_id = session.get('active_session_id')
    user_history_path = os.path.join(HISTORY_BASE_DIR, username)
    os.makedirs(user_history_path, exist_ok=True)

    session_name = request.json.get('session_name', '').strip()
    sanitized_name = "".join(c for c in session_name if c.isalnum() or c in (' ', '_')).rstrip()

    filename_base = None
    if active_session_id and active_session_id != 'Phiên Mới':
        # Nếu có active_session_id và không phải "Phiên Mới", sử dụng lại nó
        filename_base = active_session_id
        message = f"Đã cập nhật phiên: {filename_base}"
    elif sanitized_name:
        # Nếu không có active_session_id nhưng có tên mới được cung cấp
        filename_base = sanitized_name
        message = f"Đã lưu phiên mới: {filename_base}"
    else:
        # Nếu không có cả hai, tạo tên tự động
        filename_base = datetime.now().strftime('%Y%m%d_%H%M%S')
        message = f"Đã lưu phiên mới tự động: {filename_base}"
    
    filename = f"{filename_base}.json"
    session['active_session_id'] = filename_base # Cập nhật ID phiên đang hoạt động

    filepath = os.path.join(user_history_path, filename)
    with open(filepath, "w", encoding="utf-8") as f: json.dump(chat_history, f, ensure_ascii=False, indent=2)
    logging.info(f"User '{username}' saved session: {filename}")
    return jsonify({"status": "success", "message": message, "new_session_id": session.get('active_session_id')})


@app.route('/history/list', methods=['GET'])
def list_history_api():
    if 'username' not in session: return jsonify({"error": "Chưa đăng nhập"}), 401
    username = session['username']
    user_history_path = os.path.join(HISTORY_BASE_DIR, username)
    
    sessions = []
    if os.path.exists(user_history_path):
        for f in os.listdir(user_history_path):
            if f.endswith(".json"):
                sessions.append(os.path.splitext(f)[0]) # Lấy tên file không có đuôi .json
        sessions.sort(reverse=True) # Sắp xếp để các phiên mới nhất lên đầu
    
    logging.info(f"User '{username}' requested history list. Found {len(sessions)} sessions.")
    return jsonify(sessions)

@app.route('/history/load', methods=['POST'])
def load_history_api():
    if 'username' not in session: return jsonify({"error": "Chưa đăng nhập"}), 401
    session_id = request.json['session_id']
    username = session['username']
    filepath = os.path.join(HISTORY_BASE_DIR, username, f"{session_id}.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f: chat_history = json.load(f)
        session['chat_history'] = chat_history
        session['active_session_id'] = session_id # Đặt phiên này là phiên đang hoạt động
        logging.info(f"User '{username}' loaded session: {session_id}")
        return jsonify({"status": "success", "history": chat_history, "session_id": session_id})
    logging.warning(f"User '{username}' attempted to load non-existent session: {session_id}")
    return jsonify({"error": "Không tìm thấy phiên chat"}), 404

@app.route('/history/delete', methods=['POST'])
def delete_history_api():
    if 'username' not in session: return jsonify({"error": "Chưa đăng nhập"}), 401
    session_id = request.json['session_id']
    username = session['username']
    filepath = os.path.join(HISTORY_BASE_DIR, username, f"{session_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        logging.info(f"User '{username}' deleted session: {session_id}")
        return jsonify({"status": "success"})
    logging.warning(f"User '{username}' attempted to delete non-existent session: {session_id}")
    return jsonify({"error": "Không tìm thấy phiên chat"}), 404

@app.route('/history/rename', methods=['POST'])
def rename_history_api():
    if 'username' not in session: return jsonify({"error": "Chưa đăng nhập"}), 401
    old_id = request.json['old_id']
    new_name = request.json['new_name'].strip()
    username = session['username']
    user_path = os.path.join(HISTORY_BASE_DIR, username)

    sanitized_new_name = "".join(c for c in new_name if c.isalnum() or c in (' ', '_')).rstrip()
    if not sanitized_new_name:
        return jsonify({"error": "Tên mới không hợp lệ."}), 400

    old_filepath = os.path.join(user_path, f"{old_id}.json")
    new_filepath = os.path.join(user_path, f"{sanitized_new_name}.json")

    if os.path.exists(new_filepath):
        return jsonify({"status": "error", "error": "Tên mới đã tồn tại."}), 409
    if os.path.exists(old_filepath):
        os.rename(old_filepath, new_filepath)
        # Nếu phiên hiện tại đang hoạt động bị đổi tên, cập nhật active_session_id
        if session.get('active_session_id') == old_id:
            session['active_session_id'] = sanitized_new_name
        logging.info(f"User '{username}' renamed '{old_id}' to '{sanitized_new_name}'")
        return jsonify({"status": "success", "message": f"Đã đổi tên phiên từ '{old_id}' thành '{sanitized_new_name}'."})
    logging.warning(f"User '{username}' attempted to rename non-existent session: {old_id}")
    return jsonify({"status": "error", "error": "Không tìm thấy phiên chat"}), 404

@app.route('/rag_save', methods=['POST'])
def rag_save_api():
    if 'username' not in session: return jsonify({"error": "Chưa đăng nhập"}), 401
    username = session['username']
    chat_history = session.get('chat_history', [])

    if not chat_history:
        return jsonify({"status": "error", "message": "Không có lịch sử chat để gửi."}), 400

    rag_pending_user_dir = os.path.join(RAG_PENDING_DIR, username)
    os.makedirs(rag_pending_user_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rag_feedback_{timestamp}.json"
    filepath = os.path.join(rag_pending_user_dir, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
        logging.info(f"User '{username}' submitted chat for RAG feedback: {filename}")
        return jsonify({"status": "success", "message": "Phiên chat đã được gửi thành công để góp ý RAG. Cảm ơn phản hồi của bạn!"})
    except Exception as e:
        logging.error(f"Lỗi khi lưu feedback RAG: {e}")
        return jsonify({"status": "error", "message": f"Đã xảy ra lỗi khi gửi góp ý RAG: {e}"}), 500

# ===================== KHỞI ĐỘNG ỨNG DỤNG =====================
if __name__ == '__main__':
    APP_PORT = 5000
    kill_process_on_port(APP_PORT) # Gọi hàm để kill port trước khi khởi động
    time.sleep(1) # Đợi một chút để tiến trình cũ được giải phóng hoàn toàn
    load_vector_engine()
    print(f"🚀 Bắt đầu khởi động AI Agent trên http://0.0.0.0:{APP_PORT}")
    serve(app, host='0.0.0.0', port=APP_PORT, threads=8)
