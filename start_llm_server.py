# FILE: start_llm_server.py
import os
import subprocess
import platform
import sys
import time
from datetime import datetime

# --- Cấu hình Server ---
MODEL_PATH = os.path.expanduser("~/AIagent/models/llm/llama-3-13b-instruct.Q4_K_M.gguf")
N_GPU_LAYERS = -1
PORT = 8000
LOG_DIR = "logs"

def kill_process_on_port(port):
    print(f"Kiểm tra và dừng tiến trình cũ trên port {port}...")
    try:
        if platform.system() == "Windows":
            command = f"netstat -aon | findstr :{port}"
            output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.DEVNULL)
            if "LISTENING" in output:
                pid = output.strip().split()[-1]
                os.system(f"taskkill /F /PID {pid}")
        else: # Linux & macOS
            command = f"lsof -t -i:{port}"
            pids = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.DEVNULL).strip().split('\n')
            for pid in filter(None, pids):
                os.system(f"kill -9 {pid}")
        print(f"✅ Đã dọn dẹp port {port}.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"✅ Port {port} đang rảnh.")
    except Exception as e:
        print(f"🟡 Lỗi khi dọn dẹp port: {e}")

def start_server():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = os.path.join(LOG_DIR, f"server_{datetime.now().strftime('%Y-%m-%d')}.log")

    command = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", MODEL_PATH,
        "--n_gpu_layers", str(N_GPU_LAYERS),
        "--port", str(PORT)
    ]
    
    print("\n🚀 Bắt đầu khởi động server LLM chạy nền...")
    print(f"   Mọi output sẽ được ghi vào file: {log_filename}")
    print(f"   Để xem log trực tiếp, mở terminal khác và gõ: tail -f {log_filename}")

    try:
        with open(log_filename, 'a') as log_file:
            subprocess.Popen(command, stdout=log_file, stderr=log_file, start_new_session=True)
        print(f"✅ Server đã được khởi động thành công trên port {PORT}.")
    except Exception as e:
        print(f"\n❌ Lỗi nghiêm trọng khi khởi động server: {e}")

if __name__ == "__main__":
    kill_process_on_port(PORT)
    time.sleep(1) 
    start_server()
