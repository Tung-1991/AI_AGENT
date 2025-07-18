import os
import subprocess
import platform
import sys
import time
from datetime import datetime

MODEL_PATH = os.path.expanduser("~/AIagent/models/llm/llama-3-13b-instruct.Q4_K_M.gguf")
MIG_DEVICE_UUID = "MIG-c14a1f4b-3025-5c63-9b6f-2de7c2227f03" 
N_GPU_LAYERS = -1
N_CTX = 8192
PORT = 8000
LOG_DIR = "logs"

def kill_process_on_port(port):
    print(f"Kiểm tra port {port}...")
    try:
        if platform.system() in ["Linux", "Darwin"]:
            command = f"lsof -t -i:{port}"
            pids_output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
            if pids_output:
                subprocess.run(f"kill -9 {pids_output}", shell=True, check=True)
        print(f"✅ Port {port} đã sẵn sàng.")
    except Exception:
        print(f"✅ Port {port} đang rảnh.")

def start_server():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = os.path.join(LOG_DIR, f"server_13b_{datetime.now().strftime('%Y-%m-%d')}.log")

    command = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", MODEL_PATH,
        "--n_gpu_layers", str(N_GPU_LAYERS),
        "--n_ctx", str(N_CTX),
        "--port", str(PORT)
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = MIG_DEVICE_UUID
    
    print(f"\n🚀 Khởi động server trên MIG device: {MIG_DEVICE_UUID}")
    print(f"   Log sẽ được ghi tại: {log_filename}")

    try:
        with open(log_filename, 'a') as log_file:
            subprocess.Popen(command, stdout=log_file, stderr=log_file, env=env)
        print(f"✅ Server đã khởi động trên port {PORT}. Chạy 'tail -f {log_filename}' để xem log.")
    except Exception as e:
        print(f"\n❌ Lỗi khởi động server: {e}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Lỗi: Không tìm thấy model tại '{MODEL_PATH}'.")
        sys.exit(1)
        
    kill_process_on_port(PORT)
    time.sleep(1)
    start_server()import os
