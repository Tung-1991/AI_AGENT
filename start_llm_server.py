# FILE: start_llm_server.py (Phiên bản chuẩn)
import os
import subprocess
import platform
import sys
import time
from datetime import datetime

# --- Cấu hình Server ---
MODEL_PATH = os.path.expanduser("~/AIagent/models/llm/llama-3-13b-instruct.Q4_K_M.gguf")
N_GPU_LAYERS = -1
N_CTX = 8192      # <-- Thêm: Khai báo Context Window cho model
PORT = 8000
LOG_DIR = "logs"

def kill_process_on_port(port):
    """
    Tìm và dừng bất kỳ tiến trình nào đang lắng nghe trên một port cụ thể.
    Đã sửa lỗi logic để hoạt động đúng trên cả Windows, Linux và macOS.
    """
    print(f"Kiểm tra và dừng tiến trình cũ trên port {port}...")
    try:
        system = platform.system()
        if system == "Windows":
            command = f"netstat -aon | findstr :{port}"
            output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.DEVNULL)
            if "LISTENING" in output:
                pid = output.strip().split()[-1]
                subprocess.run(f"taskkill /F /PID {pid}", shell=True, check=True)
        # SỬA LỖI: Dùng elif để phân tách logic cho các hệ điều hành khác nhau
        elif system in ["Linux", "Darwin"]: # "Darwin" là tên của macOS
            command = f"lsof -t -i:{port}"
            pids_output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
            # Xử lý trường hợp không có PID nào được trả về
            pids = pids_output.split('\n') if pids_output else []
            for pid in filter(None, pids):
                subprocess.run(f"kill -9 {pid}", shell=True, check=True)
        
        print(f"✅ Đã dọn dẹp port {port}.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"✅ Port {port} đang rảnh.")
    except Exception as e:
        print(f"🟡 Lỗi khi dọn dẹp port: {e}")

def start_server():
    """Khởi động server LLM trong một tiến trình chạy nền."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = os.path.join(LOG_DIR, f"server_{datetime.now().strftime('%Y-%m-%d')}.log")

    command = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", MODEL_PATH,
        "--n_gpu_layers", str(N_GPU_LAYERS),
        "--n_ctx", str(N_CTX), # <-- Thêm: Truyền tham số n_ctx vào server
        "--port", str(PORT)
    ]

    print("\n🚀 Bắt đầu khởi động server LLM chạy nền...")
    print(f"   Mọi output sẽ được ghi vào file: {log_filename}")
    print(f"   Để xem log trực tiếp, mở terminal khác và gõ: tail -f {log_filename}")

    try:
        with open(log_filename, 'a') as log_file:
            # Dùng start_new_session=True trên Linux/macOS để tiến trình không bị kill khi terminal đóng
            kwargs = {"start_new_session": True} if platform.system() != "Windows" else {}
            subprocess.Popen(command, stdout=log_file, stderr=log_file, **kwargs)
        
        print(f"✅ Server đã được khởi động thành công trên port {PORT}.")
    except Exception as e:
        print(f"\n❌ Lỗi nghiêm trọng khi khởi động server: {e}")

if __name__ == "__main__":
    kill_process_on_port(PORT)
    time.sleep(1) # Chờ một chút để port được giải phóng hoàn toàn
    start_server()
