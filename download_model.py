# FILE: download_model.py
import os
from sentence_transformers import SentenceTransformer

# Danh sách các model cần tải
MODELS = {
    "vi": "bkai-foundation-models/vietnamese-bi-encoder",
    "en": "sentence-transformers/all-MiniLM-L6-v2" # Model đa ngôn ngữ mạnh, nhẹ
}

base_dir = os.path.dirname(os.path.abspath(__file__))

def download_and_save(model_name, save_dir):
    if os.path.exists(os.path.join(save_dir, "modules.json")):
        print(f"✅ Model '{model_name}' đã tồn tại tại: {save_dir}")
    else:
        print(f"🔍 Bắt đầu tải và lưu model '{model_name}'...")
        os.makedirs(save_dir, exist_ok=True)
        model = SentenceTransformer(model_name)
        model.save(save_dir)
        print(f"✅ Đã tải model '{model_name}' thành công tại: {save_dir}")

if __name__ == "__main__":
    print("--- Bắt đầu kiểm tra và tải các model embedding ---")
    for lang_code, model_name in MODELS.items():
        # Đổi tên thư mục lưu để phân biệt
        model_path = os.path.join(base_dir, "models", f"embed_{lang_code}")
        download_and_save(model_name, model_path)
    print("--- Hoàn tất quá trình tải model ---")
