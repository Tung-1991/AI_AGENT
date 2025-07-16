# FILE: download_model.py
import os
from sentence_transformers import SentenceTransformer

def download_embedding_model():
    """
    Kiểm tra và tải model embedding (bkai-foundation-models/vietnamese-bi-encoder)
    vào đúng thư mục 'models/embed' của dự án.
    """
    # --- Cấu hình ---
    model_name = "bkai-foundation-models/vietnamese-bi-encoder"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "embed")

    # --- Kiểm tra xem model đã tồn tại chưa ---
    config_file_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_file_path):
        print(f"✅ Model đã tồn tại tại: {model_path}")
        print("👍 Bỏ qua bước tải về.")
        return

    # --- Tải và lưu model nếu chưa có ---
    print(f"🔍 Không tìm thấy model. Bắt đầu tải '{model_name}'...")
    try:
        # Tạo thư mục nếu nó chưa tồn tại
        os.makedirs(model_path, exist_ok=True)
        
        # Tải model
        model = SentenceTransformer(model_name)
        
        # Lưu model vào đường dẫn đã định
        model.save(model_path)
        
        print(f"✅ Đã tải và lưu model thành công vào: {model_path}")
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi khi tải model: {e}")

if __name__ == "__main__":
    download_embedding_model()
