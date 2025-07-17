# FILE: create_user.py (Phiên bản nâng cấp với Salt)
import json
import hashlib
import os
import secrets
from getpass import getpass

# --- Cấu hình đường dẫn ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "data", "users", "users.json")
os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)

def hash_password(password, salt):
    """Băm mật khẩu với salt sử dụng SHA256."""
    return hashlib.sha256(salt.encode() + password.encode()).hexdigest()

def create_user():
    """Tạo người dùng mới với mật khẩu đã được băm và 'salt'."""
    print("--- TẠO NGƯỜI DÙNG MỚI (BẢO MẬT NÂNG CAO) ---")
    username = input("Nhập username: ").strip()
    
    if not username:
        print("Lỗi: Tên người dùng không được để trống.")
        return
        
    password = getpass("Nhập mật khẩu: ").strip()
    password_confirm = getpass("Xác nhận mật khẩu: ").strip()

    if not password:
        print("Lỗi: Mật khẩu không được để trống.")
        return

    if password != password_confirm:
        print("Lỗi: Mật khẩu không khớp.")
        return

    # Tải danh sách người dùng hiện có
    if not os.path.exists(USERS_FILE):
        users = []
    else:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            try:
                users = json.load(f)
            except json.JSONDecodeError:
                users = [] # Tạo mới nếu file bị lỗi

    # Kiểm tra username đã tồn tại chưa
    if any(u.get("username") == username for u in users):
        print(f"Lỗi: Người dùng '{username}' đã tồn tại.")
        return

    # Tạo salt ngẫu nhiên cho người dùng mới
    salt = secrets.token_hex(16)
    
    # Băm mật khẩu với salt
    password_hash = hash_password(password, salt)

    # Tạo đối tượng người dùng mới với cấu trúc chuẩn
    new_user = {
        "username": username,
        "password_hash": password_hash,
        "salt": salt  # Thêm trường salt
    }
    users.append(new_user)

    # Lưu lại vào file
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

    print(f"✅ Đã tạo thành công người dùng '{username}' với cơ chế bảo mật nâng cao.")

if __name__ == "__main__":
    create_user()
