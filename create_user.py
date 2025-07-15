# FILE: create_user.py
import json
import hashlib
import os
from getpass import getpass

USERS_FILE = os.path.join("data", "users", "users.json")
os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user():
    print("--- TẠO NGƯỜI DÙNG MỚI ---")
    username = input("Nhập username: ").strip()
    password = getpass("Nhập mật khẩu: ").strip()
    password_confirm = getpass("Xác nhận mật khẩu: ").strip()

    if password != password_confirm:
        print("Lỗi: Mật khẩu không khớp.")
        return

    if not os.path.exists(USERS_FILE):
        users = []
    else:
        with open(USERS_FILE, "r") as f:
            try:
                users = json.load(f)
            except json.JSONDecodeError:
                users = []

    if any(u["username"] == username for u in users):
        print(f"Lỗi: Người dùng '{username}' đã tồn tại.")
        return

    new_user = {
        "username": username,
        "password_hash": hash_password(password)
    }
    users.append(new_user)

    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

    print(f"✅ Đã tạo thành công người dùng '{username}'.")

if __name__ == "__main__":
    create_user()
