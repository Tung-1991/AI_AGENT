# download_model.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
model.save("/tmp/embed_bkai")
print("✅ Đã tải và lưu model vào /tmp/embed_bkai")
