from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # 1. Import ini
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Inisialisasi Aplikasi API
app = FastAPI(title="Berita Classifier API", description="API Prediksi Kategori Berita")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Mengizinkan akses dari semua URL/File lokal
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Skema Input JSON (Wajib memiliki judul dan content)
class BeritaInput(BaseModel):
    judul: str
    content: str

# 3. Definisi Kategori Utama
kategori_utama = [
    'Politik', 'Hukum & Kriminal', 'Ekonomi',
    'Bencana & Insiden', 'Internasional',
    'Gaya Hidup & Hiburan', 'Olahraga'
] #[cite: 1]

# 4. Load Model dan Tokenizer
# Ganti 'model_name' dengan path direktori './results' tempat Anda menyimpan model hasil training
# model_name = "indobenchmark/indobert-base-p1" #[cite: 1]
model_name = "rabbanee/indobert-klasifikasi-berita" #[cite: 1]

print("Memuat tokenizer dan model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(kategori_utama),
    problem_type="multi_label_classification"
) #[cite: 1]

# Pindahkan ke GPU jika tersedia, lalu set ke mode evaluasi
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 5. Endpoint Prediksi dengan metode POST
@app.post("/predict")
def predict(berita: BeritaInput, threshold: float = 0.3):

    try:
        # Menggabungkan judul dan konten menjadi satu teks
        teks_input = berita.judul + " " + berita.content #[cite: 1]

        # Proses tokenisasi teks input
        inputs = tokenizer(
            teks_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ) #[cite: 1]

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Prediksi (Inferensi)
        with torch.no_grad():
            outputs = model(**inputs)
            # Konversi nilai logits menjadi persentase probabilitas menggunakan Sigmoid
            probs = torch.sigmoid(outputs.logits)[0] #[cite: 1]

        hasil_detail = {}
        kategori_terpilih = []

        # Cek probabilitas tiap kategori
        for i, prob in enumerate(probs):
            nama_kategori = kategori_utama[i]
            persen = round(prob.item() * 100, 2)
            hasil_detail[nama_kategori] = f"{persen}%"

            # Jika lolos threshold, masukkan ke hasil akhir
            if prob.item() >= threshold: #[cite: 1]
                kategori_terpilih.append(nama_kategori)

        # Fallback jika probabilitas semuanya di bawah threshold
        if not kategori_terpilih:
             idx_max = probs.argmax().item()
             kategori_terpilih.append(kategori_utama[idx_max])

        # Output JSON Response
        return {
            "judul": berita.judul,
            "kategori_prediksi": kategori_terpilih,
            "probabilitas_detail": hasil_detail
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
