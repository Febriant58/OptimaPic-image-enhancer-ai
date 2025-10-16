from flask import Flask, request, render_template, send_file
import onnxruntime as rt
import numpy as np
from PIL import Image
import os, time, mimetypes, glob

# =========================================================
# Konfigurasi Aplikasi Flask
# =========================================================
app = Flask(__name__)

# TENTUKAN ROOT PATH DENGAN PASTI
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# Folder upload & hasil
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULTS_FOLDER = os.path.join('static', 'results')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Pastikan folder ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Tambahkan mime-type untuk PNG
mimetypes.add_type('image/png', '.png')

# =========================================================
# Inisialisasi Model ONNX
# =========================================================
ONNX_LOADED = False
session_pre = None
session_end = None

# --- PERBAIKAN: GUNAKAN PATH ABSOLUT UNTUK MODEL ---
MODEL_PRE_PATH = os.path.join(BASE_DIR, "esrgan-small-pre.onnx")
MODEL_END_PATH = os.path.join(BASE_DIR, "esrgan-small-end.onnx")

try:
    if not os.path.exists(MODEL_PRE_PATH) or not os.path.exists(MODEL_END_PATH):
        # Ini akan tampil di log Vercel jika gagal
        raise FileNotFoundError(f"Model ONNX tidak ditemukan di path: {MODEL_PRE_PATH}")

    session_pre = rt.InferenceSession(MODEL_PRE_PATH) # Gunakan path absolut
    session_end = rt.InferenceSession(MODEL_END_PATH) # Gunakan path absolut
    print("✅ Model ONNX berhasil dimuat.")
    ONNX_LOADED = True

except Exception as e:
    print(f"⚠️ Gagal memuat model ONNX. ERROR: {e}")
    # Jika ini gagal, 500 Internal Server Error akan muncul
    # Cek Log Vercel untuk melihat pesan error ini
# =========================================================
# Fungsi Peningkatan Citra dengan ONNX
# ... (sisanya sama)
# =========================================================
def enhance_image_onnx(img):
    if not ONNX_LOADED:
        raise Exception("Model ONNX belum dimuat. Pastikan file model tersedia.")

    img = img.convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0

    # HWC → NCHW
    img_tensor = np.transpose(img_np, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)

    # --- Model Pre ---
    input_pre_name = session_pre.get_inputs()[0].name
    output_pre = session_pre.run(None, {input_pre_name: img_tensor})[0]

    # --- Model End ---
    input_end_names = [inp.name for inp in session_end.get_inputs()]
    input_end_data = {
        input_end_names[0]: img_tensor,   # Input asli (LR)
        input_end_names[1]: output_pre    # Residual dari model pre
    }

    output_final = session_end.run(None, input_end_data)[0]

    # Postprocessing: NCHW → HWC dan denormalisasi
    output_final = np.clip(output_final[0] * 255.0, 0, 255).astype(np.uint8)
    output_final = np.transpose(output_final, (1, 2, 0))

    return Image.fromarray(output_final)


# =========================================================
# ROUTES
# =========================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if not ONNX_LOADED:
        return render_template('index.html', error="❌ Sistem Super-Resolution tidak aktif (model ONNX tidak ditemukan).")

    if 'file' not in request.files or not request.files['file'].filename:
        return render_template('index.html', error="⚠️ Tidak ada file yang dipilih.")

    file = request.files['file']

    try:
        img = Image.open(file.stream).convert('RGB')

        # Hapus hasil lama agar folder tidak menumpuk
        for old_file in glob.glob(os.path.join(app.config['RESULTS_FOLDER'], 'enhanced_*.png')):
            try:
                os.remove(old_file)
            except:
                pass

        for old_file in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], 'original_*.png')):
            try:
                os.remove(old_file)
            except:
                pass

        # Timer mulai
        start_time = time.perf_counter()

        # Proses peningkatan
        enhanced_img = enhance_image_onnx(img)

        # Timer selesai
        end_time = time.perf_counter()
        process_time = round(end_time - start_time, 2)

        # Simpan hasil
        unique_id = str(int(time.time()))
        original_filename = f'original_{unique_id}.png'
        enhanced_filename = f'enhanced_{unique_id}.png'

        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        enhanced_path = os.path.join(app.config['RESULTS_FOLDER'], enhanced_filename)

        img.save(original_path)
        enhanced_img.save(enhanced_path)

        # Kirim path relatif ke template
        original_image = f'uploads/{original_filename}'
        enhanced_image = f'results/{enhanced_filename}'

        return render_template(
            'index.html',
            original_image=original_image,
            enhanced_image=enhanced_image,
            process_time=process_time,
            upscale_factor=4
        )

    except Exception as e:
        print(f"❌ Error saat memproses gambar: {e}")
        return render_template('index.html', error=f"Terjadi kesalahan: {str(e)}")


@app.route('/download/<filename>')
def download(filename):
    """Download hasil citra yang telah ditingkatkan"""
    if 'enhanced_' not in filename or '..' in filename:
        return "File tidak valid.", 404

    # Pastikan file diambil dari folder RESULTS_FOLDER
    path = os.path.join(app.config['RESULTS_FOLDER'], filename)

    try:
        return send_file(path, as_attachment=True, download_name="Enhanced_Image_SR.png")
    except FileNotFoundError:
        return "File hasil tidak ditemukan.", 404


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == '__main__':
    print("Aplikasi berjalan di http://127.0.0.1:5000")
    app.run(debug=True)
```
eof

---

### Langkah Verifikasi Kritis

Setelah Anda mengganti `app.py` dengan versi di atas:

1.  **Commit dan Push Ulang:** Lakukan `git add .`, `git commit`, dan `git push` ulang ke GitHub.
2.  **Cek Log Vercel:** Jika `500 Internal Server Error` masih muncul, segera **cek *Deployment Logs* di *dashboard* Vercel** Anda. Pesan *error* Python yang sebenarnya (seperti `FileNotFoundError` atau `ImportError`) akan muncul di sana.

**CATATAN PENTING TENTANG DEPENDENSI:**
Pastikan `requirements.txt` Anda mencantumkan **dependensi yang sesuai untuk Vercel**:

```
Flask
Pillow
numpy
onnxruntime
