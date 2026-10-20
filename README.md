# Klasifikasi Pendorong (Drivers) Deforestasi menggunakan CNN 🌳📡

Proyek *Computer Vision* dan Penginderaan Jauh untuk mengklasifikasikan penyebab langsung (*direct drivers*) hilangnya tutupan hutan berdasarkan analisis citra satelit **Sentinel-1 (Radar)** dan **Sentinel-2 (Optik)**.

## 📌 Deskripsi Proyek
Meskipun sistem peringatan dini deforestasi memberi tahu *kapan* dan *di mana* hutan hilang, mengetahui **apa penyebabnya** (misal: pertanian kecil, perkebunan besar, pertambangan, penebangan/logging, kebakaran, atau faktor alam) membutuhkan analisis lanjutan. 

Proyek ini menggunakan arsitektur *Convolutional Neural Network (CNN)* yang dibangun dengan TensorFlow/Keras untuk mengklasifikasikan *patch* citra satelit ke dalam 7 kelas penyebab deforestasi, dengan mempertimbangkan tantangan *imbalanced dataset* (kelas pertanian sangat dominan).

## 🛠️ Stack Teknologi
- **Data Sumber:** Google Earth Engine (Sentinel-1 VV/VH & Sentinel-2 B2,B3,B4,B8).
- **Pemrosesan Citra:** `rasterio`, NumPy.
- **Deep Learning:** TensorFlow & Keras.
- **Evaluasi Model:** Scikit-Learn (Macro F1-Score & Confusion Matrix).

## 🚀 Cara Menjalankan (via Google Colab)
Karena proyek ini membutuhkan pengunduhan data GeoTIFF dan komputasi GPU:
1. Buka [Google Colab](https://colab.research.google.com/).
2. Unggah file `notebooks/01_ambil_citra_gee.ipynb`. Jalankan untuk melakukan otentikasi GEE dan mengekspor GeoTIFF ke Google Drive.
3. Unggah file `notebooks/02_latih_cnn.ipynb`. Jalankan untuk memproses file TIFF dan melatih model CNN.
   - *Catatan:* *Notebook* pelatihan sudah dilengkapi dengan *script* dummy dataset agar kode dapat disimulasikan sukses meskipun data TIFF mentah belum terkumpul penuh.

## 📊 Keterbatasan
- Klasifikasi ini mengandalkan *Direct Drivers* visual dari satelit. Penyebab tidak langsung atau aktivitas berskala kecil di bawah kanopi (seperti *artisanal logging* / pembuatan arang) tidak dapat dideteksi hanya dari resolusi Sentinel (10 meter).
- Terdapat latensi sistem hingga satu bulan karena tutupan awan atau jadwal *flyover* satelit.
