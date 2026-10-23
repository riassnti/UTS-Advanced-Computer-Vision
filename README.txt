PROYEK KLASIFIKASI KEMATANGAN PISANG - MobileNetV4 Conv Small
============================================================

Konfigurasi:
- Arsitektur: MobileNetV4 Conv Small pretrained
- Input model: 224 x 224 piksel
- Dataset sumber boleh berukuran 500 x 500; resize dilakukan otomatis
- Pembagian data: 70% train, 15% validasi, 15% test (stratified)
- Early stopping dan laporan evaluasi otomatis
- Dapat berjalan di CPU; GPU digunakan otomatis jika tersedia

STRUKTUR FOLDER DATASET
Letakkan dataset Anda seperti ini (nama folder kelas menjadi label):

data/
  banana_dataset_300/
    belum_matang/
      gambar1.jpg
    matang/
      gambar1.jpg
    terlalu_matang_busuk/
      gambar1.jpg

Folder kelas harus berisi gambar. Gunakan nama kelas yang konsisten. Dataset tidak disertakan dalam ZIP ini; salin dataset Anda ke folder data/banana_dataset_300/.

INSTALASI (Windows / Anaconda Prompt atau terminal VS Code)
1. Ekstrak ZIP.
2. Buka terminal di folder proyek.
3. Aktifkan environment yang digunakan:
   conda activate banana_cnn_fix
4. Install dependensi:
   pip install -r requirements.txt

Jika environment belum ada, buat environment Python 3.10 atau 3.11 terlebih dahulu, lalu aktifkan dan install requirements.txt.

MENJALANKAN TRAINING
Dari folder proyek:
   python -u train_new.py --data-dir "data\banana_dataset_300" --epochs 15

Opsional, mengatur batch size, learning rate, patience, dan lokasi output:
   python -u train_new.py --data-dir "data\banana_dataset_300" --epochs 15 --batch-size 4 --lr 1e-5 --patience 4 --output-dir .

HASIL TRAINING
- models/banana_mobilenetv4.pth       model terbaik berdasarkan validation loss
- models/banana_mobilenetv4_latest.pth checkpoint epoch terakhir
- class_names.json                   urutan nama kelas
- reports/training_history.png       grafik loss dan akurasi
- reports/confusion_matrix.png       confusion matrix
- reports/class_metrics.png          metrik per kelas
- reports/test_metrics.txt           ringkasan evaluasi test
- reports/classification_report.json laporan precision/recall/F1
- reports/predictions_test.csv       prediksi data test
- reports/training_history.json      data untuk grafik

PREDIKSI SATU GAMBAR
Setelah training selesai:
   python predict_image.py --image "data\contoh_pisang.jpg"

Jika model disimpan di lokasi lain, gunakan --model dan --classes.

CATATAN
- Pada penggunaan pertama, timm dapat mengunduh bobot pretrained dari internet.
- Input 224x224 tidak mengubah file dataset asli yang berukuran 500x500.
- Hasil akurasi tidak dijamin; evaluasi bergantung pada kualitas dan jumlah data.
- Jika muncul error instalasi, pastikan terminal menggunakan environment Python yang sama saat menjalankan script.
