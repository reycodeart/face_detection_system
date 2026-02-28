# 🧠 Face Detection System

![Face Detection](https://img.shields.io/badge/Face%20Detection-System-blue)
![Python](https://img.shields.io/badge/Python-3.x-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-brightgreen)

**Face Detection System**, Python ve OpenCV kullanılarak geliştirilmiş modüler, GUI destekli ve performans karşılaştırmalı bir yüz tespit projesidir. Hem klasik Haar Cascade hem de modern Deep Learning tabanlı DNN (SSD + ResNet) algoritmalarını içerir.

---

## 🧩 Proje Özeti

Bu proje ile:

✔ Görüntü veya video içindeki yüzler tespit edilir  
✔ İki farklı yöntem karşılaştırılır (Haar vs DNN)  
✔ FPS ve yüz sayısı ölçülür  
✔ Görsel performans karşılaştırma grafikleri üretilir  
✔ Kullanıcı dostu GUI ile işlem yapılır  

---

## 💡 Özellikler

### 🔎 Algoritmalar
| Yöntem | Açıklama |
|--------|----------|
| **Haar Cascade (OpenCV)** | Klasik yüz tespiti, daha hızlı ancak ışık ve açı hassas |
| **DNN (SSD + ResNet)** | Modern deep learning tabanlı, daha yüksek doğruluk |

---

### 🧪 Desteklenen Modlar

✔ Tek bir görüntüde yüz tespiti  
✔ Video dosyasında yüz tespiti  
✔ Gerçek zamanlı webcam yüz tespiti  
✔ Yöntem seçimi (Haar / DNN)  
✔ Performans grafikleri  

---

## 🧠 Teknolojiler

Bu proje aşağıdaki teknolojilerle yazılmıştır:

- Python 3.x  
- OpenCV  
- NumPy  
- Matplotlib (performans grafikleri)  
- Tkinter (GUI arayüz)  
- Pillow  

---

## 📁 Proje Yapısı

```text
face_detection_system/
│
├── models/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── detectors/
│   ├── haar_detector.py
│   └── dnn_detector.py
│
├── performance/
│   └── metrics.py
│
├── gui.py
├── main.py
└── requirements.txt
