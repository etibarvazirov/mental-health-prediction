# 🧠 Stress və Psixoloji Sağlamlıq Proqnoz Sistemi  
### *Multimodal (MLP + BERT + Fusion) Süni İntellekt Modeli ilə Stress Səviyyəsinin Analizi*

---

## 📌 Layihənin Təsviri

Bu layihənin məqsədi istifadəçilərin yuxu, həyat tərzi və emosional vəziyyətinə dair məlumatları analiz edərək **stress səviyyəsini avtomatik proqnozlaşdırmaqdır**.

Model aşağıdakı üç süni intellekt sistemini birləşdirir:

- **MLP (Multi-Layer Perceptron)** – yuxu və həyat tərzi məlumatlarından 128-dimensional embedding çıxarır.
- **BERT (DistilBERT)** – istifadəçinin yazdığı mətni 768-dimensional emosional embedding-ə çevirir.
- **Fusion Model** – hər iki embedding-i və numeric xüsusiyyətləri birləşdirərək stress səviyyəsini hesablayır.

Bu sistem psixoloji rifahın erkən aşkarlanması, fərdi analiz və sağlamlıqla bağlı qərarların dəstəklənməsi üçün real tətbiq dəyəri olan AI həllidir.

---

## 🌍 Layihənin Əhəmiyyəti

- Stress XXI əsrdə ən geniş yayılmış psixoloji problemlərdən biridir.
- Uzunmüddətli stress öyrənməyə, məhsuldarlığa və sağlamlığa ciddi təsir edir.
- Erkən proqnoz sistemləri psixoloji problemlərin qarşısını almaqda böyük rol oynayır.
- Layihə təhsil, səhiyyə, iş mühiti və psixoloji yardım kimi sahələrdə istifadə edilə bilər.

---

## 📦 Repo Strukturu

mental-health-prediction/
│
├── app.py # Streamlit Web App (main file)
├── requirements.txt # Python dependency-lər
│
├── data/
│ ├── sleep_encoded.csv
│ ├── survey.csv
│
├── models/
│ ├── fusion_model_config.json
│
├── images/
│ ├── fig4_shap_clean.png
│ ├── fig1_prediction_vs_actual.png
│ ├── fig3_pca.png
│ ├── fig2_model_comparison.png
│ ├── fusion_architecture.png
│
└── README.md

---

## 🧬 Model Arxitekturası

### 🔹 **MLP Projection Layer**
Sleep & Lifestyle numeric məlumatlarını 128-dimensional embedding-ə çevirir.

### 🔹 **BERT (distilbert-base-uncased)**
İstifadəçinin yazdığı mətndən emosional xüsusiyyətləri çıxarır → 768 dim.

### 🔹 **Fusion Model Arxitekturası**
```text
Input: 768 (BERT) + 128 (MLP) + 12 (numeric) = 908 dim

908 → Linear → 256 → ReLU → Dropout(0.2)
256 → Linear → 128 → ReLU → Dropout(0.2)
128 → Linear → 1  (Stress səviyyəsi)


🚀 Tətbiqin İşə Salınması

1️⃣ Repo-nu klonla

git clone https://github.com/YOUR_USERNAME/mental-health-prediction.git
cd mental-health-prediction

2️⃣ Dependency-ləri quraşdır
pip install -r requirements.txt

3️⃣ Streamlit tətbiqini işə sal
streamlit run app.py

