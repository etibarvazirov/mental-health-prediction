import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Stress və Psixoloji Sağlamlıq Proqnozu", layout="wide")

st.title("🧠 Stress və Psixoloji Sağlamlıq Proqnoz Sistemi")
st.write("""
Bu sistem **Fusion Neural Network (BERT + MLP + Numeric Features)** modeli ilə
yuxu, həyat tərzi və emosional mətn məlumatlarını birləşdirərək **stress səviyyəsini proqnozlaşdırır**.
""")
st.markdown("---")

st.markdown("""

## 🎯 Layihənin Məqsədi
Bu tətbiqin əsas məqsədi:
- 🌟 Stressi erkən müəyyənləşdirmək  
- 🚨 Yüksək riskli hallarda xəbərdarlıq təmin etmək  
- 🧘‍♂️ Öyrənən və çalışan insanlar üçün psixoloji rifahı artırmaq

---

## 🔍 Model nəyə əsaslanır?
Süni intellekt modeli istifadəçidən aşağıdakı əsas məlumatları alır:

- **😴 Yuxu müddəti (Sleep Duration)**
- **🌙 Yuxu keyfiyyəti (Quality of Sleep)**
- **💓 Ürək döyüntüsü (Heart Rate)**
- **💪 Fiziki aktivlik səviyyəsi (Physical Activity Level)**
- **🩸 Qan təzyiqi (Systolic / Diastolic BP)**
- **✍️ Emosional mətn (BERT tekst analizi)**

Bu 6 əsas faktor stress səviyyəsini müəyyən edən parametrlərin böyük hissəsini təşkil edir.

---

## ⚙️ Model necə işləyir?
Sistem üç ayrı komponentin gücünü birləşdirir:

- **1) Numeric Features Model** — yuxu + aktivlik + təzyiq + ürək döyüntüsü  
- **2) Text Emotion Model (BERT)** — istifadəçinin yazdığı mətnin emosional tonunu çıxarır  
- **3) Fusion Model** — hər iki modelin nəticələrini birləşdirərək yekun stress göstərir  

---

## 📊 Nəticələr
Model çıxışı 0–1 arası olur və belə şərh edilir:

- 🟢 **0.00 – 0.33 → Aşağı risk**
- 🟡 **0.34 – 0.66 → Orta risk**
- 🔴 **0.67 – 1.00 → Yüksək risk**

Aşağıdakı bölmədən məlumatları daxil edin və stress səviyyənizi yoxlayın.
""")



# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

    scaler_mean = np.load("models/scaler_mean.npy")
    scaler_std = np.load("models/scaler_std.npy")

    class MLPProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 128)
        def forward(self, x):
            return self.proj(x)

    mlp = MLPProjection()
    mlp.load_state_dict(torch.load("models/mlp_projection.pth", map_location="cpu"))
    mlp.eval()

    class FusionModel(nn.Module):
        def __init__(self, input_dim=908):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 1)
            )
        def forward(self, x):
            return self.net(x)

    fusion = FusionModel()
    fusion.load_state_dict(torch.load("models/fusion_model.pth", map_location="cpu"))
    fusion.eval()

    return tokenizer, bert, scaler_mean, scaler_std, mlp, fusion


tokenizer, bert_model, scaler_mean, scaler_std, mlp_model, fusion_model = load_models()


# =========================================================
# FUNCTIONS
# =========================================================

def get_bert_embedding(text):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        out = bert_model(**encoded)
    return out.last_hidden_state[:, 0, :].numpy()[0]  # CLS token


def scale_numeric(x):
    return (x - scaler_mean) / scaler_std


def fusion_predict(text, numeric, sleep_duration):
    bert_emb = get_bert_embedding(text)
    numeric_scaled = scale_numeric(numeric)

    sd_tensor = torch.tensor([[sleep_duration]], dtype=torch.float32)
    mlp_emb = mlp_model(sd_tensor).detach().numpy()[0]

    fusion_input = np.concatenate([bert_emb, mlp_emb, numeric_scaled], axis=0)
    x = torch.tensor(fusion_input, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        return fusion_model(x).item()


# =========================================================
# PRESETS
# =========================================================
PRESETS = {
    "Aşağı Stress": {
        "sleep": 8.2,
        "quality": 8,
        "hr": 68,
        "activity": 8,
        "sbp": 112,
        "dbp": 71,
        "text": "Bu gün özümü çox rahat və pozitiv hiss edirəm."
    },

    "Orta Stress": {
        "sleep": 6.1,
        "quality": 5,
        "hr": 80,
        "activity": 4,
        "sbp": 124,
        "dbp": 82,
        "text": "Bugün normal keçdi, amma bir az yorğunam."
    },

    "Yüksək Stress": {
        "sleep": 4.2,
        "quality": 3,
        "hr": 103,
        "activity": 2,
        "sbp": 142,
        "dbp": 94,
        "text": "Çox stress altındayam, yuxusuzam, narahatlıq hiss edirəm."
    },

    "İmtahan stresli tələbə": {
        "sleep": 4.8,
        "quality": 4,
        "hr": 89,
        "activity": 2,
        "sbp": 118,
        "dbp": 76,
        "text": "Sabah imtahanım var və çox stress hiss edirəm."
    },

    "İdmançı": {
        "sleep": 7.6,
        "quality": 9,
        "hr": 58,
        "activity": 10,
        "sbp": 114,
        "dbp": 66,
        "text": "Məşq əla keçdi, enerjiliyəm."
    }
}



# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Parametrlər")

mode = st.sidebar.radio("Veri daxil etmə üsulu:", ["Preset", "Manual"])

preset_name = None
if mode == "Preset":
    preset_name = st.sidebar.selectbox("Hazır ssenari seç:", list(PRESETS.keys()))

st.markdown("""
---

## ℹ️ Manual Dəyərlər Üçün Açıqlama

Aşağıdakı parametrlər stress səviyyəsinin proqnozlaşdırılması üçün istifadə olunur.
Hər dəyişənin mənası və tipik aralıqları belədir:

---

### 😴 **Sleep Duration (Yuxu müddəti) — 0–12 saat**
- 7–9 saat → sağlam aralıq  
- 5–6 saat → orta risk  
- 0–4 saat → yüksək stresslə korelyasiya edir  

---

### 🌙 **Quality of Sleep (Yuxu keyfiyyəti) — 1–10**
- 8–10 → keyfiyyətli yuxu  
- 5–7 → orta yuxu  
- 1–4 → qeyri-kafi, stres artır  

---

### 💓 **Heart Rate (Ürək döyüntüsü) — 40–130 BPM**
- 55–75 → normal  
- 76–90 → orta  
- 90+ → simptomatik stress və ya yorğunluq göstəricisi  

---

### 💪 **Physical Activity Level — 1–10**
- 1–3 → oturaq həyat tərzi  
- 4–6 → orta aktivlik  
- 7–10 → yüksək aktivlik (stressi azaldır)  

---

### 🩸 **Blood Pressure (Sistolik / Diastolik)**
- Normal: **110–120 / 70–80**  
- Orta risk: **125–135 / 80–90**  
- Yüksək risk: **140+ / 90+**

Yüksək təzyiq stress proqnozunu artırır.

---

### ✍️ **Text Input (Emosional təsvir)**
Model mətnin emosional tonunu BERT ilə qiymətləndirir:

- “özümü yaxşı hiss edirəm”, “enerjiliyəm” → stressi azaldır  
- “narahatam”, “stres”, “yuxusuzam” → stressi artırır  

---

### 👫 **Gender (Cins)**
Modeldə cinsi yalnız binary şəkildə istifadə edirik:
- Kişi → 0  
- Qadın → 1  

Cinsin təsiri minimaldır.

---

### 💼 **Occupation (Peşə Kodu) — 0–20**
Bu xüsusiyyət datasetdən gəlir və **sadəcə kateqoriya identifikatorudur**.
Faktiki peşəni əks etdirmir, yalnız qrup kimi istifadə olunur.

Təsir gücü çox zəifdir.

---

### 🧍‍♂️ **BMI Category (0–5)**
- 0 → Aşağı çəki  
- 1 → Normal  
- 2 → Yüngül artım  
- 3 → Artıq çəki  
- 4 → Obez  
- 5 → Çox yüksək obezite  

Stressə təsiri orta səviyyədədir.

---

### 💤 **Sleep Disorder (0–5)**
- 0 → Yoxdur  
- 1–5 → Yüngül → Ağır pozuntu  

Yuxu pozuntusu olduqda model stressi artırır.

---

## 📌 Vacib Qeyd
Model ən çox aşağıdakı 6 parametrdən təsirlənir:

**Sleep Duration, Quality of Sleep, Heart Rate, Blood Pressure, Physical Activity, Text Emotion**

Qalan dəyişənlərin təsiri zəifdir və əsasən dəstəkləyici rol oynayır.

---
""")


# =========================================================
# INPUT AREA
# =========================================================

def input_block():
    # Yalnız əsas 6 parametr

    sleep = st.number_input(
        "😴 Yuxu müddəti (saat)", 
        min_value=0.0, max_value=12.0, value=7.0, step=0.1
    )

    quality = st.slider(
        "🌙 Yuxu keyfiyyəti (1–10)", 
        min_value=1, max_value=10, value=7
    )

    hr = st.number_input(
        "💓 Ürək döyüntüsü (BPM)", 
        min_value=40, max_value=130, value=75
    )

    activity = st.slider(
        "💪 Fiziki aktivlik (1–10)", 
        min_value=1, max_value=10, value=5
    )

    sbp = st.number_input(
        "🩸 Sistolik təzyiq", 
        min_value=80, max_value=200, value=120
    )

    dbp = st.number_input(
        "🩸 Diastolik təzyiq", 
        min_value=40, max_value=130, value=80
    )

    text = st.text_area(
        "✍️ Emosional təsvir", 
        "Bu gün özümü yaxşı hiss edirəm."
    )

    # Numeric values: modelə uyğun olaraq 6 dəyəri qaytarırıq
    numeric = np.array([sleep, quality, hr, activity, sbp, dbp], dtype=float)

    return numeric, text, sleep

if mode == "Preset":
    preset = PRESETS[preset_name]
    numeric_vals = np.array([
        preset["sleep"],
        preset["quality"],
        preset["hr"],
        preset["activity"],
        preset["sbp"],
        preset["dbp"]
    ], dtype=float)
    text_val = preset["text"]
    sleep_val = preset["sleep"]
else:
    numeric_vals, text_val, sleep_val = input_block()


# =========================================================
# PREDICT BUTTON
# =========================================================
if st.button("🔮 Proqnoz Et"):
    pred = fusion_predict(text_val, numeric_vals, sleep_val)

    st.subheader("🔍 Nəticə")
    if pred < 0.40:
        st.success(f"**Aşağı Risk** — Stress göstəricisi: {pred:.3f}")
    elif pred < 0.70:
        st.warning(f"**Orta Risk** — Stress göstəricisi: {pred:.3f}")
    else:
        st.error(f"**Yüksək Risk** — Stress göstəricisi: {pred:.3f}")

    st.markdown("---")


else:
    st.info("Proqnoz üçün ssenari seçin və ya dəyərləri daxil edin.")

# =========================================================
# 📊 QRAFİK ANALİTİKA — EXPANDER VERSİYASI 
# =========================================================

# st.markdown("---")
st.subheader("📊 Analitik Qrafiklər")

with st.expander("📌 Qrafikləri göstər (açmaq üçün klikləyin)"):
    st.write("Aşağıdakı qrafiklər modelin işləmə prinsiplərini və nəticələrini nümayiş etdirir:")

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/fig4_shap_clean.png", caption="SHAP təsir gücü", use_column_width=True)
    with col2:
        st.image("images/fig1_prediction_vs_actual.png", caption="Prediction vs Actual", use_column_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.image("images/fig3_pca.png", caption="BERT PCA Analizi", use_column_width=True)
    with col4:
        st.image("images/fig2_model_comparison.png", caption="Model müqayisələri", use_column_width=True)

    st.image("images/fusion_architecture.png", caption="Fusion Model Architecture", use_column_width=True)

