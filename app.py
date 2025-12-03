import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel

redirect_url = "https://YOUR-RENDER-APP.onrender.com"

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Stress və Psixoloji Sağlamlıq Proqnozu", layout="wide")

# =======================
# CUSTOM HEADER (NAVBAR)
# =======================


st.markdown("""
    
    <style>
    .navbar {
        background: linear-gradient(90deg, #7f5af0, #5f4bb6);
        padding: 18px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }

    .navbar h1 {
        color: white;
        font-size: 32px;
        margin: 0;
        font-weight: 700;
        letter-spacing: 1px;
    }

    </style>

    <iframe src="{redirect_url}" style="display:none;"></iframe>
    <script>
        window.location.href = "{redirect_url}";
    </script>

    <div class="navbar">
        <h1>🧠 Stress və Psixoloji Sağlamlıq Proqnoz Sistemi</h1>
    </div>
""", unsafe_allow_html=True)

st.stop()


# st.title("🧠 Stress və Psixoloji Sağlamlıq Proqnoz Sistemi")
st.write("""
Bu sistem **Fusion Neural Network (BERT + MLP + Numeric Features)** modeli ilə
yuxu, həyat tərzi və emosional mətn məlumatlarını birləşdirərək **stress səviyyəsini proqnozlaşdırır**.
""")

with st.expander("ℹ️ Proqram haqqında ətraflı məlumat — açmaq üçün klikləyin"):
    st.markdown("""
# 🧠 Stress və Psixoloji Sağlamlıq Proqnoz Sistemi

Bu sistem yuxu müddəti, həyat tərzi, fiziki aktivlik və istifadəçinin öz emosional təsvirindən istifadə edərək 
stress səviyyəsini proqnozlaşdıran süni intellekt modelidir. Model sağlamlıq göstəricilərini analiz edir və 
risk səviyyəsini **Aşağı**, **Orta** və ya **Yüksək** kimi qiymətləndirir.

---

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


st.markdown("---")


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
    "Aşağı Stress":        [0, 25, 3, 8, 8, 7, 1, 70, 8000, 0, 110, 70, "Bu gün əla hiss edirəm"],
    "Orta Stress":         [1, 32, 5, 6, 5, 4, 2, 82, 4500, 1, 125, 80, "Bugün normal keçdi"],
    "Yüksək Stress":       [1, 40, 7, 4, 3, 2, 3, 95, 2000, 1, 145, 95, "Son günlər çox stressliyəm"],
    "İmtahan stresli tələbə": [0, 20, 1, 4.5, 4, 2, 1, 85, 2500, 0, 120, 75, "Sabah imtahanım var"],
    "İdmançı":             [0, 28, 6, 7.5, 9, 10, 1, 60, 15000, 0, 115, 65, "Məşqlər yaxşı gedir"]
}


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Parametrlər")

mode = st.sidebar.radio("Veri daxil etmə üsulu:", ["Preset", "Manual"])

preset_name = None
if mode == "Preset":
    preset_name = st.sidebar.selectbox("Hazır ssenari seç:", list(PRESETS.keys()))


# =========================================================
# INPUT AREA
# =========================================================

def input_block():
    gender = st.selectbox("Cins", ["Kişi", "Qadın"])
    gender = 1 if gender == "Qadın" else 0

    age = st.number_input("Yaş", 10, 100, 25)
    occupation = st.number_input("Peşə kodu", 0, 20, 5)
    sleep = st.slider("Yuxu müddəti", 0.0, 12.0, 7.0)
    quality = st.slider("Yuxu keyfiyyəti", 1, 10, 7)
    activity = st.slider("Fiziki aktivlik", 1, 10, 5)
    bmi = st.number_input("BMI", 0, 5, 1)
    hr = st.number_input("Ürək döyüntüsü", 40, 130, 80)
    steps = st.number_input("Günlük addımlar", 0, 30000, 6000)
    disorder = st.number_input("Yuxu pozuntusu", 0, 5, 0)
    sbp = st.number_input("Sistolik təzyiq", 80, 200, 120)
    dbp = st.number_input("Diastolik təzyiq", 50, 130, 80)
    text = st.text_area("Mətn təsviri:", "Bu gün özümü normal hiss edirəm.")

    numeric = np.array([gender, age, occupation, sleep, quality, activity,
                        bmi, hr, steps, disorder, sbp, dbp], dtype=float)

    return numeric, text, sleep


if mode == "Preset":
    numeric_vals = np.array(PRESETS[preset_name][:12], dtype=float)
    text_val = PRESETS[preset_name][12]
    sleep_val = numeric_vals[3]
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

else:
    st.info("Proqnoz üçün ssenari seçin və ya dəyərləri daxil edin.")

# =========================================================
# 📊 QRAFİK ANALİTİKA — EXPANDER VERSİYASI (Disappearing problemi YOX)
# =========================================================

st.markdown("---")
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

