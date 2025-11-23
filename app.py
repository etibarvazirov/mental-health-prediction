import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertModel
import matplotlib.pyplot as plt
from PIL import Image


# =========================================================
#                 MODEL ARXITEKTURALARI
# =========================================================

class FusionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)


class MLPProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 128)  # MLP → 128 dim
    def forward(self, x):
        return self.proj(x)


# =========================================================
#                 YÜKLƏNMƏ BÖLMƏSİ
# =========================================================

@st.cache_resource
def load_bert():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    return tokenizer, model

tokenizer, bert_model = load_bert()

# MLP Projection Layer
proj_layer = MLPProjection()


# =========================================================
#                 FUSION MODEL YARADILMASI
# =========================================================

def build_fusion_model(n_numeric):
    input_dim = 768 + 128 + n_numeric  # BERT + MLP + numeric
    model = FusionModel(input_dim)
    return model

fusion_model = None  # initialize later


# =========================================================
#                 BERT EMBEDDING FUNKSIYASI
# =========================================================

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()[0]


# =========================================================
#                 APP DIZAYNI BAŞLAYIR
# =========================================================

st.set_page_config(page_title="Stress və Psixoloji Sağlamlıq Proqnozu",
                   layout="wide")

st.title("🧠 Stress və Psixoloji Sağlamlıq Proqnoz Sistemi")
st.write("Bu sistem yuxu, həyat tərzi və emosional məlumatlar əsasında stress səviyyəsini proqnozlaşdırır.")

st.markdown("---")


# =========================================================
#                 SIDEBAR — INPUT FORM
# =========================================================

st.sidebar.header("📝 Məlumatları daxil edin")

gender = st.sidebar.selectbox("Cins", ["Kişi", "Qadın"])
age = st.sidebar.number_input("Yaş", 10, 100, 25)
occupation = st.sidebar.number_input("Peşə (kodu)", 0, 20, 5)
sleep_duration = st.sidebar.slider("Yuxu müddəti (saat)", 0.0, 12.0, 7.0)
quality_sleep = st.sidebar.slider("Yuxu keyfiyyəti (1–10)", 1, 10, 7)
activity = st.sidebar.slider("Fiziki Aktivlik (1–10)", 1, 10, 5)
bmi = st.sidebar.number_input("BMI Kateqoriyası (kodu)", 0, 5, 2)
heartrate = st.sidebar.number_input("Ürək döyüntüsü", 40, 130, 80)
steps = st.sidebar.number_input("Günlük addım sayı", 0, 30000, 5000)
disorder = st.sidebar.number_input("Yuxu pozuntusu (kodu)", 0, 5, 0)
sbp = st.sidebar.number_input("Sistolik təzyiq", 80, 200, 120)
dbp = st.sidebar.number_input("Diastolik təzyiq", 40, 130, 80)

user_text = st.sidebar.text_area("Günlük əhval və stress barədə qısa təsvir yazın:",
                                 "Bu gün özümü bir az yorğun hiss edirəm...")

if st.sidebar.button("🔮 Proqnoz Et"):
    # =========================================================
    # NUMERIC FEATURES
    # =========================================================
    numeric = np.array([
        1 if gender == "Qadın" else 0,
        age,
        occupation,
        sleep_duration,
        quality_sleep,
        activity,
        bmi,
        heartrate,
        steps,
        disorder,
        sbp,
        dbp
    ], dtype=float)

    numeric_tensor = torch.tensor(numeric, dtype=torch.float32).unsqueeze(0)

    # =========================================================
    # MLP EMBEDDINGS
    # =========================================================
    mlp_emb = proj_layer(torch.tensor([[sleep_duration]], dtype=torch.float32)).detach().numpy()[0]

    # =========================================================
    # BERT EMBEDDINGS
    # =========================================================
    bert_emb = get_bert_embedding(user_text)

    # =========================================================
    # CONCAT ALL
    # =========================================================
    fusion_input = np.concatenate([bert_emb, mlp_emb, numeric])
    fusion_input_tensor = torch.tensor(fusion_input, dtype=torch.float32).unsqueeze(0)

    # Fusion model qurulur (input dim auto)
    if fusion_model is None:
        fusion_model = build_fusion_model(len(numeric))

    fusion_model.eval()
    with torch.no_grad():
        pred = fusion_model(fusion_input_tensor).item()

    # =========================================================
    # RISK LEVEL
    # =========================================================
    if pred < 0.33:
        risk = "Aşağı"
        color = "green"
    elif pred < 0.66:
        risk = "Orta"
        color = "orange"
    else:
        risk = "Yüksək"
        color = "red"

    # =========================================================
    # RESULT PANEL
    # =========================================================
    st.subheader("🔍 Proqnoz nəticəsi")
    st.markdown(f"""
    <div style='padding:15px; background-color:{color}; color:white; border-radius:10px;'>
        <h2>{risk} risk səviyyəsi</h2>
        <p>Stress göstəricisi: <b>{pred:.3f}</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================================================
    # DASHBOARD QRAFIKLARI
    # =========================================================

    st.subheader("📊 Qrafik Analitika")

    col1, col2 = st.columns(2)

    with col1:
        st.image("fig4_shap_clean.png", caption="SHAP — Faktorların təsir gücü")

    with col2:
        st.image("fig1_prediction_vs_actual.png", caption="Fusion Model — Prediction vs Actual")

    col3, col4 = st.columns(2)

    with col3:
        st.image("fig3_pca.png", caption="BERT PCA — Emosional mətn analizi")

    with col4:
        st.image("fig2_model_comparison.png", caption="Modellərin ümumi müqayisəsi")

    st.image("fusion_architecture.png", caption="Fusion Model Arxitekturası")

else:
    st.info("Proqnoz üçün məlumatları daxil edin və 'Proqnoz Et' düyməsinə basın.")
