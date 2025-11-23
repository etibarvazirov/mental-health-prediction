import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Stress və Psixoloji Sağlamlıq Proqnozu",
    layout="wide"
)

st.title("🧠 Stress və Psixoloji Sağlamlıq Proqnoz Sistemi")
st.write("Bu sistem yuxu, həyat tərzi və emosional məlumatlar əsasında stress səviyyəsini proqnozlaşdırır.")
st.markdown("---")


# ============================================================
# LOAD MODELS (Fusion, MLP, Scaler)
# ============================================================

# Load scaler
scaler_mean = np.load("models/scaler_mean.npy")
scaler_std = np.load("models/scaler_std.npy")

def scale_numeric(x):
    return (x - scaler_mean) / scaler_std


# MLP projection layer
class MLPProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 128)

    def forward(self, x):
        return self.proj(x)

proj_layer = MLPProjection()
proj_layer.load_state_dict(torch.load("models/mlp_projection.pth", map_location="cpu"))
proj_layer.eval()


# Fusion Model
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

fusion_input_dim = 908
fusion_model = FusionModel(fusion_input_dim)
fusion_model.load_state_dict(torch.load("models/fusion_model.pth", map_location="cpu"))
fusion_model.eval()


# ============================================================
# LOAD BERT
# ============================================================
@st.cache_resource
def load_bert():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    return tokenizer, model

tokenizer, bert_model = load_bert()


def get_bert_embedding(text):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        out = bert_model(**encoded)
    return out.last_hidden_state[:, 0, :].numpy()[0]


# ============================================================
# PRESETS
# ============================================================

def get_preset(name):
    presets = {
        "Aşağı Stress": {
            "gender": "Kişi", "age": 25, "occupation": 3, "sleep": 8,
            "quality": 8, "activity": 7, "bmi": 1, "hr": 70,
            "steps": 8000, "disorder": 0, "sbp": 110, "dbp": 70,
            "text": "Bu gün özümü çox yaxşı hiss edirəm."
        },

        "Orta Stress": {
            "gender": "Qadın", "age": 32, "occupation": 5, "sleep": 6,
            "quality": 5, "activity": 4, "bmi": 2, "hr": 82,
            "steps": 4500, "disorder": 1, "sbp": 125, "dbp": 80,
            "text": "Bugün normal keçdi, bir az yorğunam."
        },

        "Yüksək Stress": {
            "gender": "Qadın", "age": 40, "occupation": 7, "sleep": 4,
            "quality": 3, "activity": 2, "bmi": 3, "hr": 95,
            "steps": 2000, "disorder": 1, "sbp": 145, "dbp": 95,
            "text": "Çox stress hiss edirəm, yuxularım pozulub."
        },

        "Yuxusuzluq stressi": {
            "gender": "Kişi", "age": 29, "occupation": 4, "sleep": 3,
            "quality": 2, "activity": 3, "bmi": 2, "hr": 100,
            "steps": 3500, "disorder": 1, "sbp": 130, "dbp": 85,
            "text": "Gecələr yata bilmirəm, yorğunam."
        },

        "İş gərginliyi": {
            "gender": "Qadın", "age": 36, "occupation": 9, "sleep": 5,
            "quality": 4, "activity": 3, "bmi": 2, "hr": 90,
            "steps": 3000, "disorder": 0, "sbp": 140, "dbp": 88,
            "text": "İş çox gərgin idi, başım ağrıyır."
        },

        "İmtahan stresli tələbə": {
            "gender": "Kişi", "age": 20, "occupation": 1, "sleep": 4.5,
            "quality": 4, "activity": 2, "bmi": 1, "hr": 85,
            "steps": 2500, "disorder": 0, "sbp": 120, "dbp": 75,
            "text": "Sabah imtahanım var, çox həyəcanlıyam."
        },

        "İdmançı": {
            "gender": "Kişi", "age": 28, "occupation": 6, "sleep": 7.5,
            "quality": 9, "activity": 10, "bmi": 1, "hr": 60,
            "steps": 15000, "disorder": 0, "sbp": 115, "dbp": 65,
            "text": "Enerjiliyəm, məşq yaxşı keçdi."
        },

        "Sağlam həyat tərzi": {
            "gender": "Qadın", "age": 30, "occupation": 5, "sleep": 8,
            "quality": 9, "activity": 8, "bmi": 1, "hr": 68,
            "steps": 12000, "disorder": 0, "sbp": 110, "dbp": 70,
            "text": "Günü sakit və sağlam keçirdim."
        }
    }
    return presets.get(name)


# ============================================================
# SIDEBAR INPUT
# ============================================================

st.sidebar.header("📝 Məlumatları daxil edin")

preset_name = st.sidebar.selectbox(
    "📌 Hazır ssenari seç",
    ["— Manual —", "Aşağı Stress", "Orta Stress", "Yüksək Stress",
     "Yuxusuzluq stressi", "İş gərginliyi", "İmtahan stresli tələbə",
     "İdmançı", "Sağlam həyat tərzi"]
)

p = get_preset(preset_name)


def ui_value(field, default):
    return p[field] if p else default


# ============================================================
# SIDEBAR INPUT (preset seçiləndə inputlar gizlənir)
# ============================================================

manual_mode = (preset_name == "— Manual —")

if manual_mode:
    gender = st.sidebar.selectbox("Cins", ["Kişi", "Qadın"])
    age = st.sidebar.number_input("Yaş", 10, 100, 25)
    occupation = st.sidebar.number_input("Peşə", 0, 20, 5)
    sleep = st.sidebar.slider("Yuxu müddəti", 0.0, 12.0, 7.0)
    quality = st.sidebar.slider("Yuxu keyfiyyəti", 1, 10, 7)
    activity = st.sidebar.slider("Fiziki Aktivlik", 1, 10, 5)
    bmi = st.sidebar.number_input("BMI", 0, 5, 2)
    hr = st.sidebar.number_input("Ürək döyüntüsü", 40, 130, 80)
    steps = st.sidebar.number_input("Günlük addım", 0, 30000, 5000)
    disorder = st.sidebar.number_input("Yuxu pozuntusu", 0, 5, 0)
    sbp = st.sidebar.number_input("Sistolik təzyiq", 80, 200, 120)
    dbp = st.sidebar.number_input("Diastolik təzyiq", 40, 130, 80)
    text = st.sidebar.text_area("Mətn:", "")
else:
    p = get_preset(preset_name)
    gender = p["gender"]
    age = p["age"]
    occupation = p["occupation"]
    sleep = p["sleep"]
    quality = p["quality"]
    activity = p["activity"]
    bmi = p["bmi"]
    hr = p["hr"]
    steps = p["steps"]
    disorder = p["disorder"]
    sbp = p["sbp"]
    dbp = p["dbp"]
    text = p["text"]

    st.sidebar.success("Preset tətbiq olundu ✓\n\nDəyərlər avtomatik dolduruldu.")



# ============================================================
# PREDICT
# ============================================================

if st.sidebar.button("🔮 Proqnoz Et"):

    # numeric vector
    numeric = np.array([
        1 if gender == "Qadın" else 0,
        age, occupation, sleep, quality, activity,
        bmi, hr, steps, disorder, sbp, dbp
    ], dtype=float)

    numeric_scaled = scale_numeric(numeric)

    # MLP sleep duration embedding
    mlp_emb = proj_layer(torch.tensor([[sleep]], dtype=torch.float32)).detach().numpy()[0]

    # BERT embedding
    bert_emb = get_bert_embedding(text)

    # concat all
    fusion_in = np.concatenate([bert_emb, mlp_emb, numeric_scaled])
    fusion_t = torch.tensor(fusion_in, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = fusion_model(fusion_t).item()

    # risk
    if pred < 3:
        risk = "Aşağı"; color = "green"
    elif pred < 6:
        risk = "Orta"; color = "orange"
    else:
        risk = "Yüksək"; color = "red"

    st.subheader("🔍 Nəticə")
    st.markdown(f"""
    <div style='padding:15px; background-color:{color}; color:white; border-radius:10px;'>
        <h2>{risk} risk səviyyəsi</h2>
        <p>Stress göstəricisi: <b>{pred:.2f}</b></p>
    </div>
    """, unsafe_allow_html=True)


    # ============================================================
    # SHOW GRAPHICS (CHECKBOX)
    # ============================================================

    st.markdown("---")
    show_figs = st.checkbox("📊 Qrafikləri göstər", value=False)

    if show_figs:
        col1, col2 = st.columns(2)
        with col1:
            st.image("images/fig4_shap_clean.png")
        with col2:
            st.image("images/fig1_prediction_vs_actual.png")

        col3, col4 = st.columns(2)
        with col3:
            st.image("images/fig3_pca.png")
        with col4:
            st.image("images/fig2_model_comparison.png")

        st.image("images/fusion_architecture.png")


else:
    st.info("Proqnoz üçün məlumatları daxil edin və düyməyə basın.")
