import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Stress və Psixoloji Sağlamlıq Proqnozu",
    layout="wide"
)

st.title("🧠 Stress və Psixoloji Sağlamlıq Proqnoz Sistemi")
st.write("Bu sistem yuxu, həyat tərzi və emosional məlumatlar əsasında stress səviyyəsini proqnozlaşdırır.")
st.markdown("---")


# ============================================================
# LOAD SCALER
# ============================================================
scaler_mean = np.load("models/scaler_mean.npy")
scaler_std = np.load("models/scaler_std.npy")

def scale_numeric(x):
    return (x - scaler_mean) / scaler_std


# ============================================================
# LOAD MLP PROJECTION
# ============================================================
class MLPProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 128)

    def forward(self, x):
        return self.proj(x)

proj_layer = MLPProjection()
proj_layer.load_state_dict(torch.load("models/mlp_projection.pth", map_location="cpu"))
proj_layer.eval()


# ============================================================
# LOAD FUSION MODEL
# ============================================================
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

fusion_model = FusionModel(908)
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
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        out = bert_model(**enc)
    return out.last_hidden_state[:, 0, :].numpy()[0]


# ============================================================
# STABIL, REAL DATA PRESETLƏR
# ============================================================
def get_preset(name):
    presets = {

        "Aşağı Stress": {
            "gender": "Kişi", "age": 28, "occupation": 2, "sleep": 8,
            "quality": 8, "activity": 6, "bmi": 1, "hr": 70,
            "steps": 9000, "disorder": 0, "sbp": 115, "dbp": 75,
            "text": "Bu gün özümü sakit, enerjili və pozitiv hiss edirəm."
        },

        "Orta Stress": {
            "gender": "Qadın", "age": 35, "occupation": 4, "sleep": 6.5,
            "quality": 6, "activity": 4, "bmi": 1, "hr": 82,
            "steps": 6000, "disorder": 0, "sbp": 125, "dbp": 82,
            "text": "Bugün normal idi, amma bir az yorğun hiss edirəm."
        },

        "Yüksək Stress": {
            "gender": "Qadın", "age": 42, "occupation": 7, "sleep": 4.5,
            "quality": 4, "activity": 2, "bmi": 2, "hr": 95,
            "steps": 3500, "disorder": 1, "sbp": 140, "dbp": 90,
            "text": "Son günlərdən bəri çox stress hiss edirəm, yuxum pozulub."
        },

        "Yuxusuzluq stressi": {
            "gender": "Kişi", "age": 31, "occupation": 3, "sleep": 3.5,
            "quality": 3, "activity": 3, "bmi": 1, "hr": 90,
            "steps": 4500, "disorder": 1, "sbp": 130, "dbp": 85,
            "text": "Bu həftə düzgün yata bilmədim. Hiss olunur ki, stres artıb."
        },

        "İş gərginliyi": {
            "gender": "Qadın", "age": 38, "occupation": 6, "sleep": 5.5,
            "quality": 5, "activity": 3, "bmi": 1, "hr": 88,
            "steps": 5000, "disorder": 0, "sbp": 135, "dbp": 88,
            "text": "İş çox gərgin idi. Daimi təzyiq hiss edirəm."
        },

        "İmtahan stresli tələbə": {
            "gender": "Kişi", "age": 21, "occupation": 1, "sleep": 5,
            "quality": 5, "activity": 2, "bmi": 1, "hr": 85,
            "steps": 3000, "disorder": 0, "sbp": 120, "dbp": 78,
            "text": "Sabah imtahanım var. Narahatlıq hissi var."
        },

        "İdmançı": {
            "gender": "Kişi", "age": 26, "occupation": 5, "sleep": 7.5,
            "quality": 9, "activity": 7, "bmi": 1, "hr": 62,
            "steps": 14000, "disorder": 0, "sbp": 112, "dbp": 68,
            "text": "Formam yerindədir, məşq əla keçdi."
        },

        "Sağlam həyat tərzi": {
            "gender": "Qadın", "age": 30, "occupation": 4, "sleep": 8.5,
            "quality": 9, "activity": 7, "bmi": 1, "hr": 68,
            "steps": 12000, "disorder": 0, "sbp": 110, "dbp": 70,
            "text": "Günü sakit, sağlam və balanslı keçirdim."
        }
    }

    return presets.get(name)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📝 Məlumatları daxil edin")

preset_name = st.sidebar.selectbox(
    "Hazır ssenari seç",
    ["— Manual —", "Aşağı Stress", "Orta Stress", "Yüksək Stress",
     "Yuxusuzluq stressi", "İş gərginliyi", "İmtahan stresli tələbə",
     "İdmançı", "Sağlam həyat tərzi"]
)

p = get_preset(preset_name)

manual_mode = (preset_name == "— Manual —")


# ============================================================
# INPUT UI (preset → inputlar gizlənir)
# ============================================================

if manual_mode:
    gender = st.sidebar.selectbox("Cins", ["Kişi", "Qadın"])
    age = st.sidebar.number_input("Yaş", 10, 100, 25)
    occupation = st.sidebar.number_input("Peşə (LabelEncoder kodu)", 0, 9, 3)
    sleep = st.sidebar.slider("Yuxu müddəti", 3.0, 9.0, 7.0)
    quality = st.sidebar.slider("Yuxu keyfiyyəti", 3, 9, 7)
    activity = st.sidebar.slider("Fiziki Aktivlik", 1, 7, 4)
    bmi = st.sidebar.number_input("BMI kodu", 0, 2, 1)
    hr = st.sidebar.number_input("Ürək döyüntüsü", 60, 100, 80)
    steps = st.sidebar.number_input("Günlük addım", 3000, 15000, 6000)
    disorder = st.sidebar.number_input("Yuxu pozuntusu (0/1)", 0, 1, 0)
    sbp = st.sidebar.number_input("Sistolik təzyiq", 105, 145, 120)
    dbp = st.sidebar.number_input("Diastolik təzyiq", 65, 95, 80)
    text = st.sidebar.text_area("Mətn:", "")
else:
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

    st.sidebar.success("Preset tətbiq olundu ✓")


# ============================================================
# PREDICT
# ============================================================

if st.sidebar.button("🔮 Proqnoz Et"):

    numeric = np.array([
        1 if gender == "Qadın" else 0,
        age, occupation, sleep, quality,
        activity, bmi, hr, steps,
        disorder, sbp, dbp
    ], dtype=float)

    numeric_scaled = scale_numeric(numeric)

    mlp_emb = proj_layer(torch.tensor([[sleep]], dtype=torch.float32)).detach().numpy()[0]
    bert_emb = get_bert_embedding(text)

    fusion_in = np.concatenate([bert_emb, mlp_emb, numeric_scaled])
    fusion_tensor = torch.tensor(fusion_in, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = fusion_model(fusion_tensor).item()

    if pred < 3:
        risk = "Aşağı"; color = "green"
    elif pred < 6:
        risk = "Orta"; color = "orange"
    else:
        risk = "Yüksək"; color = "red"

    st.subheader("🔍 Proqnoz nəticəsi")
    st.markdown(f"""
    <div style='padding:15px; background-color:{color}; color:white; border-radius:10px;'>
        <h2>{risk} risk səviyyəsi</h2>
        <p>Stress göstəricisi: <b>{pred:.2f}</b></p>
    </div>
    """, unsafe_allow_html=True)


    # ============================================================
    # CHARTS
    # ============================================================
    st.markdown("---")

    if st.checkbox("📊 Qrafikləri göstər"):
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
