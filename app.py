import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel


# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="Stress və Psixoloji Sağlamlıq Proqnozu",
    layout="wide"
)

st.title("🧠 Stress və Psixoloji Sağlamlıq Proqnoz Sistemi")
st.write("Bu sistem yuxu, həyat tərzi və emosional məlumatlar əsasında stress səviyyəsini proqnozlaşdırır.")
st.markdown("---")


# =========================================================
# LOAD SCALER
# =========================================================
scaler_mean = np.load("models/scaler_mean.npy")
scaler_std = np.load("models/scaler_std.npy")

def scale_numeric(x):
    return (x - scaler_mean) / scaler_std


# =========================================================
# FUSION MODEL 908-DIM
# =========================================================
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


# Load trained model
fusion_model = FusionModel(908)
fusion_model.load_state_dict(torch.load("models/fusion_model.pth", map_location="cpu"))
fusion_model.eval()


# =========================================================
# LOAD BERT
# =========================================================
@st.cache_resource
def load_bert():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    return tokenizer, model

tokenizer, bert_model = load_bert()


def get_bert_embedding(text):
    encoded = tokenizer(text, return_tensors="pt",
                        truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        out = bert_model(**encoded)
    return out.last_hidden_state[:, 0, :].numpy()[0]


# =========================================================
# MLP PROJECTION (Sleep Duration → 128 dim)
# =========================================================
class MLPProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 128)

    def forward(self, x):
        return self.proj(x)


proj_layer = MLPProjection()


# =========================================================
# PRESETS
# =========================================================
def get_preset(name):

    presets = {

        "Yuxusuzluq stressi": {
            "gender": "Kişi", "age": 29, "occupation": 4, "sleep": 3,
            "quality": 2, "activity": 3, "bmi": 2, "hr": 100,
            "steps": 3500, "disorder": 1, "sbp": 130, "dbp": 85,
            "text": "Bu həftə yaxşı yata bilmədim, başım ağrıyır."
        },

        "İş gərginliyi": {
            "gender": "Qadın", "age": 36, "occupation": 9, "sleep": 5,
            "quality": 4, "activity": 3, "bmi": 2, "hr": 90,
            "steps": 3000, "disorder": 0, "sbp": 140, "dbp": 88,
            "text": "İşdə çox gərgin gün keçirdim, narahatam."
        },

        "İmtahan stresli tələbə": {
            "gender": "Kişi", "age": 20, "occupation": 1, "sleep": 4.5,
            "quality": 4, "activity": 2, "bmi": 1, "hr": 85,
            "steps": 2500, "disorder": 0, "sbp": 120, "dbp": 75,
            "text": "Sabah imtahanım var, çox stressliyəm."
        },

        "İdmançı": {
            "gender": "Kişi", "age": 28, "occupation": 6, "sleep": 7.5,
            "quality": 9, "activity": 10, "bmi": 1, "hr": 60,
            "steps": 15000, "disorder": 0, "sbp": 115, "dbp": 65,
            "text": "Özümü formada hiss edirəm, məşqlər yaxşı gedir."
        },

        "Sağlam həyat tərzi": {
            "gender": "Qadın", "age": 30, "occupation": 5, "sleep": 8,
            "quality": 9, "activity": 8, "bmi": 1, "hr": 68,
            "steps": 12000, "disorder": 0, "sbp": 110, "dbp": 70,
            "text": "Günüm sakit və enerjili keçdi, yaxşı hiss edirəm."
        },
    }

    return presets.get(name, None)


# =========================================================
# SIDEBAR INPUTS
# =========================================================
preset_name = st.sidebar.selectbox(
    "📌 Hazır ssenarilər",
    ["— Manual —",
     "Yuxusuzluq stressi",
     "İş gərginliyi",
     "İmtahan stresli tələbə",
     "İdmançı",
     "Sağlam həyat tərzi"]
)

preset = get_preset(preset_name)

gender = preset["gender"] if preset else st.sidebar.selectbox("Cins", ["Kişi", "Qadın"])
age = preset["age"] if preset else st.sidebar.number_input("Yaş", 10, 100, 25)
occupation = preset["occupation"] if preset else st.sidebar.number_input("Peşə", 0, 20, 5)
sleep_duration = preset["sleep"] if preset else st.sidebar.slider("Yuxu müddəti", 0.0, 12.0, 7.0)
quality = preset["quality"] if preset else st.sidebar.slider("Yuxu keyfiyyəti", 1, 10, 7)
activity = preset["activity"] if preset else st.sidebar.slider("Fiziki Aktivlik", 1, 10, 5)
bmi = preset["bmi"] if preset else st.sidebar.number_input("BMI", 0, 5, 2)
hr = preset["hr"] if preset else st.sidebar.number_input("Ürək döyüntüsü", 40, 130, 80)
steps = preset["steps"] if preset else st.sidebar.number_input("Addım sayı", 0, 30000, 5000)
disorder = preset["disorder"] if preset else st.sidebar.number_input("Yuxu pozuntusu", 0, 5, 0)
sbp = preset["sbp"] if preset else st.sidebar.number_input("Sistolik", 80, 200, 120)
dbp = preset["dbp"] if preset else st.sidebar.number_input("Diastolik", 40, 130, 80)

user_text = preset["text"] if preset else st.sidebar.text_area("Mətn:", "Bu gün özümü yorğun hiss edirəm.")


# =========================================================
# PREDICTION
# =========================================================
if st.sidebar.button("🔮 Proqnoz Et"):

    numeric_raw = np.array([
        1 if gender == "Qadın" else 0,
        age, occupation, sleep_duration, quality,
        activity, bmi, hr, steps,
        disorder, sbp, dbp
    ], dtype=float)

    numeric_scaled = scale_numeric(numeric_raw)

    # MLP projection for sleep duration → 128 dim
    mlp_emb = proj_layer(torch.tensor([[sleep_duration]], dtype=torch.float32)).detach().numpy()[0]

    bert_emb = get_bert_embedding(user_text)

    # CONCAT 768 + 128 + 12 = 908
    fusion_input = np.concatenate([bert_emb, mlp_emb, numeric_scaled])

    x = torch.tensor(fusion_input, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = fusion_model(x).item()

    # pred is in 0–1 → convert to 1–10
    stress_score = 1 + pred * 9

    if pred < 0.33:
        risk = "Aşağı"; color = "green"
    elif pred < 0.66:
        risk = "Orta"; color = "orange"
    else:
        risk = "Yüksək"; color = "red"


    st.markdown(f"""
    <div style='padding:15px; background-color:{color}; color:white; border-radius:10px;'>
        <h2>{risk} risk səviyyəsi</h2>
        <p>Stress göstəricisi: <b>{stress_score:.2f}</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================================================
    # OPTIONAL DIAGRAMS (Checkbox)
    # =========================================================
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
    st.info("Məlumatları daxil edib düyməyə basın.")
