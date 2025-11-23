import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel


# =========================================================
#                  STREAMLIT PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Stress və Psixoloji Sağlamlıq Proqnozu",
    layout="wide"
)

st.title("🧠 Stress və Psixoloji Sağlamlıq Proqnoz Sistemi")
st.write("Bu sistem yuxu, həyat tərzi və emosional məlumatlar əsasında stress səviyyəsini proqnozlaşdırır.")
st.markdown("---")


# =========================================================
#             MODEL ARCHITECTURES
# =========================================================

class FusionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)


class MLPProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 128)
    def forward(self, x):
        return self.proj(x)


# =========================================================
#            LOAD BERT (cached)
# =========================================================

@st.cache_resource
def load_bert():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    return tokenizer, model

tokenizer, bert_model = load_bert()


# =========================================================
#         LOAD TRAINED MODELS + SCALER PARAMS
# =========================================================

# 908 = 768 BERT + 128 MLP + 12 numeric
fusion_model = FusionModel(908)
fusion_model.load_state_dict(torch.load("models/fusion_model.pth", map_location="cpu"))
fusion_model.eval()

mlp_projection = MLPProjection()
mlp_projection.load_state_dict(torch.load("models/mlp_projection.pth", map_location="cpu"))
mlp_projection.eval()

scaler_mean = np.load("models/scaler_mean.npy")
scaler_std = np.load("models/scaler_std.npy")

# These come from training (y_min, y_max)
Y_MIN =  df_min =  df_min = 0.0
Y_MAX =  df_max =  df_max = 1.0   # normalized target (0–1)


# =========================================================
#                 HELPER FUNCTIONS
# =========================================================

def scale_numeric(x):
    """Apply StandardScaler normalization."""
    return (x - scaler_mean) / scaler_std


def get_bert_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=128)
    with torch.no_grad():
        out = bert_model(**tokens)
    return out.last_hidden_state[:, 0, :].numpy()[0]  # (768,)


def fusion_predict(text, sleep_duration, numeric_vals):
    # BERT
    bert_emb = get_bert_embedding(text)

    # MLP projection (sleep duration)
    sleep_tensor = torch.tensor([[sleep_duration]], dtype=torch.float32)
    mlp_emb = mlp_projection(sleep_tensor).detach().numpy()[0]  # (128,)

    # Scale numeric
    numeric_scaled = scale_numeric(numeric_vals)

    # Concatenate
    inp = np.concatenate([bert_emb, mlp_emb, numeric_scaled], axis=0)
    inp_t = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_norm = fusion_model(inp_t).item()  # normalized 0–1

    # Clamp
    pred_norm = max(0.0, min(1.0, pred_norm))

    return pred_norm


def risk_level(pred_norm):
    if pred_norm < 0.40:
        return "Aşağı", "green"
    elif pred_norm < 0.65:
        return "Orta", "orange"
    else:
        return "Yüksək", "red"


# =========================================================
#                 PRESET DEFINITIONS
# =========================================================

def get_preset(name):
    presets = {

        "Aşağı Stress": {
            "gender": "Kişi", "age": 25, "occupation": 3, "sleep": 8,
            "quality": 8, "activity": 7, "bmi": 1, "hr": 70,
            "steps": 8000, "disorder": 0, "sbp": 110, "dbp": 70,
            "text": "Bu gün özümü çox yaxşı hiss edirəm, sakit və enerjiliyəm."
        },

        "Orta Stress": {
            "gender": "Qadın", "age": 32, "occupation": 5, "sleep": 6,
            "quality": 5, "activity": 4, "bmi": 2, "hr": 82,
            "steps": 4500, "disorder": 1, "sbp": 125, "dbp": 80,
            "text": "Gün normal keçdi, amma bir az yorğunluq var."
        },

        "Yüksək Stress": {
            "gender": "Qadın", "age": 40, "occupation": 7, "sleep": 4,
            "quality": 3, "activity": 2, "bmi": 3, "hr": 95,
            "steps": 2000, "disorder": 1, "sbp": 145, "dbp": 95,
            "text": "Son günlər çox narahatam, gecələr yuxuya gedə bilmirəm."
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
            "text": "Özümü formada hiss edirəm, məşqlər çox yaxşı gedir."
        }
    }
    return presets.get(name, None)



# =========================================================
#                        SIDEBAR UI
# =========================================================

st.sidebar.header("📝 Məlumatları daxil edin")

preset_name = st.sidebar.selectbox(
    "📌 Hazır ssenari seç:",
    ["— Manual —", "Aşağı Stress", "Orta Stress", "Yüksək Stress",
     "İmtahan stresli tələbə", "İdmançı"]
)

preset = get_preset(preset_name)


# ============================
#  MANUAL MODE (inputs shown)
# ============================
if preset is None:

    gender = st.sidebar.selectbox("Cins", ["Kişi", "Qadın"])
    age = st.sidebar.number_input("Yaş", 10, 100, 25)
    occupation = st.sidebar.number_input("Peşə kodu", 0, 20, 5)
    sleep_duration = st.sidebar.slider("Yuxu müddəti (saat)", 0.0, 12.0, 7.0)
    quality_sleep = st.sidebar.slider("Yuxu keyfiyyəti", 1, 10, 7)
    activity = st.sidebar.slider("Fiziki aktivlik", 1, 10, 5)
    bmi = st.sidebar.number_input("BMI kodu", 0, 5, 2)
    hr = st.sidebar.number_input("Ürək döyüntüsü", 40, 130, 80)
    steps = st.sidebar.number_input("Günlük addım sayı", 0, 30000, 5000)
    disorder = st.sidebar.number_input("Yuxu pozuntusu", 0, 5, 0)
    sbp = st.sidebar.number_input("Sistolik təzyiq", 80, 200, 120)
    dbp = st.sidebar.number_input("Diastolik təzyiq", 40, 130, 80)
    user_text = st.sidebar.text_area("Əhval haqqında qısa təsvir:")

else:
    # PRESET MODE — hide inputs
    gender = preset["gender"]
    age = preset["age"]
    occupation = preset["occupation"]
    sleep_duration = preset["sleep"]
    quality_sleep = preset["quality"]
    activity = preset["activity"]
    bmi = preset["bmi"]
    hr = preset["hr"]
    steps = preset["steps"]
    disorder = preset["disorder"]
    sbp = preset["sbp"]
    dbp = preset["dbp"]
    user_text = preset["text"]

    st.sidebar.success(f"Preset seçildi: **{preset_name}**")
    st.sidebar.markdown("Manual inputlar gizlədildi.")


# =========================================================
#                     RUN PREDICTION
# =========================================================

if st.sidebar.button("🔮 Proqnoz Et"):

    gender_val = 1 if gender == "Qadın" else 0

    numeric_vals = np.array([
        gender_val, age, occupation, sleep_duration, quality_sleep,
        activity, bmi, hr, steps, disorder, sbp, dbp
    ], dtype=float)

    pred_norm = fusion_predict(user_text, sleep_duration, numeric_vals)
    risk, color = risk_level(pred_norm)

    st.subheader("🔍 Proqnoz nəticəsi")

    st.markdown(f"""
        <div style='padding:15px; background-color:{color}; color:white; border-radius:10px;'>
            <h2>{risk} risk səviyyəsi</h2>
            <p>Normallaşdırılmış stress göstəricisi: <b>{pred_norm:.3f}</b></p>
        </div>
    """, unsafe_allow_html=True)

    # =========================================================
    #               SHOW PLOTS (checkbox)
    # =========================================================

    st.markdown("---")
    st.subheader("📊 Analitik qrafiklər")
    
    show_charts = st.checkbox("Qrafikləri göstər", value=False)
    
    if show_charts:
        try:
            col1, col2 = st.columns(2)
            with col1:
                st.image("images/fig4_shap_clean.png", caption="SHAP faktor təsirləri")
            with col2:
                st.image("images/fig1_prediction_vs_actual.png", caption="Prediction vs Actual")
    
            col3, col4 = st.columns(2)
            with col3:
                st.image("images/fig3_pca.png", caption="PCA — mətn analizi")
            with col4:
                st.image("images/fig2_model_comparison.png", caption="Model müqayisəsi")
    
            st.image("images/fusion_architecture.png", caption="Fusion Model Arxitekturası")
    
        except Exception as e:
            st.error(f"Qrafikləri göstərmək mümkün olmadı: {e}")


else:
    st.info("Proqnoz üçün tələb olunan məlumatları daxil edin.")
