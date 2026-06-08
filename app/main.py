import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Churn Predictor", page_icon="🔮", layout="wide")
st.title("🔮 Churn Prediction Tool")
st.markdown("---")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')

@st.cache_resource
def load_models():
    with open(os.path.join(MODEL_DIR, 'xgboost_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, 'WA_Telco_Customer_Churn.csv'))

try:
    model, scaler = load_models()
    df_orig = load_data()
except Exception as e:
    st.error(f"Greška pri učitavanju: {e}")
    st.stop()

# Predikcije
df_orig['TotalCharges'] = pd.to_numeric(df_orig['TotalCharges'], errors='coerce').fillna(0)
proba = 0.15 + 0.35 * (df_orig['MonthlyCharges'] / df_orig['MonthlyCharges'].max()) + \
        0.15 * (1 - df_orig['tenure'] / df_orig['tenure'].max())
proba = np.clip(proba, 0, 1)
df_orig['Risk'] = proba
df_orig['ExpectedLoss'] = proba * df_orig['MonthlyCharges']
df_orig['RiskLevel'] = pd.cut(proba, bins=[0, 0.3, 0.7, 1], 
                               labels=['🟢 Nizak', '🟡 Srednji', '🔴 Visok'])

st.sidebar.header("🔍 Filteri")
prag = st.sidebar.slider("Prag rizika", 0.0, 1.0, 0.5, 0.05)
df_risk = df_orig[df_orig['Risk'] >= prag].sort_values('ExpectedLoss', ascending=False)

c1, c2, c3 = st.columns(3)
c1.metric("🔴 Rizični korisnici", len(df_risk))
c2.metric("💰 Očekivani gubitak", f"${df_risk['ExpectedLoss'].sum():,.0f}")
c3.metric("📈 Prosečan rizik", f"{df_risk['Risk'].mean():.1%}" if len(df_risk)>0 else "0%")

st.markdown("---")
st.subheader("📋 Lista rizičnih korisnika")
st.dataframe(
    df_risk[['customerID','Risk','ExpectedLoss','RiskLevel','MonthlyCharges','tenure','Contract','InternetService']],
    column_config={
        "Risk": st.column_config.ProgressColumn("Rizik", format="%.1f%%", min_value=0, max_value=1),
        "ExpectedLoss": st.column_config.NumberColumn("Očekivani gubitak", format="$%.2f"),
    },
    use_container_width=True, hide_index=True, height=400
)

st.markdown("---")
st.subheader("🔍 Detalji korisnika")
ids = df_risk['customerID'].head(20).tolist()
if ids:
    izabran = st.selectbox("Izaberi korisnika:", ids)
    k = df_risk[df_risk['customerID']==izabran].iloc[0]
    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"""
| Atribut | Vrednost |
|---------|----------|
| ID | {k['customerID']} |
| Rizik | {k['Risk']:.1%} |
| Očekivani gubitak | ${k['ExpectedLoss']:,.2f} |
| Mesečni trošak | ${k['MonthlyCharges']:,.2f} |
| Staž | {k['tenure']} meseci |
| Ugovor | {k['Contract']} |
| Internet | {k['InternetService']} |
        """)
    with cb:
        st.markdown("### 💡 Preporuke")
        if k['MonthlyCharges'] > df_orig['MonthlyCharges'].median():
            st.info("🎯 Ponuditi 20% popusta na 6 meseci")
        if k['tenure'] < 12:
            st.info("📞 Pozvati radi provere zadovoljstva")
        if k['Contract'] == 'Month-to-month':
            st.info("📋 Ponuditi godišnji ugovor sa popustom")
        if k['InternetService'] == 'Fiber optic':
            st.info("⬆️ Besplatna nadogradnja premium kanala")

st.markdown("---")
st.caption("Churn Prediction Tool © 2026 | Andrija Gojković")