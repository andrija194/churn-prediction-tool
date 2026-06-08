import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ===== PODEŠAVANJE STRANICE =====
st.set_page_config(page_title="Churn Predictor", page_icon="🔮", layout="wide")
st.title("🔮 Churn Prediction Tool")
st.markdown("---")

# ===== PUTANJE DO MODELA =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')

# ===== UČITAVANJE MODELA =====
@st.cache_resource
def load_models():
    model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    feature_names = pd.read_csv(os.path.join(MODEL_DIR, 'feature_names.csv')).iloc[:, 0].tolist()
    
    # Feature importance (ako postoji)
    imp_path = os.path.join(MODEL_DIR, 'feature_importance.csv')
    if os.path.exists(imp_path):
        importance = pd.read_csv(imp_path)
    else:
        importance = None
    
    return model, scaler, feature_names, importance

@st.cache_data
def get_data():
    sys.path.insert(0, os.path.join(BASE_DIR, '..', 'src'))
    from data_preprocessing import run_preprocessing
    X, y = run_preprocessing(os.path.join(DATA_DIR, 'WA_Telco_Customer_Churn.csv'))
    df = pd.read_csv(os.path.join(DATA_DIR, 'WA_Telco_Customer_Churn.csv'))
    return X, y, df

try:
    model, scaler, feature_names, importance = load_models()
    X, y, df_orig = get_data()
    st.success("✅ Modeli i podaci učitani!")
except Exception as e:
    st.error(f"Greška pri učitavanju: {e}")
    st.stop()

# ===== PREDIKCIJE =====
proba = model.predict_proba(X)[:, 1]
df_orig['Risk'] = proba
df_orig['ExpectedLoss'] = proba * df_orig['MonthlyCharges']
df_orig['RiskLevel'] = pd.cut(proba, bins=[0, 0.3, 0.7, 1], labels=['🟢 Nizak', '🟡 Srednji', '🔴 Visok'])

# ===== SIDEBAR =====
st.sidebar.header("🔍 Filteri")
prag = st.sidebar.slider("Prag rizika", 0.0, 1.0, 0.5, 0.05)

df_risk = df_orig[df_orig['Risk'] >= prag].sort_values('ExpectedLoss', ascending=False)

# ===== KPI =====
c1, c2, c3 = st.columns(3)
c1.metric("🔴 Rizični korisnici", len(df_risk))
c2.metric("💰 Očekivani gubitak", f"${df_risk['ExpectedLoss'].sum():,.0f}")
c3.metric("📈 Prosečan rizik", f"{df_risk['Risk'].mean():.1%}" if len(df_risk) > 0 else "0%")

st.markdown("---")

# ===== TABELA =====
st.subheader("📋 Lista rizičnih korisnika")
st.dataframe(
    df_risk[['customerID', 'Risk', 'ExpectedLoss', 'RiskLevel', 'MonthlyCharges', 'tenure', 'Contract', 'InternetService']],
    column_config={
        "Risk": st.column_config.ProgressColumn("Rizik", format="%.1f%%", min_value=0, max_value=1),
        "ExpectedLoss": st.column_config.NumberColumn("Očekivani gubitak", format="$%.2f"),
    },
    use_container_width=True, hide_index=True, height=400
)

st.markdown("---")

# ===== DETALJI =====
st.subheader("🔍 Detalji korisnika")
ids = df_risk['customerID'].head(20).tolist()
if ids:
    izabran = st.selectbox("Izaberi korisnika:", ids)
    korisnik = df_risk[df_risk['customerID'] == izabran].iloc[0]

    cA, cB = st.columns(2)
    with cA:
        st.markdown(f"""
        | Atribut | Vrednost |
        |---------|----------|
        | ID | {korisnik['customerID']} |
        | Rizik | {korisnik['Risk']:.1%} |
        | Očekivani gubitak | ${korisnik['ExpectedLoss']:,.2f} |
        | Mesečni trošak | ${korisnik['MonthlyCharges']:,.2f} |
        | Staž | {korisnik['tenure']} meseci |
        | Ugovor | {korisnik['Contract']} |
        | Internet | {korisnik['InternetService']} |
        """)
    with cB:
        st.markdown("### 💡 Preporuke")
        if korisnik['MonthlyCharges'] > df_orig['MonthlyCharges'].median():
            st.info("🎯 Ponuditi 20% popusta na 6 meseci")
        if korisnik['tenure'] < 12:
            st.info("📞 Pozvati radi provere zadovoljstva")
        if korisnik['Contract'] == 'Month-to-month':
            st.info("📋 Ponuditi godišnji ugovor sa popustom")
        if korisnik['InternetService'] == 'Fiber optic':
            st.info("⬆️ Besplatna nadogradnja premium kanala")

st.markdown("---")
st.caption("Churn Prediction Tool © 2026 | Andrija Gojković")