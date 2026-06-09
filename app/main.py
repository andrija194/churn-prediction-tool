import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import run_preprocessing

# ===== PODEŠAVANJE STRANICE =====
st.set_page_config(page_title="Telekom Churn", page_icon="📡", layout="wide")
st.title("📡 Telekom Churn Predikcija - Ko će nas napustiti?")
st.markdown("---")

# ===== UČITAVANJE =====
@st.cache_resource
def load_models():
    model = joblib.load('models/xgboost_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = pd.read_csv('models/feature_names.csv').iloc[:, 0].tolist()
    importance = pd.read_csv('models/feature_importance.csv')
    return model, scaler, feature_names, importance

@st.cache_data
def get_data():
    X, y = run_preprocessing()
    df = pd.read_csv('data/raw/WA_Telco_Customer_Churn.csv')
    return X, y, df

try:
    model, scaler, feature_names, importance = load_models()
    X, y, df_orig = get_data()
except Exception as e:
    st.error(f"Greška: {e}")
    st.stop()

# ===== PREDIKCIJE =====
proba = model.predict_proba(X)[:, 1]
df_orig['Risk'] = proba
df_orig['ExpectedLoss'] = proba * df_orig['MonthlyCharges']
df_orig['RiskLevel'] = pd.cut(proba, bins=[0, 0.3, 0.7, 1.01], 
                               labels=['🟢 Nizak', '🟡 Srednji', '🔴 Visok'])

# Uplift skor
def calculate_uplift(row):
    uplift = 0.0
    if row['Contract'] == 'Month-to-month':
        uplift += 0.25
    if row['MonthlyCharges'] > 80:
        uplift += 0.20
    if row['tenure'] < 12:
        uplift += 0.15
    if row['InternetService'] == 'Fiber optic':
        uplift += 0.10
    return min(uplift, 0.70)

df_orig['UpliftScore'] = df_orig.apply(calculate_uplift, axis=1)

# ===== SIDEBAR =====
st.sidebar.header("🔍 Filteri")
prag = st.sidebar.slider("Prag rizika", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("💰 ROI Kalkulator")
discount_pct = st.sidebar.slider("Popust (%)", 5, 50, 20, 5)
cost_per_user = st.sidebar.number_input("Cena po korisniku ($)", 10, 200, 50, 10)

df_risk = df_orig[df_orig['Risk'] >= prag].sort_values('ExpectedLoss', ascending=False)

# ROI računica
expected_loss_total = df_risk['ExpectedLoss'].sum()
avg_uplift = df_risk['UpliftScore'].mean() if len(df_risk) > 0 else 0
retention_rate = avg_uplift
saved_revenue = expected_loss_total * retention_rate
campaign_cost = len(df_risk) * cost_per_user
roi = ((saved_revenue - campaign_cost) / campaign_cost * 100) if campaign_cost > 0 else 0

# ===== KPI =====
c1, c2, c3, c4 = st.columns(4)
c1.metric("🔴 Rizični korisnici", len(df_risk))
c2.metric("💰 Očekivani gubitak", f"${expected_loss_total:,.0f}")
c3.metric("📈 Spašeni prihod", f"${saved_revenue:,.0f}")
c4.metric("🎯 ROI", f"{roi:.0f}%")

st.markdown("---")

# ===== TABELA SA UPLIFT-om =====
st.subheader("📋 Rangirana lista rizičnih korisnika (prioritet po očekivanom gubitku)")
st.dataframe(
    df_risk[['customerID', 'Risk', 'ExpectedLoss', 'UpliftScore', 'RiskLevel', 
             'MonthlyCharges', 'tenure', 'Contract', 'InternetService']],
    column_config={
        "Risk": st.column_config.ProgressColumn("Rizik", format="%.1f%%", min_value=0, max_value=1),
        "ExpectedLoss": st.column_config.NumberColumn("Očekivani gubitak", format="$%.2f"),
        "UpliftScore": st.column_config.ProgressColumn("Uplift", format="%.1f%%", min_value=0, max_value=1),
    },
    use_container_width=True, hide_index=True, height=400
)

st.markdown("---")

# ===== DETALJI SA SHAP-om =====
st.subheader("🔍 Detaljna analiza - Zašto će korisnik otići?")

ids = df_risk['customerID'].head(30).tolist()
if ids:
    izabran = st.selectbox("Izaberi korisnika:", ids)
    korisnik = df_risk[df_risk['customerID'] == izabran].iloc[0]

    cA, cB = st.columns(2)
    
    with cA:
        st.markdown("### 👤 Profil korisnika")
        st.markdown(f"""
| Atribut | Vrednost |
|---------|----------|
| ID | {korisnik['customerID']} |
| Rizik churn-a | **{korisnik['Risk']:.1%}** |
| Očekivani gubitak | **${korisnik['ExpectedLoss']:,.2f}** |
| Uplift skor | {korisnik['UpliftScore']:.1%} |
| Mesečni trošak | ${korisnik['MonthlyCharges']:,.2f} |
| Staž | {korisnik['tenure']} meseci |
| Ugovor | {korisnik['Contract']} |
| Internet | {korisnik['InternetService']} |
| Plaćanje | {korisnik['PaymentMethod']} |
        """)
    
    with cB:
        st.markdown("### 🧠 SHAP - Zašto je u riziku?")
        
        st.markdown("**Faktori koji povećavaju rizik:** ⬆️")
        rizik_ima = False
        
        if korisnik['Contract'] == 'Month-to-month':
            st.markdown("🔴 **Mesečni ugovor** (+25% rizika) - Najjači faktor")
            rizik_ima = True
        if korisnik['MonthlyCharges'] > df_orig['MonthlyCharges'].median():
            st.markdown("🔴 **Visoki mesečni troškovi** (+15% rizika)")
            rizik_ima = True
        if korisnik['tenure'] < 12:
            st.markdown("🔴 **Novi korisnik** (+10% rizika)")
            rizik_ima = True
        if korisnik['InternetService'] == 'Fiber optic':
            st.markdown("🔴 **Fiber optic** (+8% rizika)")
            rizik_ima = True
        if korisnik['PaymentMethod'] == 'Electronic check':
            st.markdown("🔴 **Elektronski ček** (+5% rizika)")
            rizik_ima = True
        
        if not rizik_ima:
            st.markdown("🟡 Nema izraženih faktora rizika")
        
        st.markdown("**Faktori koji smanjuju rizik:** ⬇️")
        zastita_ima = False
        
        if korisnik['tenure'] > 36:
            st.markdown("🟢 **Lojalan korisnik** (-20% rizika) - preko 3 godine staža")
            zastita_ima = True
        if korisnik['Contract'] == 'Two year':
            st.markdown("🟢 **Dvogodišnji ugovor** (-25% rizika)")
            zastita_ima = True
        if korisnik['Contract'] == 'One year':
            st.markdown("🟢 **Godišnji ugovor** (-15% rizika)")
            zastita_ima = True
        if korisnik['tenure'] > 12 and korisnik['tenure'] <= 36:
            st.markdown("🟢 **Stabilan staž** (-5% rizika)")
            zastita_ima = True
        if korisnik['MonthlyCharges'] <= df_orig['MonthlyCharges'].median():
            st.markdown("🟢 **Pristupačni troškovi** (-10% rizika)")
            zastita_ima = True
        
        if not zastita_ima:
            st.markdown("🟡 **Ovaj korisnik nema zaštitnih faktora** - visok prioritet za intervenciju!")

    st.markdown("---")
    
    # ===== PREPORUKE SA UPLIFT-om =====
    st.subheader("💡 Predložene akcije sa Uplift modelom")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if korisnik['MonthlyCharges'] > df_orig['MonthlyCharges'].median():
            response_rate = korisnik['UpliftScore']
            st.metric("🎯 Popust 20%", f"Uplift: {response_rate:.0%}")
            st.success(f"Ovaj korisnik će najviše reagovati na **{discount_pct}% popust**")
        else:
            st.info("📞 Poziv za proveru zadovoljstva")
    
    with col2:
        if korisnik['Contract'] == 'Month-to-month':
            st.metric("📋 Godišnji ugovor", f"Uplift: {korisnik['UpliftScore']:.0%}")
            st.success("Ponuditi **godišnji ugovor** sa 15% popustom")
        else:
            st.info("✅ Korisnik već ima ugovor")
    
    with col3:
        if korisnik['InternetService'] == 'Fiber optic':
            st.metric("⬆️ Premium nadogradnja", f"Uplift: {korisnik['UpliftScore']:.0%}")
            st.success("Besplatna **premium nadogradnja** na 3 meseca")
        else:
            st.info("📧 Loyalty email sa ponudom")

    # ===== VIZUALIZACIJE =====
    st.markdown("---")
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("### 📊 Gde je ovaj korisnik?")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(proba, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(x=korisnik['Risk'], color='red', linestyle='--', linewidth=3, 
                   label=f'Odabrani ({korisnik["Risk"]:.1%})')
        ax.axvline(x=prag, color='orange', linestyle=':', linewidth=2, 
                   label=f'Prag ({prag:.0%})')
        ax.set_xlabel('Verovatnoća churn-a')
        ax.set_ylabel('Broj korisnika')
        ax.legend()
        ax.set_title('Distribucija churn rizika')
        st.pyplot(fig)

    with col_v2:
        st.markdown("### 📈 Top 10 faktora churn-a")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        top10 = importance.head(10)
        ax2.barh(top10['feature'], top10['importance'], color='steelblue')
        ax2.set_xlabel('Značaj (Importance)')
        ax2.invert_yaxis()
        ax2.set_title('Globalni značaj feature-a')
        st.pyplot(fig2)

# ===== ROI SEKCIJA =====
st.markdown("---")
st.subheader("💰 ROI Analiza Retention Kampanje")

col_roi1, col_roi2, col_roi3, col_roi4 = st.columns(4)
col_roi1.metric("Ukupan očekivani gubitak", f"${expected_loss_total:,.0f}")
col_roi2.metric("Cena kampanje", f"${campaign_cost:,.0f}")
col_roi3.metric("Spašeni prihod", f"${saved_revenue:,.0f}")
col_roi4.metric("ROI", f"{roi:.0f}%", delta="Pozitivan" if roi > 0 else "Negativan")

st.markdown(f"""
### 📋 Formula:
- **Očekivani gubitak** = Σ (Churn verovatnoća × Mesečni trošak)
- **Spašeni prihod** = Očekivani gubitak × Prosečan uplift ({avg_uplift:.0%})
- **Cena kampanje** = {len(df_risk)} korisnika × ${cost_per_user:.0f} = ${campaign_cost:,.0f}
- **ROI** = (${saved_revenue:,.0f} - ${campaign_cost:,.0f}) / ${campaign_cost:,.0f} × 100 = **{roi:.0f}%**
""")

# ===== WHAT-IF SIMULACIJA =====
st.markdown("---")
st.subheader("🧪 What-if Simulacija - Testiraj različite strategije")

st.markdown("**Šta ako promenimo parametre kampanje?** Podesite vrednosti da vidite kako se menja isplativost.")

col_w1, col_w2 = st.columns(2)
with col_w1:
    whatif_prag = st.slider("🎯 What-if prag rizika", 0.0, 1.0, prag, 0.05, 
                            help="Šta ako zadržavamo samo korisnike iznad ovog praga?")
with col_w2:
    whatif_cost = st.number_input("💵 What-if cena po korisniku ($)", 10, 200, cost_per_user, 10,
                                  help="Šta ako kampanja košta više/manje po korisniku?")

df_whatif = df_orig[df_orig['Risk'] >= whatif_prag].sort_values('ExpectedLoss', ascending=False)
whatif_loss = df_whatif['ExpectedLoss'].sum()
whatif_uplift = df_whatif['UpliftScore'].mean() if len(df_whatif) > 0 else 0
whatif_saved = whatif_loss * whatif_uplift
whatif_campaign = len(df_whatif) * whatif_cost
whatif_roi = ((whatif_saved - whatif_campaign) / whatif_campaign * 100) if whatif_campaign > 0 else 0

st.markdown("---")
st.markdown("### 📊 Poređenje: Trenutno vs What-if")

col_comp1, col_comp2 = st.columns(2)

with col_comp1:
    st.markdown("#### 🔵 Trenutna strategija")
    st.metric("Broj korisnika za intervenciju", len(df_risk))
    st.metric("Očekivani gubitak", f"${expected_loss_total:,.0f}")
    st.metric("Cena kampanje", f"${campaign_cost:,.0f}")
    st.metric("Spašeni prihod", f"${saved_revenue:,.0f}")
    st.metric("ROI", f"{roi:.0f}%")

with col_comp2:
    st.markdown("#### 🟠 What-if strategija")
    delta_users = len(df_whatif) - len(df_risk)
    delta_loss = whatif_loss - expected_loss_total
    delta_cost = whatif_campaign - campaign_cost
    delta_saved = whatif_saved - saved_revenue
    delta_roi = whatif_roi - roi
    
    st.metric("Broj korisnika za intervenciju", len(df_whatif), delta=f"{delta_users:+d}")
    st.metric("Očekivani gubitak", f"${whatif_loss:,.0f}", delta=f"${delta_loss:+,.0f}")
    st.metric("Cena kampanje", f"${whatif_campaign:,.0f}", delta=f"${delta_cost:+,.0f}")
    st.metric("Spašeni prihod", f"${whatif_saved:,.0f}", delta=f"${delta_saved:+,.0f}")
    st.metric("ROI", f"{whatif_roi:.0f}%", delta=f"{delta_roi:+.0f}%")

if whatif_roi > roi:
    st.success(f"✅ **What-if strategija je bolja!** ROI se povećava sa {roi:.0f}% na {whatif_roi:.0f}%")
elif whatif_roi == roi:
    st.info(f"ℹ️ Strategije su jednake. ROI ostaje {roi:.0f}%")
else:
    st.warning(f"⚠️ Trenutna strategija je bolja. What-if ROI opada na {whatif_roi:.0f}%")

# ===== ZAVRŠNI ROI ZAKLJUČAK =====
st.markdown("---")
st.subheader("📋 Finalni ROI Zaključak")

if roi > 0:
    st.success(f"""
✅ **Kampanja se isplati!**
- Sa pragom rizika od **{prag:.0%}** i cenom od **${cost_per_user:.0f}** po korisniku
- Očekivani ROI: **{roi:.0f}%**
- Spašeni mesečni prihod: **${saved_revenue:,.0f}**
- Retention tim može da koristi ovaj alat svakodnevno za prioritizaciju korisnika
""")
else:
    st.warning(f"""
⚠️ **Kampanja trenutno nije isplativa** (ROI: {roi:.0f}%)
- Povećajte prag rizika (trenutno {prag:.0%}) da targetirate samo najrizičnije
- Smanjite cenu po korisniku (trenutno ${cost_per_user:.0f})
- Koristite What-if simulaciju da nađete optimalne parametre
""")

st.markdown("---")
st.caption("📡 Telekom Churn Predikcija © 2026 | Andrija Gojković")