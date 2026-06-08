# 🔮 Churn Prediction & Retention Tool

**Predmet:** Mašinsko učenje - Praktični projekat  
**Student:** Andrija Gojković  
**Datum:** Jun 2026.

---

## 📌 Opis

Alat za predikciju churn-a u telekomunikacionoj kompaniji koji:
- Predviđa rizične korisnike (XGBoost + Survival Analysis)
- Objašnjava razloge rizika
- Predlaže akcije za zadržavanje
- Računa očekivani ROI

**Dataset:** Telco Customer Churn (Kaggle)

---

## 🚀 Kako pokrenuti

```bash
git clone https://github.com/andrija194/churn-prediction-tool.git
cd churn-prediction-tool
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python src/churn_model.py
python src/survival_analysis.py
streamlit run app/main.py