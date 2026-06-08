import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath="data/raw/WA_Telco_Customer_Churn.csv"):
    print("[1/3] Ucitavanje podataka...")
    df = pd.read_csv(filepath)
    df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    print(f"   Ucitano: {df.shape[0]} redova, {df.shape[1]} kolona")
    return df

def feature_engineering(df):
    print("[2/3] Feature engineering...")
    
    # Broj dodatnih servisa
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    for s in services:
        df[f'Has_{s}'] = (df[s] == 'Yes').astype(int)
    df['NumServices'] = df[[f'Has_{s}' for s in services]].sum(axis=1)
    
    # Prosek mesecnih troskova
    df['AvgMonthly'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # Interakcija: skup fiber + mesecni ugovor
    df['Fiber_Monthly'] = ((df['InternetService'] == 'Fiber optic') & 
                           (df['Contract'] == 'Month-to-month')).astype(int)
    
    # Tenure grupe
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0,12,36,100], 
                               labels=['Novi', 'Srednji', 'Lojalni'])
    
    print(f"   Kreirano feature-a, ukupno kolona: {df.shape[1]}")
    return df

def encode_and_scale(df):
    print("[3/3] Encoding i skaliranje...")
    df_enc = df.copy()
    
    cat_cols = df_enc.select_dtypes(include=['object', 'category']).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    
    if 'Churn' in df_enc.columns:
        y = df_enc['Churn'].copy()
        X = df_enc.drop('Churn', axis=1)
    else:
        X = df_enc
        y = None
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    X.columns.to_series().to_csv('models/feature_names.csv', index=False)
    
    print(f"   Gotovo! X shape: {X_scaled.shape}")
    return X_scaled, y

def run_preprocessing(filepath="data/raw/WA_Telco_Customer_Churn.csv"):
    print("="*40)
    print("POKRECEM PREPROCESSING")
    print("="*40)
    df = load_data(filepath)
    df = feature_engineering(df)
    X, y = encode_and_scale(df)
    return X, y

if __name__ == "__main__":
    X, y = run_preprocessing()
    print("\nDistribucija churn-a:")
    print(y.value_counts())