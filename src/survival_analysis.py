import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
from data_preprocessing import load_data, feature_engineering

def train_survival():
    print("\n" + "="*40)
    print("TRENIRAM SURVIVAL MODEL")
    print("="*40)
    
    df = load_data()
    df = feature_engineering(df)
    
    # Konvertuj sve u numericko
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = pd.factorize(df[col].astype(str))[0]
    
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df, duration_col='tenure', event_col='Churn')
    
    print(f"\nConcordance Index: {cph.concordance_index_:.4f}")
    print("(>0.7 je dobro, >0.8 odlicno)")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(cph, 'models/survival_model.pkl')
    print("\nModel sacuvan: models/survival_model.pkl")
    
    return cph

if __name__ == "__main__":
    train_survival()