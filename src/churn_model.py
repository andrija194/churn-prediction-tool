"""
XGBoost model za predikciju churn-a.
Uključuje baseline (Logistic Regression) i XGBoost sa evaluacijom.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
from data_preprocessing import run_preprocessing


def train_model(X, y):
    """
    Trenira baseline (Logistic Regression) i XGBoost model.
    Prikazuje poboljšanje XGBoost-a nad baseline-om.
    """
    print("\n" + "="*50)
    print("TRENIRANJE MODELA")
    print("="*50)
    
    # Podeli na train i test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Churn rate u train: {y_train.mean():.2%}")
    
    # ===== BASELINE MODEL - Logistic Regression =====
    print("\n" + "-"*40)
    print("BASELINE: Logistic Regression")
    print("-"*40)
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_pred = lr.predict(X_test)
    
    lr_auc = roc_auc_score(y_test, lr_proba)
    print(f"ROC-AUC: {lr_auc:.4f}")
    print("\nClassification Report (Baseline):")
    print(classification_report(y_test, lr_pred, target_names=['No Churn', 'Churn']))
    
    # ===== XGBOOST MODEL (poboljšanje nad baseline-om) =====
    print("\n" + "-"*40)
    print("XGBOOST (poboljšanje nad baseline-om)")
    print("-"*40)
    
    scale_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric='auc',
        verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    xgb_auc = roc_auc_score(y_test, y_proba)
    
    print(f"ROC-AUC: {xgb_auc:.4f}")
    print(f"Poboljšanje nad baseline-om: +{(xgb_auc - lr_auc):.4f} ({(xgb_auc - lr_auc) / lr_auc * 100:.1f}%)")
    
    print("\nClassification Report (XGBoost):")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives:  {cm[0][0]:>5}  (Ispravno: neće otići)")
    print(f"False Positives: {cm[0][1]:>5}  (Greška: predviđen churn, ali neće)")
    print(f"False Negatives: {cm[1][0]:>5}  (Greška: nije predviđen churn, ali hoće)")
    print(f"True Positives:  {cm[1][1]:>5}  (Ispravno: predviđen churn)")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Najvažnijih Feature-a:")
    print(importance.head(10).to_string(index=False))
    
    # Sačuvaj model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_model.pkl')
    importance.to_csv('models/feature_importance.csv', index=False)
    
    print(f"\n✓ XGBoost model sačuvan: models/xgboost_model.pkl")
    print(f"✓ Feature importance sačuvana: models/feature_importance.csv")
    
    # Prikaži poređenje
    print("\n" + "="*50)
    print("POREDENJE MODELA")
    print("="*50)
    print(f"┌─────────────────────┬──────────┐")
    print(f"│ Model               │ ROC-AUC  │")
    print(f"├─────────────────────┼──────────┤")
    print(f"│ Logistic Regression │ {lr_auc:.4f}   │")
    print(f"│ XGBoost             │ {xgb_auc:.4f}   │")
    print(f"│ Poboljšanje         │ +{(xgb_auc - lr_auc):.4f}   │")
    print(f"└─────────────────────┴──────────┘")
    
    return model, lr_auc, xgb_auc


if __name__ == "__main__":
    print("1. Pokrećem preprocessing...")
    X, y = run_preprocessing()
    
    print("\n2. Treniram modele...")
    model, lr_auc, xgb_auc = train_model(X, y)
    
    print("\n" + "="*50)
    print("TRENIRANJE ZAVRŠENO!")
    print("="*50)