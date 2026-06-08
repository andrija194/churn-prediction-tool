import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
from data_preprocessing import run_preprocessing

def train_model(X, y):
    print("\n" + "="*40)
    print("TRENIRAM XGBOOST MODEL")
    print("="*40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scale_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_weight, random_state=42,
        eval_metric='auc', verbosity=0
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 feature-a:")
    print(importance.head(10).to_string(index=False))
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_model.pkl')
    importance.to_csv('models/feature_importance.csv', index=False)
    print("\nModel sacuvan: models/xgboost_model.pkl")
    
    return model

if __name__ == "__main__":
    X, y = run_preprocessing()
    model = train_model(X, y)
    print("\nGOTOVO!")