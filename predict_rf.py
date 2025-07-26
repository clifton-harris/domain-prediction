"""Prediction helper that loads RandomForest models and scores domains."""

import pandas as pd
import joblib

# Path to the pickled model
#MODEL_PATH = "/var/task/rf_classifier.pkl"
MODEL_A_PATH = 'rf_classifier.pkl'
MODEL_B_PATH = 'rf_classifier_whois.pkl'

# Columns the model expects
SELECTED_FEATURES = [
    'no_domains',
    'validity_days',
    'domain_length',
    'deeply_nested_subdomains',
    'num_subdomains',
    'hyphen_count',
    'shannon_entropy',
    'suspicious_keyword',
    'suspicious_tld',
    'has_inner_tld',
    'issued_hour_utc',
    'is_weekend',
    'has_unusual_token',
    'ca_name_Buypass',
    'ca_name_DigiCert',
    'ca_name_GlobalSign nv-sa',
    'ca_name_GoDaddy',
    'ca_name_Google Trust Services LLC',
    'ca_name_Internet Security Research Group',
    'ca_name_SSL.com',
    'ca_name_Sectigo',
    'max_jaro_winkler_similarity',
    'parsed_domainname',  # Optional: for identifying outputs
    'domain_age_days'
]

model_a = joblib.load(MODEL_A_PATH)
model_b = joblib.load(MODEL_B_PATH)

def predict_malicious(df: pd.DataFrame) -> pd.DataFrame:
    """Score each row and indicate which model was used."""

    input_df = df[SELECTED_FEATURES].copy()
    features_only = input_df.drop(columns=['parsed_domainname'], errors='ignore')

    # Split DataFrame based on whether domain_age_days is null
    null_mask = df['domain_age_days'].isnull()

    # Predict using model_a (no domain_age_days)
    features_a = features_only[null_mask].drop(columns=["domain_age_days"], errors="ignore")
    preds_a = model_a.predict_proba(features_a)[:, 1] if not features_a.empty else pd.Series(dtype=float, index=features_a.index)

    # Predict using model_b (has domain_age_days)
    features_b = features_only[~null_mask]
    preds_b = model_b.predict_proba(features_b)[:, 1] if not features_b.empty else pd.Series(dtype=float, index=features_b.index)

    # Combine predictions and model labels using pd.concat
    predictions = pd.concat([
        pd.Series(preds_a, index=features_a.index),
        pd.Series(preds_b, index=features_b.index)
    ]).sort_index()

    model_used = pd.concat([
        pd.Series(['model_no_whois'] * len(preds_a), index=features_a.index),
        pd.Series(['model_whois'] * len(preds_b), index=features_b.index)
    ]).sort_index()

    input_df['prediction'] = predictions
    input_df['model_used'] = model_used

    return input_df



