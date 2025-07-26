import pandas as pd
import joblib
from domain_extractor import DomainFeatureExtractor

# Load whitelist from CSV (update path as needed)
whitelist_csv = 'running_whitelist_reduced100k.csv'
df = pd.read_csv(whitelist_csv)
whitelist = df['domain'].dropna().unique().tolist()

# Instantiate extractor
extractor = DomainFeatureExtractor(whitelist)

# Save to .pkl
joblib.dump(extractor, 'domain_feature_extractor.pkl')

print("✅ domain_feature_extractor.pkl created successfully.")
