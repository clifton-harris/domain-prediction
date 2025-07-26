import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from domain_extractor import DomainFeatureExtractor
from preprocess import prepare_feature_input
from predict_rf import predict_malicious

sample_events = [
    {
        "san_identities": [
            "googlevads-cn.com",
            "*.ficoanalyticcloud.com",
            "*.dms.apset2.ficoanalyticcloud.com",
            "*.cqa.worldpay.ficoanalyticcloud.com",
            "*.dms.uset2.ficoanalyticcloud.com",
            "*.insurancescores.fico.com",
            "*.ort.pnc.fairisaac.com",
            "*.dms.cact1.ficoanalyticcloud.com",
            "*.ficoccs-cqa.net",
            "*.dms.apset1.ficoanalyticcloud.com",
            "*.ficoflows-stage.net",
            "yubikey.dev.sss.fico.com",
            "fawb.dms.uset2.ficoanalyticcloud.com",
            "*.qa4.appliance.fico.com"
        ],
        "not_before": "2025-03-19T08:01:22Z",
        "not_after": "2025-09-15T08:01:22Z",
        "ca_name": "Google Trust Services LLC",
        "domain_created": ["2021-04-19 21:44:18.000", None, None, None, None, None, None, None, None, None, None, None]
    },
    {
        "san_identities": ["secure-login-paypal.net"],
        "not_before": "2024-06-01T00:00:00Z",
        "not_after": "2024-09-01T00:00:00Z",
        "ca_name": "GoDaddy",
        "domain_created": ["2024-05-01T00:00:00"]
    }
]

def test_prediction_workflow():
    whitelist = ["google.com", "example.com"]
    extractor = DomainFeatureExtractor(whitelist)
    results = []
    for record in sample_events:
        df_input = prepare_feature_input(record)
        df_features = extractor.extract(df_input)
        preds = predict_malicious(df_features)
        results.append(preds)
    assert len(results) == 2
    assert all("prediction" in df.columns for df in results)
