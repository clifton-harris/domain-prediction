import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocess import prepare_feature_input

def test_prepare_feature_input_single():
    event = {
        "san_identities": ["example.com"],
        "not_before": "2024-01-01T00:00:00Z",
        "not_after": "2024-12-31T00:00:00Z",
        "ca_name": "GoDaddy",
        "domain_created": ["2022-01-01T00:00:00"]
    }
    df = prepare_feature_input(event)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1
    assert df["parsed_domainname"].iloc[0] == "example.com"
