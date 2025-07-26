"""Preprocessing helpers for preparing model input."""

import pandas as pd
import json

def prepare_feature_input(event: dict) -> pd.DataFrame:
    """Convert a JSON event into a DataFrame suitable for feature extraction."""

    # parse san_identities string into list
    san_list = json.loads(event["san_identities"]) if isinstance(event["san_identities"], str) else event["san_identities"]
    domain_created_date = json.loads(event["domain_created"]) if isinstance(event["domain_created"], str) else event["domain_created"]

    shared_metadata = {
        "not_before": event.get("not_before"),
        "not_after": event.get("not_after"),
        "ca_name": event.get("ca_name"),
        "certificate_type": event.get("certificate_type"),
        "no_domains": len(san_list)
    }

    # Create a row for each domain + corresponding creation_date
    rows = []
    for domain, created_date in zip(san_list, domain_created_date):
        row = {
            "parsed_domainname": domain,
            "creation_date": created_date,
            **shared_metadata
        }
        rows.append(row)
           
    return pd.DataFrame(rows)
