"""AWS Lambda entry point for domain prediction service."""

import json
import logging
import os
from io import BytesIO

import boto3
import joblib
import pandas as pd

from domain_extractor import DomainFeatureExtractor
from preprocess import prepare_feature_input

from predict_rf import predict_malicious
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load whitelist and model at cold start
s3 = boto3.client("s3")
bucket = os.environ.get("WHITELIST_BUCKET", "capstonedata05212025")
whitelist_key = os.environ.get(
    "WHITELIST_KEY", "feature_extractor/running_whitelist_100k.csv"
)

try:
    obj = s3.get_object(Bucket=bucket, Key=whitelist_key)
    whitelist_df = pd.read_csv(obj["Body"])
    whitelist = whitelist_df["domain"].dropna().unique().tolist()
    logger.info("Whitelist loaded from S3 with %d entries", len(whitelist))
except Exception as exc:  # pragma: no cover - network errors
    logger.error("Failed to load whitelist from S3: %s", exc)
    whitelist = []
#print(f"Memory after whitelist: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")

# Initialize extractor once
extractor = DomainFeatureExtractor(whitelist)
#print(f"Memory after DomainFeatureExtractor: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")

def load_pickle_from_s3(bucket: str, key: str):
    """Load a pickled object from S3."""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return joblib.load(BytesIO(obj["Body"].read()))

def load_whitelist_from_s3(bucket: str, key: str):
    """Retrieve a whitelist CSV from S3."""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj["Body"])
    return df["domain"].dropna().unique().tolist()

#import json
#from preprocess import prepare_feature_input

def lambda_handler(event, context):
    """Handle Lambda invocation and return predictions."""

    logger.info("Received event")

    # 🧩 If event is a string (e.g., from test console), parse it
    if isinstance(event, str):
        try:
            event = json.loads(event)
        except json.JSONDecodeError:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid JSON string"})
            }

    # 🌐 If triggered via API Gateway with JSON in 'body', parse that
    if isinstance(event, dict) and "body" in event:
        try:
            event = json.loads(event["body"])
        except json.JSONDecodeError:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid JSON in request body"})
            }

    # 🧰 Batch handling
    if isinstance(event, list):
        all_results = []
        for record in event:
            df_input = prepare_feature_input(record)
            df_features = extractor.extract(df_input)
            predictions = predict_malicious(df_features)
            predictions_f = predictions[['parsed_domainname', 'prediction','model_used']]
            all_results.extend(json.loads(predictions_f.to_json(orient="records")))
        logger.info("Processed %d records", len(all_results))
        return {
            "statusCode": 200,
            "body": json.dumps({"features": all_results})
        }

    # 🧪 Single-record handling
    df_input = prepare_feature_input(event)
    df_features = extractor.extract(df_input)
    predictions = predict_malicious(df_features)
    predictions_f = predictions[['parsed_domainname', 'prediction','model_used']]

    logger.info("Processed single record")
    return {
        "statusCode": 200,
        "body": json.dumps({"features": json.loads(predictions_f.to_json(orient="records"))})
    }


if __name__ == "__main__":
    with open("test_event.json") as f:
        event = json.load(f)
    result = lambda_handler(event, None)
    logger.info(json.dumps(result, indent=2))
