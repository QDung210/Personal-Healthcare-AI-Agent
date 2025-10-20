"""
Test AWS Bedrock connection 
This script verifies that AWS credentials and permissions
are correctly configured for Amazon Bedrock.
"""
import os
import boto3
import json
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
def test_bedrock_connection():
    print(" Testing AWS Bedrock connection...\n")

    try:
        #  Test 1: Validate credentials
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        print("AWS credentials are valid!")
        print(f"   Account: {identity['Account']}")
        print(f"   User ARN: {identity['Arn']}\n")

        #  Test 2: Bedrock model list
        bedrock = boto3.client("bedrock", region_name=REGION)
        print("Checking Bedrock access...")

        try:
            response = bedrock.list_foundation_models()
            model_count = len(response.get("modelSummaries", []))
            print(f"Found {model_count} available foundation models.\n")
        except Exception as e:
            print(f"Could not list Bedrock models: {e}\n")

        #  Test 3: Try invoking a model (replace model_id with your own)
        print("Testing model invocation...")
        bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)

        # Example payload (change to fit your model)
        payload = {
            "prompt": "Hello from Bedrock test script!",
            "max_tokens": 50
        }

        # Example model (safe placeholder)
        model_id = "anthropic.claude-v2"  # <-- change to your model

        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(payload)
        )

        result = json.loads(response["body"].read())
        print(" Model invocation successful!")
        print(f"   Response snippet: {str(result)[:200]}...\n")

    except (NoCredentialsError, PartialCredentialsError):
        print(" AWS credentials not found or incomplete.")
        print("   → Run `aws configure` or set environment variables.")
    except Exception as e:
        print(f" Error: {e}")
        print("\nPossible fixes:")
        print("   1. Check that AWS credentials are valid")
        print("   2. Verify IAM policy includes Bedrock permissions")
        print("   3. Ensure model ID/ARN is correct")
        print(f"   4. Confirm AWS region: {REGION}")

if __name__ == "__main__":
    test_bedrock_connection()
