"""
aws_config.py — AWS region for all Bedrock calls (agents + LLM judges).

Set AWS_REGION to override the default:
    export AWS_REGION=us-west-2
"""
import os

AWS_REGION: str = os.environ.get("AWS_REGION", "us-west-2")
