"""
Load environment variables from .env file
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# AWS Credentials
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', '')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', '')
os.environ['AWS_DEFAULT_REGION'] = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

# Hugging Face Token
os.environ["HF_TOKEN"] = os.getenv('HF_TOKEN', '')

# Validate required environment variables
required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'HF_TOKEN']
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
