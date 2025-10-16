"""
Config main models
"""

import src.config as config  
import os

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from pydantic_ai import Agent
from src.models.bedrock_model import BedrockCustomModel

# Sentence Embedder for RAG
print("Loading embedder model...")
EMBEDDER = SentenceTransformer(
    "Dqdung205/medical_vietnamese_embedding", 
    trust_remote_code=True
)

# Qdrant Vector Database Client
print("Connecting to Qdrant...")
QDRANT_CLIENT = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY'),
    timeout=int(os.getenv('QDRANT_TIMEOUT', '30')),
)

# Function Calling Model for Function Calling 
print("Loading function calling model from Bedrock...")
ROUTER_MODEL = BedrockCustomModel(
    model_arn=os.getenv('BEDROCK_MODEL_ARN')
)

# Text Generation Model for RAG 
print("Loading text generation model for RAG...")
from pydantic_ai.models.huggingface import HuggingFaceModel
TEXT_GEN_MODEL = HuggingFaceModel("Qwen/Qwen2.5-7B-Instruct")

# RAG Agent
RAG_AGENT = Agent(
    TEXT_GEN_MODEL,
    system_prompt="Bạn là bác sĩ y tế chuyên nghiệp. Hãy trả lời câu hỏi bằng tiếng Việt một cách chi tiết, dễ hiểu và có trích dẫn nguồn."
)

print("All models loaded successfully!")
