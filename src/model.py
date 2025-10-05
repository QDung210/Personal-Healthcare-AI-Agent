import os

# Set HuggingFace Token
os.environ["HF_TOKEN"] = "token"

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from unsloth import FastLanguageModel
from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel

# Sentence Embedder for RAG
print(" Loading embedder model...")
EMBEDDER = SentenceTransformer(
    "Dqdung205/medical_vietnamese_embedding", 
    trust_remote_code=True
)

# Qdrant Vector Database Client
QDRANT_CLIENT = QdrantClient(
    url="url",
    api_key="api",
    timeout=30,
)

# Function Calling Model for Appointment Booking
print(" Loading function calling model...")
ROUTER_MODEL, ROUTER_TOKENIZER = FastLanguageModel.from_pretrained(
    model_name="Dqdung205/qwen-function-calling-model",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

# Text Generation Model for RAG (Qwen 2.5)
print(" Loading text generation model for RAG...")
TEXT_GEN_MODEL = HuggingFaceModel("Qwen/Qwen2.5-7B-Instruct")
RAG_AGENT = Agent(
    TEXT_GEN_MODEL,
    system_prompt="Bạn là bác sĩ y tế chuyên nghiệp. Hãy trả lời câu hỏi bằng tiếng Việt một cách chi tiết, dễ hiểu và có trích dẫn nguồn."
)

print(" All models loaded successfully!")
print(f"   • Embedder: Ready")
print(f"   • Qdrant Client: Connected")
print(f"   • Function Calling Model: Ready")
print(f"   • Text Generation Model (Qwen 2.5): Ready")
