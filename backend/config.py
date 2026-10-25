from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DB_PATH = "./chroma_db"
DOCUMENTS_PATH = "./documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 10
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
EMBEDDING_MODEL = "all-mpnet-base-v2"