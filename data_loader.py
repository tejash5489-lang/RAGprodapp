import os
import fitz  # pymupdf

from google import genai
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(file_path):
    doc = fitz.open(file_path)
    texts = [page.get_text() for page in doc]
    doc.close()
    chunks = []
    for t in texts:
        if t.strip():
            chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return [embedding.values for embedding in response.embeddings]