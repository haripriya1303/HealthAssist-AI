from dotenv import load_dotenv
import os

import pinecone
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

extracted_data = load_pdf_file(data='data/')
filter_data =filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

embedding= download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc= Pinecone(pinecone_api_key)

index_name = "healthassist-ai"

# Step 1: Create index if not exists
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Step 2: Connect to index
index = pc.Index(index_name)

# Step 3: Check if data already exists
index_stats = index.describe_index_stats()
total_vectors = index_stats.get("total_vector_count", 0)

print(f"Vectors in index: {total_vectors}")

# Step 4: Upload only if empty
if total_vectors == 0:
    print("Uploading embeddings for the first time...")

    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embedding,
        index_name=index_name
    )

    print("Upload completed ✅")

else:
    print("Index already contains data. Skipping upload ✅")

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )