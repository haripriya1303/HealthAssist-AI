from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

# Credentials
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Model Setup (Groq)
chatModel = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    temperature=0,
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# Embeddings & Vector Store
embedding = download_embeddings()
index_name = "healthassist-ai" # Must be lowercase to match store_index.py

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# RAG Chain Setup
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User Query: {msg}")
    
    # Generate Response
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    
    return str(response["answer"])

if __name__ == '__main__':
    # Change port from 8080 to 8000
    app.run(host="0.0.0.0", port=8000, debug=True)

