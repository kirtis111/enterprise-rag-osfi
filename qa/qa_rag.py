# ---------------------------
# Imports
# ---------------------------
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
import os
import csv
import json
from datetime import datetime

# ---------------------------
# Load Environment Variables
# ---------------------------
load_dotenv()  # loads OPENAI_API_KEY from .env
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

# ---------------------------
# Load Embeddings and Vectorstore
# ---------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = FAISS.load_local(
    "../faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# ---------------------------
# Setup Retriever
# ---------------------------
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ---------------------------
# Initialize LLM
# ---------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------
# Build RetrievalQA Chain
# ---------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# ---------------------------
# Define your query
# ---------------------------
query = "What are OSFI capital requirements related to credit risk?"

# ---------------------------
# Run QA
# ---------------------------
result = qa_chain.invoke(query)

# ---------------------------
# Print Answer
# ---------------------------
print("\nANSWER:\n")
print(result["result"])

print("\nSOURCES:\n")
for doc in result["source_documents"]:
    print(f"- {doc.metadata.get('source')} | page {doc.metadata.get('page', 'N/A')}")

# ---------------------------
# STEP 6: Logging QA Interactions
# ---------------------------

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- CSV Logging ---
log_file_csv = "qa_logs.csv"
sources_csv = "; ".join([
    f"{doc.metadata.get('source')} | page {doc.metadata.get('page', 'N/A')}" 
    for doc in result["source_documents"]
])
with open(log_file_csv, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([timestamp, query, result["result"], sources_csv])
print(f"\nLogged QA interaction to {log_file_csv}")

# --- JSON Logging ---
log_file_json = "qa_logs.json"
log_entry = {
    "timestamp": timestamp,
    "query": query,
    "answer": result["result"],
    "sources": [
        {"file": doc.metadata.get("source"), "page": doc.metadata.get("page", "N/A")}
        for doc in result["source_documents"]
    ]
}

try:
    with open(log_file_json, "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    data = []

data.append(log_entry)

with open(log_file_json, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Logged QA interaction to {log_file_json}")