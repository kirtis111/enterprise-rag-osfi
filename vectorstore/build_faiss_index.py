print("build_faiss_index.py started")

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# ===============================
# 1️. Load & Chunk PDFs
# ===============================
pdf_files = [
    r"..\data\raw_pdfs\capital_requirements.pdf",
    r"..\data\raw_pdfs\credit_risk.pdf",
    r"..\data\raw_pdfs\operational_risk.pdf"
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

all_docs = []

for pdf in pdf_files:
    print(f"Loading: {os.path.basename(pdf)}")
    loader = PyPDFLoader(pdf)
    docs = loader.load()
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(pdf)

    all_docs.extend(chunks)

print(f"Total chunks prepared: {len(all_docs)}")

# ===============================
# 2. Create Embeddings
# ===============================
print("🔐 Initializing OpenAI embeddings")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# ===============================
# 3. Build FAISS Index
# ===============================
print("Creating FAISS index")
vectorstore = FAISS.from_documents(all_docs, embeddings)

# ===============================
# 4. Save FAISS Index
# ===============================
FAISS_INDEX_PATH = "../faiss_index"

vectorstore.save_local(FAISS_INDEX_PATH)

print(f"FAISS index saved at: {FAISS_INDEX_PATH}")