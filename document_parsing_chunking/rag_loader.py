
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# List all your PDFs
pdf_files = [
    r"..data\raw_pdfs\capital_requirements.pdf",
    r"..\data\raw_pdfs\credit_risk.pdf",
    r"..\data\raw_pdfs\operational_risk.pdf"
]

all_docs = []

# Load and split each PDF
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    
    # Add metadata for citation
    for chunk in chunks:
        chunk.metadata["source"] = pdf_file
    all_docs.extend(chunks)

print(f"Total chunks loaded: {len(all_docs)}")