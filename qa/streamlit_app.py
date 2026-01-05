import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from pathlib import Path
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Enterprise RAG – OSFI Credit Risk Assistant",
    layout="wide"
)

st.title("Enterprise RAG Assistant – OSFI Credit Risk")
st.caption("Secure Retrieval-Augmented Generation with Guardrails")

# -----------------------------
# ENV VARIABLES
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY environment variable is not set.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -----------------------------
# USE ABSOLUTE PATH FOR VECTORSTORE
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # root of your project
FAISS_INDEX_PATH = PROJECT_ROOT / "vector_store" / "faiss_index"

@st.cache_resource(show_spinner=True)
def load_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY
    )
    return FAISS.load_local(
        str(FAISS_INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

# -----------------------------
# PROMPT WITH GUARDRAILS
# -----------------------------
PROMPT_TEMPLATE = """
You are an enterprise AI assistant answering questions strictly using the provided context.

Rules:
- Use ONLY the retrieved documents.
- If the answer is not present, respond with:
  "The provided documents do not contain sufficient information to answer this question."
- Be concise, factual, and regulatory-compliant.
- Cite sources when available.
- Return each fact or number only once.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# -----------------------------
# LLM & QA CHAIN
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# -----------------------------
# STREAMLIT UI
# -----------------------------
query = st.text_input(
    "Ask a question about OSFI Credit or Operational Risk:",
    placeholder="e.g., What are CET1 capital requirements?"
)

if st.button("Get Answer") and query:
    with st.spinner("Analyzing regulatory documents..."):
        response = qa_chain.invoke({"query": query})

        st.subheader("Answer")
        st.write(response["result"].strip())

        st.subheader("Sources")
        sources_seen = set()
        for doc in response.get("source_documents", []):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            key = f"{source} | page {page}"
            if key not in sources_seen:
                st.write(f"- {key}")
                sources_seen.add(key)
