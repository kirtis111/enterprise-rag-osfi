"""
Enterprise RAG Q&A with Hallucination Guardrails
For Safety, Precision, Non-Redundancy
"""

import os
from datetime import datetime
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

# -----------------------------
# CONFIG
# -----------------------------
FAISS_INDEX_PATH = "../vectorstore/faiss_index"

print("Enterprise RAG Q&A with Guardrails & Safety Prompts started\n")

# -----------------------------
# LOAD VECTOR STORE
# -----------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# LLM
# -----------------------------
llm = OpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo-instruct"
)


# -----------------------------
# ENTERPRISE SAFETY PROMPT
# -----------------------------
GUARDRAIL_PROMPT = """
You are an enterprise financial risk assistant.

STRICT RULES:
- Use ONLY the provided context.
- If the answer is not present, respond exactly with: "I don't know."
- Return each fact, number, or bullet point ONLY ONCE.
- Do NOT repeat ratios, percentages, or headings.
- Be concise and factual.
- Use bullet points where appropriate.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=GUARDRAIL_PROMPT
)

# -----------------------------
# QA CHAIN
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "document_variable_name": "context",
        "prompt": prompt
    }
)

# -----------------------------
# TEST QUERIES
# -----------------------------
queries = [
    "What are OSFI capital requirements related to credit risk?",
    "List Tier 1 and CET1 capital ratios.",
    "Explain the Capital Conservation Buffer requirements.",
    "What are countercyclical buffer rules?",
    "What are the OSFI operational risk guidelines?"
]

# -----------------------------
# RUN QUERIES
# -----------------------------
for query in queries:
    print("Query:", query)

    response = qa_chain.invoke({"query": query})

    print("Answer:")
    print(response["result"].strip())

    print("\nSOURCES:")
    sources_seen = set()

    for doc in response["source_documents"]:
        source = doc.metadata.get("source", "Unknown file")
        page = doc.metadata.get("page", "Unknown page")
        key = f"{source} | page {page}"

        if key not in sources_seen:
            print(f"- {key}")
            sources_seen.add(key)

    print("\n" + "-" * 80 + "\n")