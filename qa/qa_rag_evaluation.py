"""
Enterprise RAG Q&A Evaluation with Guardrails
"""

import os
import csv
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

print("Enterprise RAG Q&A Evaluation Started")

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY
)

# -----------------------------
# LOAD FAISS INDEX
# -----------------------------
vectorstore = FAISS.load_local(
    "../vectorstore/faiss_index",  # absolute/relative path to your index
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # top 3 relevant docs
)

# -----------------------------
# INITIALIZE LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

# -----------------------------
# GUARDRAIL PROMPT (matches production RAG)
# -----------------------------
GUARDRAIL_PROMPT = """
You are an enterprise financial risk assistant.

STRICT RULES:
- Use ONLY the provided context.
- If the answer is not present, respond exactly with: "I don't know."
- Return each fact, number, or bullet point ONLY ONCE.
- Be concise and factual.

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
# BUILD QA CHAIN
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt,
        "document_variable_name": "context"
    }
)

# -----------------------------
# TEST QUERIES
# -----------------------------
test_queries = [
    "What are OSFI capital requirements related to credit risk?",
    "What are the OSFI operational risk guidelines?",
    "Explain the Capital Conservation Buffer requirements.",
    "List Tier 1 and CET1 capital ratios.",
    "What are countercyclical buffer rules?"
]

# -----------------------------
# OUTPUT CSV
# -----------------------------
output_file = "evaluation_log.csv"

with open(output_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Query", "Answer", "Sources", "Flag_Hallucination", "Flag_Redundancy"])

    for query in test_queries:
        timestamp = datetime.now().isoformat()
        result = qa_chain.invoke({"query": query})

        answer_text = result["result"]
        sources = [f"{doc.metadata.get('source')} | page {doc.metadata.get('page')}" 
                   for doc in result["source_documents"]]

        # Hallucination check: answer must cite sources
        flag_hallucination = "YES" if len(sources) == 0 else "NO"

        # Redundancy check: repeated numbers or bullet points
        answer_lower = answer_text.lower()
        flag_redundancy = "YES" if len(answer_lower.split("\n")) != len(set(answer_lower.split("\n"))) else "NO"

        writer.writerow([timestamp, query, answer_text, "; ".join(sources), flag_hallucination, flag_redundancy])

        print(f"\nQuery: {query}")
        print(f"Answer: {answer_text}")
        print(f"Sources: {sources}")
        print(f"Hallucination: {flag_hallucination}")
        print(f"Redundancy: {flag_redundancy}")

print(f"\nEvaluation completed. Logs saved to {output_file}")
