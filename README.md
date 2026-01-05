# Enterprise-RAG-OSFI
Enterprise-Grade Retrieval-Augmented Generation (RAG) System aligned with OSFI regulatory requirements for Canadian Financial Institutions.

## Business Problem
Canadian banks and credit unions must interpret and comply with OSFI guidelines across credit risk, operational risk, cybersecurity, and capital requirements.
Manual interpretation is slow, error-prone, and difficult to audit.

## Solution Overview
This project implements an **Enterprise RAG System** that:
- Ingests OSFI regulatory PDFs
- Chunks and embeds documents
- Enables governed Q&A with guardrails
- Logs queries for audit and compliance review

## Architecture
![rag-query-architecture](https://github.com/user-attachments/assets/1ba28134-83d5-43f5-9f5f-5b1fade2ea89)

## Key Features
- Document ingestion & chunking (LangChain)
- Vector search–based retrieval
- LLM-based answer generation
- Guardrails for hallucination prevention
- Query logging for compliance & audit
- Streamlit UI for business users

## Evaluation
- Retrieval accuracy testing
- Hallucination checks
- Response consistency evaluation

## Tech Stack
- Python
- LangChain
- FAISS / Chroma
- OpenAI / Azure OpenAI
- Streamlit
- Pandas

## Enterprise & OSFI Alignment
- Audit-friendly logging
- Explainable responses
- Data privacy by design
- Suitable for internal risk & compliance teams

## Project Structure
```text
### enterprise-rag-osfi/
│
├─ data/                         # Source PDFs
│   ├─ credit_risk.pdf
│   └─ operational_risk.pdf
│   └─ capital_requirements.pdf
│
├─ document_parsing_chunking/                             
│   └─ rag_loader.py             # Prepares OSFI regulatory documents for retrieval
│
├─ vectorstore/                  # FAISS vectorstore
│   └─ faiss_index/
│       └─ index.faiss           # Precomputed embeddings
│       └─ index.pkl             # Vectorstore Metadata & State
│
├─ qa/                           # QA scripts
│   ├─ qa_rag.py                 # Implements the core RAG pipeline
│   ├─ qa_rag_guardrails.py      # Main QA with guardrails
│   ├─ qa_rag_evaluation.py      # Evaluation of QA outputs
│   └─ streamlit_app.py          # Interactive Streamlit UI
│
├─ requirements.txt             # Python dependencies
├─ README.md                    # Project explanation & setup
└─ .gitignore
```

## 1. Clone the repository

bash 
git clone<your_repo_url> 
cd Enterprise-RAG-OSFI

## 2. Create and activate a virtual environment

**Windows**
python -m venv venv
venv\Scripts\activate

**Mac/Linux**
source venv/bin/activate

## 3. Install dependencies

pip install -r requirements.txt

## 4. Set your OpenAI API key

#### Windows
setx OPENAI_API_KEY "your_api_key_here"

#### Mac/Linux
export OPENAI_API_KEY="your_api_key_here"

## How to Run

1. CLI QA

python qa/qa_rag_guardrails.py

2. Streamlit UI

streamlit run qa/streamlit_app.py
![Streamlit RAG UI](https://github.com/user-attachments/assets/c4884e34-3afd-479b-839b-eb3fdbab01e1)

* Enter your query in the input box.
* Click **"Get Answer"** to see fact-based responses.
* Sources for each answer are displayed below the response.

3. Evaluation Logs

python qa/qa_rag_evaluation.py
Generates `evaluation_log.csv` with queries, answers, sources, and hallucination checks.

## **Key Features**

* **Non-hallucinated answers** : Responds “I don’t know” if context is missing.
* **Redundancy removal** : Returns each fact or number only once.
* **Enterprise-ready prompts** : Matches compliance and regulatory standards.
* **Interactive UI** : Streamlit interface for business users.
* **Evaluation logging** : CSV output with sources and hallucination flags.

## **Example Queries & Outputs**

| Query                                                 | Sample Answer                                                            | Sources                          |
| ----------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------- |
| What are CET1 capital requirements?                   | Minimum regulatory capital requirements: 4.5% CET1, 6% Tier 1, 8% Total. | credit_risk.pdf page 11, page 78 |
| Explain the Capital Conservation Buffer requirements. | Banks must maintain 2.5% CCB above minimum capital.                      | credit_risk.pdf page 11          |

*More examples are available in the Streamlit UI.*

## **Dependencies**

* `Python 3.11+`
* `langchain-openai`
* `langchain-classic`
* `langchain-community`
* `streamlit`
* `faiss-cpu` or `faiss-gpu`
* `python-dotenv`
* `numpy`
* `packaging`

## Use Cases
- Risk Analysts querying OSFI guidelines
- Compliance teams validating interpretations
- Internal audit support

## **Author**

**Kirti Sinha** – Agile Delivery & Data/AI Professional

* Focused on **Enterprise AI, RAG, and Business Analysis**
* [LinkedIn](https://www.linkedin.com/in/kirtisinha11/ "Let's connect")


