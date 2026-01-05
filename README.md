# Enterprise RAG Assistant – OSFI Credit & Operational Risk

Secure Retrieval-Augmented Generation (RAG) with Guardrails for Financial Regulatory Documents

---

## **Project Overview**

This project demonstrates a **Retrieval-Augmented Generation (RAG)** system designed to answer enterprise questions related to **OSFI capital requirements** and **operational risk guidelines**. The system:

- Uses **FAISS vectorstore** to store and retrieve document embeddings.
- Incorporates **OpenAI LLMs** (`gpt-4o-mini` / `gpt-3.5-turbo-instruct`) for generative responses.
- Applies **enterprise-grade guardrails** to reduce hallucinations, enforce factual accuracy, and eliminate redundancy.
- Provides both **CLI-based QA** and **Streamlit UI** for interactive querying.
- Logs evaluation results to ensure answers cite reliable sources.

---

## **Project Structure**

### Enterprise-RAG-OSFI/

│

├─ vectorstore/                  # FAISS vectorstore

│   └─ faiss_index/

│       └─ index.faiss           # Precomputed embeddings

│

├─ qa/                           # QA scripts

│   ├─ qa_rag_guardrails.py      # Main QA with guardrails

│   ├─ qa_rag_evaluation.py      # Evaluation of QA outputs

│   └─ streamlit_app.py     # Interactive Streamlit UI

│

├─ data/                             # Source PDFs

│   ├─ credit_risk.pdf

│   └─ operational_risk.pdf

│

├─ requirements.txt             # Python dependencies

├─ README.md                  # Project explanation & setup

└─ .gitignore

## **Setup Instructions**

---

1. **Clone the repository**

```bash
git clone <your_repo_url>
cd Enterprise-RAG-OSFI
---
## 2. Create and activate a virtual environment

python -m venv venv

**Windows**

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

1️⃣ CLI QA

python qa/qa_rag_guardrails.py

2️⃣ Streamlit UI

streamlit run qa/streamlit_app.py


* Enter your query in the input box.
* Click **"Get Answer"** to see fact-based responses.
* Sources for each answer are displayed below the response.

3️⃣ Evaluation Logs

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

* Python 3.11+
* `langchain-openai`
* `langchain-classic`
* `langchain-community`
* `streamlit`
* `faiss-cpu` or `faiss-gpu`
* `python-dotenv`
* `numpy`
* `packaging`


## **Notes**

* FAISS vectorstore is **precomputed** and loaded locally for fast retrieval.
* If the index is large, provide instructions to **recompute embeddings** from PDFs.
* Designed for  **Canadian banking regulatory documents** ; can be adapted for other enterprise datasets.


## **Author**

**Kirti Sinha** – Agile Delivery & Data/AI Professional

* Focused on **Enterprise AI, RAG, and Business Analysis**
* [LinkedIn](https://www.linkedin.com/in/kirtisinha11/ "Let's connect")
* [GitHub](https://github.com/kirtis111)
```
