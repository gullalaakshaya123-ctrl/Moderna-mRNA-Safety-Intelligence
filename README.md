
# 🧬 Moderna mRNA Vaccine Safety Intelligence Platform

An AI/ML research prototype combining Machine Learning, Retrieval-Augmented Generation (RAG), and Large Language Models to explore vaccine safety-related information from a supplied project dataset and reference document.

## 📌 Project Overview

This project demonstrates how machine learning and language models can be combined to analyze a sample patient-record dataset, explore sequence attention, and retrieve information from a supplied safety-guidelines document.

The application includes:
- Patient data preprocessing and exploratory data analysis.
- Machine learning classification using Logistic Regression and Random Forest.
- An illustrative mRNA sequence self-attention implementation.
- Document retrieval using Sentence Transformers and ChromaDB.
- Question answering using Llama 3.2.
- Structured report generation using Pydantic.
- An interactive Streamlit dashboard.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas, NumPy | Data processing |
| Matplotlib, Seaborn | Data visualization |
| Scikit-learn | Machine learning |
| Sentence Transformers | Text embeddings |
| ChromaDB | Vector database |
| Ollama | Local LLM execution |
| Llama 3.2 | Language model |
| Pydantic | Data validation |
| Streamlit | Interactive dashboard |
| Jupyter Notebook | Exploratory analysis |

## 📂 Project Structure

```text
moderna-mrna/
│
├── app.py
├── README.md
│
├── data/
│   ├── patient_records (1).csv
│   ├── patient_records_clean.csv
│   ├── processed_patient_records.csv
│   └── mrna_safety_guidelines.txt
│
├── notebooks/
│   └── 01_Exploratory_Data_Analysis.ipynb
│
├── src/
│   ├── inspect_data.py
│   ├── clean_data.py
│   ├── train_model.py
│   ├── visualize_results.py
│   ├── feature_analysis.py
│   ├── self_attention.py
│   ├── attention_analysis.py
│   ├── sequence_attention_report.py
│   ├── prepare_guidelines.py
│   ├── build_vector_db.py
│   ├── test_retrieval.py
│   ├── rag_qa.py
│   ├── schemas.py
│   ├── generate_report.py
│   └── benchmark_latency.py
│
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── risk_model.joblib
│
├── reports/
├── models/
└── chroma_db/
```

## 📊 Machine Learning Results

Two classification models were trained and evaluated using a stratified 80/20 train-test split.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 71% | 66% |
| Precision | 70.83% | 67.44% |
| Recall | 69.39% | 59.18% |
| F1 Score | 70.10% | 63.04% |
| ROC AUC | 0.7911 | 0.7587 |

**Best performing model:** Logistic Regression based on F1 Score.

## 🧠 mRNA Self-Attention

An illustrative scaled dot-product self-attention implementation was developed using a sample nucleotide sequence.

- Sample sequence: ACGUACGUACGGCUA
- Sequence length: 15
- Embedding dimension: 8
- Attention matrix: 15 × 15
- Attention visualization saved as a heatmap.

Note: This is a demonstration implementation using initialized random embeddings and weights, not a trained biological model.

## 🔎 Retrieval-Augmented Generation (RAG)

The project uses:
- Sentence Transformers for generating text embeddings.
- ChromaDB for storing and retrieving document chunks.
- Llama 3.2 through Ollama for answer generation.

The supplied project reference document is converted into text, divided into chunks, and indexed in the vector database.

The RAG pipeline retrieves relevant passages and uses them as context for answering questions.

## ⏱️ Latency Benchmark Results

Average timings from three test questions:

| Component | Average Time |
|---|---:|
| RAG Retrieval | 0.1676 seconds |
| LLM Generation | 23.8066 seconds |
| Total | 23.9742 seconds |

These timings are specific to the local execution environment and tested questions.

## 📈 Streamlit Dashboard

The interactive dashboard displays:

1. Model performance comparison.
2. ROC curve and AUC scores.
3. mRNA self-attention heatmap.
4. RAG vs LLM latency comparison.
5. Project technology information.

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd moderna-mrna
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment (Windows)

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter python-docx sentence-transformers chromadb tiktoken pydantic ollama streamlit joblib
```

### 5. Start Ollama

Install Ollama and download the model:

```bash
ollama pull llama3.2:3b
```

Ensure Ollama is running before executing the LLM-related scripts.

### 6. Launch the dashboard

```bash
streamlit run app.py
```

## ⚠️ Limitations and Disclaimer

- This is an educational and research prototype.
- Model results are based on the supplied project dataset and a single test split.
- Results do not establish clinical effectiveness or safety.
- The self-attention implementation is illustrative and not biologically validated.
- The supplied reference document is not a substitute for official clinical or regulatory guidance.
- This project must not be used for real patient assessment, vaccine dosage selection, or clinical decisions.

## 🎯 Project Objective

To demonstrate the integration of machine learning, natural language processing, retrieval-augmented generation, and interactive visualization in an academic AI/ML research project.

---

**Project Category:** Artificial Intelligence & Machine Learning  
**Dashboard Framework:** Streamlit  
**Year:** 2026
