
import csv
import json
import ollama

from pathlib import Path
from schemas import ProjectReport, ModelPerformance, RAGReference


# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

METRICS_FILE = BASE_DIR / "outputs" / "reports" / "model_comparison.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load actual model metrics
model_performance = []

with open(METRICS_FILE, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        model_performance.append(
            ModelPerformance(
                model_name=row["Model"],
                accuracy=float(row["Accuracy"]),
                precision=float(row["Precision"]),
                recall=float(row["Recall"]),
                f1_score=float(row["F1 Score"])
            )
        )


# 3. Define tested RAG references
rag_references = [
    RAGReference(
        question="What are the transport conditions for mRNA products?",
        retrieved_information="Validated cold-chain shipping containers and continuous temperature logging are required during transport.",
        source="mrna_safety_guidelines.txt"
    ),
    RAGReference(
        question="What are the refrigerated storage requirements?",
        retrieved_information="The project reference describes refrigerated storage between 2C and 8C for short-term use.",
        source="mrna_safety_guidelines.txt"
    ),
    RAGReference(
        question="What does the reference document say about age-banded dosing?",
        retrieved_information="The reference describes four age groups: 6 months to 4 years, 5 to 11 years, 12 to 64 years, and 65 years and older.",
        source="mrna_safety_guidelines.txt"
    ),
    RAGReference(
        question="What are the adverse event reporting and escalation criteria?",
        retrieved_information="The reference describes reporting thresholds and escalation criteria for medically significant adverse events.",
        source="mrna_safety_guidelines.txt"
    )
]


# 4. Prepare LLM prompt
metrics_text = json.dumps(
    [item.model_dump() for item in model_performance],
    indent=2
)

rag_text = json.dumps(
    [item.model_dump() for item in rag_references],
    indent=2
)

prompt = f"""
Create a concise educational project summary using only the supplied data.

Project: Moderna mRNA Vaccine Clinical Safety & Dosage Intelligence Platform.

Model performance:
{metrics_text}

RAG test references:
{rag_text}

Requirements:
- Explain the model comparison accurately.
- Summarize the RAG implementation and four tests.
- Do not invent results.
- Do not make clinical recommendations.
- State that this is an educational prototype.
- Do not present the project reference as official clinical guidance.
"""


# 5. Generate report using Llama
print("Generating Week 4 project report using Llama 3.2...")

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    options={"temperature": 0.1}
)

summary = response["message"]["content"]


# 6. Validate structured report with Pydantic
report = ProjectReport(
    project_title="Moderna mRNA Vaccine Clinical Safety & Dosage Intelligence Platform",
    project_type="Educational AI/ML Research Prototype",
    model_performance=model_performance,
    rag_references=rag_references,
    summary=summary,
    limitations=[
        "The dataset and model results do not establish clinical validity.",
        "The RAG reference is educational project material, not official clinical guidance.",
        "LLM-generated answers may contain omissions or errors.",
        "The system must not be used for real patient dosage or treatment decisions."
    ]
)


# 7. Save structured JSON report
output_file = OUTPUT_DIR / "week4_project_report.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(
        report.model_dump(),
        file,
        indent=4,
        ensure_ascii=False
    )


print("\nREPORT GENERATED SUCCESSFULLY!")
print("Pydantic validation: PASSED")
print("Models included:", len(report.model_performance))
print("RAG test cases:", len(report.rag_references))
print("Output:", output_file)
