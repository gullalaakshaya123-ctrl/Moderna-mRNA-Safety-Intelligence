
from pydantic import BaseModel, Field
from typing import List


class ModelPerformance(BaseModel):
    model_name: str
    accuracy: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1_score: float = Field(ge=0, le=1)


class RAGReference(BaseModel):
    question: str
    retrieved_information: str
    source: str


class ProjectReport(BaseModel):
    project_title: str
    project_type: str
    model_performance: List[ModelPerformance]
    rag_references: List[RAGReference]
    summary: str
    limitations: List[str]

    educational_prototype: bool = True


if __name__ == "__main__":

    print("PROJECT REPORT SCHEMA")
    print("---------------------")

    print("Pydantic schema created successfully!")

    print("\nAvailable sections:")
    print("- Project title")
    print("- Model performance")
    print("- RAG references")
    print("- Summary")
    print("- Limitations")
