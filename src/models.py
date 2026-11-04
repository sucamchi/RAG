"""RAG models defined by the subject"""

from pydantic import BaseModel, Field
import uuid
from typing import List


class MinimalSource(BaseModel):
    """A single source of information."""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without an answer."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with an answer."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A RAG dataset with questions."""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Minimal search results from retrieval."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The minimal answer."""
    answer: str


class StudentSearchResults(BaseModel):
    """My retrieved search results."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """My search results and answer."""
    search_results: List[MinimalAnswer]
    k: int
