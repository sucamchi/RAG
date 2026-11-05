"""CLI commands for uv run python -m src <command>"""
import os
import time
from typing import Any
from tqdm import tqdm

from src.indexer import Index, read_file
from src.models import (AnsweredQuestion, MinimalAnswer, MinimalSearchResults,
                        MinimalSource, RagDataset, StudentSearchResults,
                        StudentSearchResultsAndAnswer)

RAW = "data/raw"
PROCESSED = "data/processed"
SEARCH_OUT = "data/output/search_results"
ANSWER_OUT = "data/output/search_results_and_answer"
MAX_CHUNK = 2000
# Moulinette counts a source as found above this overlap score. (Chapter VII)
MIN_IOU = 0.05


def save_json(model: Any, directory: str, name: str) -> str:
    """Write a Pydantic model as JSON to directory/name and return the path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(model.model_dump_json(indent=2))
    return path


def print_sources(sources: list[MinimalSource]) -> None:
    """Print one source per line as path [first:last]."""
    for s in sources:
        print(f"{s.file_path} "
              f"[{s.first_character_index}:{s.last_character_index}]")


def read_context(sources: list[MinimalSource]) -> str:
    """Join the text of every source. Uses the same reader as indexing."""
    parts = []
    for source in sources:
        text = read_file(source.file_path)
        parts.append(text[source.first_character_index:
                          source.last_character_index])
    return "\n\n".join(parts)


def index(max_chunk_size: int = MAX_CHUNK, raw_dir: str = RAW,
          index_dir: str = PROCESSED) -> None:
    """Chunk every file under raw_dir and save the index to index_dir."""
    if not 0 < max_chunk_size <= MAX_CHUNK:
        raise ValueError(f"max_chunk_size must be between 1 and {MAX_CHUNK}")
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"corpus folder {raw_dir} does not exist")
    start = time.time()
    idx = Index()
    idx.build(raw_dir, max_chunk_size)
    idx.save(index_dir)
    print(f"Ingestion complete! Indexed {len(idx.chunks)} chunks under "
          f"{index_dir}/ in {time.time() - start:.1f}s")


def search(query: str, k: int = 10, index_dir: str = PROCESSED) -> None:
    """Print the top-k sources for one query."""
    sources = Index.load(index_dir).search(str(query), k)
    if not sources:
        print("No matching sources found.")
    print_sources(sources)


def search_dataset(dataset_path: str, k: int = 10,
                   save_directory: str = SEARCH_OUT,
                   index_dir: str = PROCESSED) -> None:
    """Search every question of a dataset and save StudentSearchResults."""
    i = Index.load(index_dir)
    with open(dataset_path, encoding="utf-8") as f:
        dataset = RagDataset.model_validate_json(f.read())
    results = []
    for q in tqdm(dataset.rag_questions, desc="Searching"):
        results.append(MinimalSearchResults(
            question_id=q.question_id, question=q.question,
            retrieved_sources=i.search(q.question, k)))
    output = StudentSearchResults(search_results=results, k=k)
    path = save_json(output, save_directory, os.path.basename(dataset_path))
    print(f"Saved student_search_results to {path}")


def answer(query: str, k: int = 10, index_dir: str = PROCESSED) -> None:
    """Retrieve sources for one query and print a generated answer."""
    from src.generator import Generator

    query = str(query)
    if not query.strip():
        raise ValueError("the query is empty")
    sources = Index.load(index_dir).search(query, k)
    print(Generator().answer(query, read_context(sources)))
    print("\nSources:")
    print_sources(sources)


def answer_dataset(student_search_results_path: str,
                   save_directory: str = ANSWER_OUT) -> None:
    """Answer every question of a search results file."""
    from src.generator import Generator

    with open(student_search_results_path, encoding="utf-8") as f:
        searched = StudentSearchResults.model_validate_json(f.read())
    print(f"Loaded {len(searched.search_results)} questions")
    generator = Generator()
    answers = []
    for r in tqdm(searched.search_results, desc="Answering"):
        answers.append(MinimalAnswer(
            question_id=r.question_id, question=r.question,
            retrieved_sources=r.retrieved_sources,
            answer=generator.answer(
                r.question, read_context(r.retrieved_sources))))
    output = StudentSearchResultsAndAnswer(search_results=answers,
                                           k=searched.k)
    name = os.path.basename(student_search_results_path)
    path = save_json(output, save_directory, name)
    print(f"Saved student_search_results_and_answer to {path}")


def iou(a: MinimalSource, b: MinimalSource) -> float:
    """Overlap of two character ranges divided by their combined extent."""
    if a.file_path != b.file_path:
        return 0.0
    overlap = (min(a.last_character_index, b.last_character_index)
               - max(a.first_character_index, b.first_character_index))
    union = (max(a.last_character_index, b.last_character_index)
             - min(a.first_character_index, b.first_character_index))
    if overlap <= 0:
        return 0.0
    return overlap / union


def recall(s: list[MinimalSource], exp: list[MinimalSource], k: int) -> float:
    """Share of the expected sources matched by one of the top-k found."""
    hits = 0
    for truth in exp:
        for source in s[:k]:
            if iou(source, truth) >= MIN_IOU:
                hits += 1
                break
    return hits / len(exp)


def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    """Print recall@1/3/5/10 of search results against a ground truth."""
    with open(student_search_results_path, encoding="utf-8") as f:
        searched = StudentSearchResults.model_validate_json(f.read())
    with open(dataset_path, encoding="utf-8") as f:
        dataset = RagDataset.model_validate_json(f.read())
    truth = {}
    for q in dataset.rag_questions:
        if isinstance(q, AnsweredQuestion) and q.sources:
            truth[q.question_id] = q.sources
    if not truth:
        raise ValueError(f"{dataset_path} has no questions with sources")
    found = {r.question_id: r.retrieved_sources
             for r in searched.search_results}
    print(f"Questions evaluated: {len(truth)}")
    for k in (1, 3, 5, 10):
        total = 0.0
        for question_id, expected in truth.items():
            total += recall(found.get(question_id, []), expected, k)
        print(f"Recall@{k}: {total / len(truth):.3f}")


COMMANDS = {
    "index": index,
    "search": search,
    "search_dataset": search_dataset,
    "answer": answer,
    "answer_dataset": answer_dataset,
    "evaluate": evaluate,
}
