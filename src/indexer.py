"""Build the index from the raw corpus, save it, load it and search it."""
import os
import pickle

from tqdm import tqdm

from src.bm25 import BM25, tokenize
from src.chunker import chunk_text
from src.models import MinimalSource

# Sources are only ever .md, .txt or .py files.
EXTENSIONS = (".py", ".md", ".txt")
INDEX_FILE = "index.pkl"

Chunk = tuple[str, int, int]


def find_files(raw_dir: str) -> list[str]:
    """List every indexable file, sorted so indexing is reproducible."""
    paths = []
    for folder, _, names in os.walk(raw_dir):
        for name in names:
            if name.endswith(EXTENSIONS):
                paths.append(os.path.join(folder, name))
    return sorted(paths)


def read_file(path: str) -> str:
    """Read a corpus file the way character indexes are counted on it."""
    # newline="" keeps "\r\n" as two characters; the default would turn it
    # into "\n" and shift every index after it.
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        return f.read()


class Index:
    """The chunk positions plus the BM25 index over their words."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.bm25 = BM25()

    def build(self, raw_dir: str, max_size: int) -> None:
        """Chunk every file of raw_dir and index each chunk."""
        files = find_files(raw_dir)
        for path in tqdm(files, desc="Indexing"):
            text = read_file(path)
            # The path's words ("lora", "quantization") describe the chunk
            # as much as its text does, so every chunk gets them too.
            path_tokens = tokenize(path.replace(raw_dir, ""))
            for start, end in chunk_text(text, path.endswith(".py"),
                                         max_size):
                self.chunks.append((path, start, end))
                self.bm25.add(path_tokens + tokenize(text[start:end]))

    def save(self, directory: str) -> None:
        """Write the index to directory/index.pkl."""
        os.makedirs(directory, exist_ok=True)
        # pickle turns the whole Index object (chunks, BM25 postings,
        # lengths) into bytes and writes them in one call. No manual
        # conversion is needed, and tuples stay tuples.
        with open(os.path.join(directory, INDEX_FILE), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(directory: str) -> "Index":
        """Read the index written by save()."""
        path = os.path.join(directory, INDEX_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no index at {path}, run the `index` command first")
        # pickle.load rebuilds the exact Index object that was saved. It can
        # run code stored in the file, so only load an index we wrote.
        with open(path, "rb") as f:
            index: Index = pickle.load(f)
        return index

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the k best chunks for the query as sources."""
        sources = []
        for chunk_id in self.bm25.search(query, k):
            path, start, end = self.chunks[chunk_id]
            sources.append(MinimalSource(
                file_path=path, first_character_index=start,
                last_character_index=end))
        return sources
