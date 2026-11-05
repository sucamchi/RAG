"""Tokenizer and BM25 retrieval.

BM25 scores and ranks chunks for a query following this formula:

    idf(word) * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / avg_length))

tf is how often the word appears in the chunk, idf rewards rare words, and
avg_length stops long chunks from winning by having more words.
"""
import math

# Standard BM25 constants.
# K1 limits how much repeating a word keeps helping.
# B sets how strongly long chunks are penalised.
K1 = 1.5
B = 0.75

# Stop words to ignore in queries.
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "of", "on", "or", "that",
    "the", "this", "to", "what", "when", "where", "which", "with", "vllm",
}


def snake_camel_split(word: str) -> list[str]:
    """Split snake_case and camelCase into parts."""
    parts: list[str] = []
    current = ""
    for i, char in enumerate(word):
        if char == "_":
            parts.append(current)
            current = ""
            continue
        # Upper case after lower case = new part (camelCase).
        if char.isupper() and i > 0 and word[i - 1].islower():
            parts.append(current)
            current = ""
        current += char
    parts.append(current)
    # remove empty parts
    return [part for part in parts if part]


def tokenize(text: str) -> list[str]:
    """Turn text into lowercase words, keeping identifiers and their parts."""
    words: list[str] = []
    current = ""
    for char in text + " ":
        if char.isalnum() or char == "_":
            current += char
            continue
        if current:
            words.append(current)
            current = ""
    tokens: list[str] = []
    for word in words:
        parts = snake_camel_split(word)
        # Keep the whole identifier too, so an exact name still matches.
        if len(parts) > 1:
            tokens.append(word.lower())
        for part in parts:
            tokens.append(part.lower())
    return [token for token in tokens if token not in STOP_WORDS]


class BM25:
    """An inverted index: word -> list of (chunk number, count in chunk)."""

    def __init__(self) -> None:
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lengths: list[int] = []

    def add(self, tokens: list[str]) -> None:
        """Add the next chunk; chunks are numbered in the order added."""
        chunk_id = len(self.lengths)
        self.lengths.append(len(tokens))
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            self.postings.setdefault(token, []).append((chunk_id, count))

    def search(self, query: str, k: int) -> list[int]:
        """Return the k best chunks for the query."""
        k = int(k)
        # If Fire passes --k as a string; int(k) raises ValueError.
        if not self.lengths:
            return []
        total = len(self.lengths)
        avg_length = sum(self.lengths) / total
        scores: dict[int, float] = {}
        for token in set(tokenize(query)):
            postings = self.postings.get(token, [])
            if not postings:
                continue
            # df (document frequency): number of chunks containing the word.
            df = len(postings)
            # idf (inverse document frequency): rare words weigh more.
            # The +1 inside the log keeps idf positive even for words that
            # appear in more than half of the chunks.
            idf = math.log((total - df + 0.5) / (df + 0.5) + 1)
            # tf (term frequency): times the word appears in this chunk.
            for chunk_id, tf in postings:
                # norm (length normalization): chunks longer than average
                # are penalized (norm > 1), shorter ones favored. B sets
                # how strong this is.
                norm = 1 - B + B * self.lengths[chunk_id] / avg_length
                # K1 makes tf saturate: repeating a word has diminishing
                # returns.
                score = idf * tf * (K1 + 1) / (tf + K1 * norm)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score
        ranked = sorted(scores, key=lambda c: scores[c], reverse=True)
        return ranked[:k]
