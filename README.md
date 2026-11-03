*This project has been created as part of the 42 curriculum by scamlett.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation system that answers questions about
the vLLM codebase. It cuts the vLLM repository into chunks, indexes them
with BM25, retrieves the chunks most relevant to a question, and passes
them to the small local model Qwen3-0.6B to generate an answer.
Retrieval quality is measured with recall@k.

In practice, RAG has four stages:
- Indexing: organize chunks for retrieval
- Retrieval: get relevant chunks for a question
- Augmentation: add retrieved chunks to context
- Generation: produce answer with Qwen3-0.6B

In this project, we hand-craft the RAG pipeline for a pre-provided vLLM repo.

## Instructions

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
make install        # uv sync
```

Place the data (not committed, see `.gitignore`):

```
data/raw/vllm-0.10.1/                    the vLLM repository to index
data/datasets/AnsweredQuestions/         datasets with reference sources
data/datasets/UnansweredQuestions/       datasets with questions only
```

### Commands

## Example usage

## System architecture

## Chunking strategy

Each file is first split into pieces at natural boundaries, then
consecutive pieces are packed together while they fit in `max_chunk_size`
(2000 characters as defined in subject).

- **Python**: a piece starts at every `def`, `async def`, `class` or
  decorator line, at any indentation, so methods are separate pieces. A
  `def` directly under its decorator stays with the decorator.
- **Markdown and text**: a piece starts at every heading (`#` at the start
  of a line), except inside ```` ``` ```` code blocks where `#` is a
  comment.

## Retrieval method

BM25 (Okapi), implemented in `src/bm25.py` with standard parameters.

Tokenization:

- lowercase words made of letters, digits and `_`;
- identifiers are split on `_` and camelCase, and the whole identifier is kept too, so
  both paraphrased and exact-name questions match;
- a short stop word list, including `vllm`;
- the words of each chunk's file path are added to the chunk.

Ranking: each chunk's score is the sum of the BM25 scores of the query
words it contains; the top k are returned.

### How BM25 scores a chunk

BM25 answers one question: *how well does this chunk match the words in the
question?* It combines three simple ideas.

1. **Rare words matter more (idf).** A word like `self` is in almost every
   chunk, so matching it says nothing. A word like `lora_rank` is in only a
   few chunks, so matching it is a strong signal. For a word found in `df`
   chunks out of `N`:

   ```
   idf = log((N - df + 0.5) / (df + 0.5) + 1)
   ```

   `df` is the *document frequency*: how many chunks contain the word.
   `idf` is large when `df` is small.

2. **More occurrences help, but with diminishing returns (tf).** `tf` is the
   *term frequency*: how many times the word appears in this chunk. A chunk
   mentioning "scheduler" 10 times is more relevant than one mentioning it
   once, but not 10 times more. The parameter `K1` (1.5 here) controls how
   quickly the gain flattens out.

3. **Long chunks are penalized (length normalization).** A long chunk
   contains more words by chance, so it is compared with the average chunk
   length. The parameter `B` (0.75 here) sets how strong this penalty is:
   0 means ignore length, 1 means full correction.

Putting them together, for one query word in one chunk:

```
norm  = 1 - B + B * (chunk_length / average_length)
score = idf * tf * (K1 + 1) / (tf + K1 * norm)
```

The chunk's final score is the sum of this value over every query word, and
the chunks with the highest scores are returned.

### Example

Question: `scheduler lora`. Chunk A mentions `scheduler` 3 times and `lora`
never. Chunk B mentions each once. Chunk A gets a high score for
`scheduler` only; chunk B gets two smaller scores that add up. Whichever
total is higher ranks first, and the rarer word (higher `idf`) pulls harder.


## Performance analysis

## Design decisions

## Challenges faced

## Resources

- en.subject.pdf provided by 42
- [Qwen3 model](https://huggingface.co/Qwen/Qwen3-0.6B).
- [Hugging Face transformers: text generation](https://huggingface.co/docs/transformers/main_classes/text_generation).
- [Python Fire CLI](https://github.com/google/python-fire/blob/master/docs/guide.md).
- [tqdm](https://github.com/tqdm/tqdm)
- [BM25 Okapi](https://medium.com/@readwith_emma/understanding-okapi-bm25-document-ranking-algorithm-70d81adab001)
- [BM25 for developers](https://medium.com/@kimdoil1211/bm25-for-developers-a-guide-to-smarter-keyword-search-e6d83e8c8c8c)

AI was used in a responsible manner to help understand new concepts, draft a step-by-step plan, assist in BM25 algorithm, testing and edge case handling.