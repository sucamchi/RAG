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

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
make install        # uv sync
```

Place the data:

```
data/raw/vllm-0.10.1/                    the vLLM repository to index
data/datasets/AnsweredQuestions/         datasets with reference sources
data/datasets/UnansweredQuestions/       datasets with questions only
```


### CLI Commands

| Command | What it does |
|---|---|
| `index --max_chunk_size 2000` | chunk `data/raw/` and save the index to `data/processed/` |
| `search "<query>" --k 10` | print the top-k sources for one query |
| `search_dataset --dataset_path <file> --k 10 --save_directory <dir>` | search every question, write `StudentSearchResults` JSON |
| `answer "<query>" --k 10` | retrieve and print a generated answer |
| `answer_dataset --student_search_results_path <file> --save_directory <dir>` | answer every question, write `StudentSearchResultsAndAnswer` JSON |
| `evaluate --student_search_results_path <file> --dataset_path <file>` | print recall@1/3/5/10 against a ground truth |

## Example usage

```bash
# Step 1: Chunk and index the repo
uv run python -m src index --max_chunk_size 2000

# Step 2: Search a query and print the top results
uv run python -m src search "How to configure the OpenAI server?" --k 5

# Step 3: Search dataset, write results to file
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions

# Step 4: Evaluate the recall@k with moulinette
./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000

# Step 5: Answer dataset questions and write results
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

## System architecture

Indexing runs once. Retrieval and generation run for every question.

```
INDEXING
data/raw/ --> chunker --> tokenizer --> BM25 index --> data/processed/index.pkl
  (files)     (spans)     (words)      (postings)

QUERYING
question --> tokenizer --> BM25 search --> top-k sources --> read text --> Qwen3-0.6B --> answer
                          (index.pkl)    (path, start, end)  (from files)
```

### Components

| File | Role |
|---|---|
| `src/chunker.py` | Cuts one file's text into `(start, end)` character ranges of at most `max_chunk_size`. Python and Markdown are split differently. |
| `src/bm25.py` | The tokenizer, and the BM25 inverted index that scores and ranks chunks for a query. |
| `src/indexer.py` | The `Index` class: walks the corpus, chunks each file, feeds each chunk's words (plus its file path words) to BM25, saves and loads the pickle, and turns ranked chunk numbers into sources. |
| `src/models.py` | Pydantic models for the datasets, sources, search results and answers, so inputs and outputs follow the subject's format. |
| `src/generator.py` | Loads Qwen3-0.6B and generates an answer from a question and a context string. |
| `src/cli.py` | The commands (`index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`), the recall@k calculation, and reading the text of retrieved sources. |
| `src/__main__.py` | Entry point: exposes the commands with Python Fire and prints errors as a short `error:` message. |

### How the pieces interact

1. **Indexing:** `index` calls `Index.build`. For each `.py`, `.md` and
   `.txt` file, the chunker returns character ranges and BM25 stores the
   words of each range. The index keeps only the location of each chunk
   (file path, start, end), not its text, and is saved with pickle.
2. **Retrieval:** `search` and `search_dataset` load the index and call
   `Index.search`. BM25 ranks the chunks and the index returns them as
   `MinimalSource` objects: a file path with a character range.
3. **Augmentation:** `answer` and `answer_dataset` use `read_context` to
   read the text of each retrieved range from its file and join them into
   one context string.
4. **Generation:** `Generator.answer` puts that context and the question in
   a prompt that tells the model to answer only from the context, and
   Qwen3-0.6B writes the answer.
5. **Output:** results are validated and saved with the Pydantic models:
   `StudentSearchResults` for search, and
   `StudentSearchResultsAndAnswer` for answers. `evaluate` compares saved
   search results with a ground-truth dataset and prints recall@k.

Retrieval works only with locations and never loads the model, so the
search commands stay fast. `torch` and `transformers` are imported only
when an `answer` command runs.

## Chunking strategy

Each file is first split into pieces at natural boundaries, then
consecutive pieces are packed together while they fit in `max_chunk_size`
(2000 characters as defined in subject). A piece longer than
`max_chunk_size` is cut just after a newline, or in the middle of a line if
a single line is longer than the limit, so no chunk is ever too big.

- **Python**: a piece starts at every `def`, `class` or
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

BM25 combines three simple ideas.

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

### Speed

Indexing the whole vLLM corpus (about 2000 files, 13,000 chunks) takes
around 3 seconds, far below the 5 minute limit. The index is a 9 MB pickle
file, and a single search takes a fraction of a second once it is loaded.

### Retrieval quality

Recall@k is the share of a question's reference sources for which one of
the top k retrieved chunks overlaps (IoU of at least 0.05). Measured with
`search_dataset` and `evaluate` on the public answered datasets
(`max_chunk_size` 2000):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required @5 |
|---|---|---|---|---|---|
| docs (100 questions) | 0.650 | 0.790 | 0.830 | 0.860 | 0.80 |
| code (99 questions) | 0.495 | 0.737 | 0.808 | 0.848 | 0.50 |

Both datasets pass the required recall@5.

What the numbers say:

- The right chunk is usually found but not always ranked first: code goes
  from 0.50 at k=1 to 0.81 at k=5. BM25 only counts words, so a chunk that
  repeats the question's words can outrank the one that answers it.
- Code probably does well because Python is cut at `def` and `class`, so a
  chunk is usually one function, and the file path words are added to every
  chunk.

### Answers

The answers come from Qwen3-0.6B, which reads the retrieved chunks. They
are short and usually quote the right identifier or command when the right
chunk is retrieved. When retrieval misses, the small model answers from the
wrong context, so answer quality follows retrieval quality.

## Design decisions

### Saving the index with pickle

Indexing is done once (`index`), and the other commands (`search`,
`answer`, ...) run later, so the index has to be saved to disk. It is
stored in `data/processed/index.pkl` with Python's `pickle` module.

How pickle works: it *serializes* a Python object, which means it walks
through the object (here the `Index`, with its chunk list and the BM25
postings and lengths) and writes it as a stream of bytes. Loading does the
reverse and rebuilds the same object in memory. In the code:

```python
pickle.dump(index, f)      # object -> bytes in the file
index = pickle.load(f)     # bytes in the file -> object
```

Benefits:

- **Short code:** two lines, with no manual JSON conversion.
- **Exact round trip:** types are kept, so a tuple comes back as a tuple.
- **Standard library:** no extra dependency.

Trade-offs:

- **Security:** unpickling can run code stored in the file, so a pickle from
  an untrusted source is dangerous. Here the file is produced by our own
  `index` command, so it is safe.
- **Fragile:** the file depends on the class definitions. If `Index` or
  `BM25` change, re-run `index` (it takes a few seconds).
- **Not readable** by humans, unlike JSON.

### Generator only loads when needed

The Generator is only imported and initialized when an `answer` command runs, keeping the other commands fast.

## Challenges faced

- **Decorators split from their function:** a `@decorator` line started a
  new piece, so it ended up away from its `def`. The splitter now keeps a
  decorator together with the definition under it.
- **Character indexes shifting:** reading files in text mode turns `\r\n`
  into `\n`, which moves every later index by one. Files are read with
  `newline=""` so the ranges match the original characters.
- **Edge cases:** BM25 and the CLI handle an empty index, empty queries,
  unknown words and missing files.
- **Understanding pickle:** learned how serialization allows reuse of the index.

## Resources

- [en.subject.pdf provided by 42](docs/en.subject.pdf)
- [uv](https://docs.astral.sh/uv/)
- [Writing your pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [Qwen3 model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Hugging Face transformers: text generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Python Fire CLI](https://github.com/google/python-fire/blob/master/docs/guide.md)
- [tqdm](https://github.com/tqdm/tqdm)
- [BM25 Okapi](https://medium.com/@readwith_emma/understanding-okapi-bm25-document-ranking-algorithm-70d81adab001)
- [BM25 for developers](https://medium.com/@kimdoil1211/bm25-for-developers-a-guide-to-smarter-keyword-search-e6d83e8c8c8c)
- [Python pickle](https://docs.python.org/3/library/pickle.html)

AI was used in a responsible manner to explain new concepts, draft a step-by-step plan, assist in BM25 algorithm, testing and edge case handling.