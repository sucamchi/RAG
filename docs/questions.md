# Evaluation questions and answers

Preparation for the "RAG against the machine" defense. It covers the five
mandatory understanding questions, the extra questions the corrector asks
while inspecting the code, the live commands, the numbers to remember, and
the things that can go wrong on the day.

Read this once end to end. Then practise saying the five answers out loud
without looking, because that is the part where you must sound like you
understand your own code.

---

## 1. The defense flow

The corrector starts the retrieval pipeline in the background (~8 min on a
cluster, ~5 s here), then asks about the code while it runs.

| Step | What happens | Where the result comes from |
|---|---|---|
| Q1 | Setup: repo, `uv sync`, private datasets | `student/`, `data/datasets/private/` |
| Q2 | Launch `exam_retrieval.sh` in the background | terminal |
| Q3 | Code quality and the 8 Pydantic models | `src/models.py`, `make lint` |
| Q4 | Two chunking strategies and the chunk size question | `src/chunker.py`, `src/cli.py` |
| Q5 | Retrieval demo | `uv run python -m src search ...` |
| Q6 | Answer demo | `uv run python -m src answer ...` |
| Q7 | Docs Recall@5 >= 80% | `summary.log` |
| Q8 | Code Recall@5 >= 50% | `summary.log` |
| Q9 | Indexing <= 300s | `summary.log` |
| Q10 | 200 questions <= 90s | `summary.log` |
| Q11 | Answer quality, 2 of 3 satisfactory | `exam_answer.sh` |
| Q12 | The five understanding questions | you |
| Q13 | README sections | `README.md` |
| Q14 | Four edge cases | `exam_edge_cases.sh` |
| Q15 | Bonus, only if everything above passes | nothing implemented |

The local results are in `docs/eval.md`. All mandatory checks pass:

- Indexing: 2s (limit 300s)
- 200 questions: 2s (limit 90s)
- Docs Recall@5: 0.820 (threshold 0.80)
- Code Recall@5: 0.720 (threshold 0.50)
- Edge cases: 4/4
- Answers: 3/3
- Tests: 39 passed, `make lint` clean

---

## 2. The five mandatory understanding questions

### 2.1 What is RAG and why is it useful?

A language model only knows what was in its training data, and that
knowledge is frozen. Retrieval-Augmented Generation adds two steps before
the model writes an answer: search an external corpus for the passages most
likely to answer the question, then put those passages in the prompt as
context. The model answers from the retrieved text instead of from memory.

It is useful for three reasons. First, the corpus can be new, private or
huge, and you do not retrain the model to add it. Second, the answer is
grounded in a real source, so you can check it and cite it. Third, the
system can say "the context does not contain this" instead of inventing an
answer.

The limit is that retrieval quality caps answer quality. If the right chunk
is not in the top k, the model cannot answer well. In this project that is
why recall@k is the main metric, not the quality of the prose.

### 2.2 Walk me through your complete RAG pipeline, from raw documents to a generated answer.

There are two phases: indexing, which runs once, and querying, which runs
for every question.

Indexing:

1. `index` walks `data/raw/` and keeps `.py`, `.md` and `.txt` files.
2. Each file is read with `open(..., newline="")`, so line endings
   stay two characters and every character index matches the original file.
3. `src/chunker.py` splits the file into pieces at natural boundaries, cuts
   pieces longer than 2000 characters, and packs small consecutive pieces
   back together. Python and Markdown use different splitters.
4. Each chunk is tokenized, and so are the words of its file path. The
   tokens go into a BM25 inverted index: word -> list of (chunk number,
   count in chunk), plus the length of every chunk.
5. The `Index` object (chunk locations and the BM25 index) is pickled to
   `data/processed/index.pkl`. The text itself is not stored, only
   `(file_path, first_character_index, last_character_index)`.

Querying:

1. The question is tokenized the same way: lowercase, split snake_case and
   camelCase, keep the whole identifier too, drop stop words.
2. Every query word gets a BM25 score in every chunk that contains it. The
   scores are summed per chunk and sorted, and the top k chunks are returned
   as `MinimalSource` objects.
3. `read_context` reads each source file again and slices the character
   range, then joins the chunks into one context string.
4. `Generator` loads `Qwen/Qwen3-0.6B`, builds a chat prompt that tells the
   model to use only the context, trust the top-ranked passages, and include
   every relevant item, then adds the context plus question, and generates a
   greedy answer of at most 256 new tokens.
5. `search_dataset` and `answer_dataset` write the results as JSON through
   the Pydantic models, so the files match the subject format.

The important separation is that search never loads the model. `Generator`
is imported inside the `answer` functions, so indexing and retrieval stay
fast and need no GPU.

### 2.3 What is the difference between TF-IDF and BM25?

Both are lexical, bag-of-words methods. They score a document by the words it
shares with the query. Neither understands meaning or synonyms.

TF-IDF multiplies term frequency (how often the word appears in the
document) by inverse document frequency (how rare the word is across the
corpus). A word that appears 50 times scores about 50 times higher than a
word that appears once. Long documents also win simply because they contain
more words.

BM25 (Okapi) keeps the same idf idea but fixes those two problems:

- Term frequency saturates. Going from 1 to 2 occurrences helps a lot;
  going from 20 to 50 helps very little. The parameter `K1` controls how
  fast the gain flattens. We use `K1 = 1.5`.
- Length is normalized against the average chunk length. Long chunks are
  penalized a little. The parameter `B` controls how strong that is. We use
  `B = 0.75`.

The formula for one query word in one chunk is:

```
norm  = 1 - B + B * chunk_length / average_length
score = idf * tf * (K1 + 1) / (tf + K1 * norm)
```

The chunk score is the sum over the query words. I chose BM25 because our
chunks have very different lengths (one function, or one markdown section),
and BM25 handles that better than plain TF-IDF.

### 2.4 What implementation choices did you make and why? What trade-offs did you consider?

The big ones:

- **BM25 instead of embeddings.** BM25 is fast on CPU, needs no model, and
  matches exact identifiers like `max_num_seqs`. The trade-off is that it
  misses paraphrases and synonyms. On this corpus that is acceptable: docs
  questions often reuse the words of the page, and code questions quote
  function names.
- **Syntax-aware chunking instead of fixed windows.** Python starts a new
  piece at `def`, `class`, `async def` and decorators, so a chunk usually
  follows a function. Markdown starts at headings, ignoring `#` inside code
  fences. The trade-off is that a function longer than 2000 characters is
  still cut by the size limit.
- **2000 character chunks.** The moulinette rejects any source longer than
  2000, and the subject sets 2000 as the default. Ten chunks of 2000 are
  about 20,000 characters, roughly 5,000 tokens, well inside Qwen3's 32k
  context. Smaller chunks lose context and can fail the IoU match; larger
  chunks are invalid and dilute the BM25 score.
- **File path words added to every chunk.** A question that mentions
  "lora" or "quantization" should find the right module even if the chunk
  body uses different words. The trade-off is a little noise from generic
  path parts.
- **Identifier splitting.** `max_num_seqs` is indexed as the whole name plus
  `max`, `num`, `seqs`. A question can quote the identifier exactly or use
  the words. The trade-off is more tokens and a slightly larger index.
- **Drop `vllm` as a stop word.** It appears in almost every question and
  document, so it does not tell chunks apart.
- **Pickle for the index.** It is two lines, standard library, fast, and
  keeps tuples as tuples. The trade-offs are that it is not human readable,
  it can break if the classes change, and unpickling untrusted data is
  unsafe. We only load an index we wrote ourselves.
- **Greedy decoding (`do_sample=False`).** Deterministic and fast. The
  trade-off is less varied wording. For a factual answer that is fine.
- **`enable_thinking=False`.** Qwen3 can spend tokens on a reasoning block.
  We disable it so the 256-token budget goes to the answer.
- **Explicit grounded generation prompt.** The system prompt tells the model
  to use only the retrieved context, trust the top-ranked passages, lead with
  the direct answer, name only commands and flags that appear in the context,
  include every relevant item, and say when the context does not answer. This
  turned the OpenAI demo from a client-only answer into one that names
  `vllm serve`, and made the API list complete. The trade-off is a slightly
  longer prompt and more verbose answers.
- **No chunk overlap.** Simpler and smaller index. The trade-off is that a
  function cut at the 2000 boundary can lose context at the edge. Adding a
  small overlap is one of the improvements I would try.

### 2.5 If you had more time, what would you improve?

In the order I would do it:

1. **Semantic embeddings plus hybrid retrieval.** Add a small CPU model such
   as `all-MiniLM-L6-v2` next to BM25, then merge the two rankings. This is
   bonus 1 and 2, and it fixes the paraphrase weakness.
2. **AST-aware Python chunking.** Use the `ast` module to cut whole
   functions with their signature and docstring, instead of matching `def`
   lines. It would avoid splitting decorators, nested functions and
   multi-line signatures by hand.
3. **Tune the ranking.** Try different `K1`, `B` and chunk sizes, and
   measure recall@5 on a validation split instead of guessing. Add
   stemming or query expansion for plurals and synonyms.
4. **Rerank the top 50 results.** A cross-encoder or embedding model on the
   shortlist would improve the order at low k, where code recall@1 is 0.44.
5. **Incremental indexing and caching.** Hash each file, reindex only what
   changed, and cache repeated query results. Bonuses 3 and 4.
6. **A local HTTP API.** A few FastAPI endpoints for search and answer.
   Bonus 5.
7. **Answer quality checks.** Read the generated answer against the
   retrieved context and flag answers that are not grounded. The moulinette
   does not score answers yet, so this would be my own test.
8. **Context budget management.** Count tokens before building the prompt
   and truncate or drop the lowest-ranked chunks if needed, instead of
   relying on the 10 x 2000 character bound.

---

## 3. Questions asked while inspecting the code (Q3 to Q6)

### 3.1 Code quality and models (Q3)

What to show:

```bash
make lint      # flake8 + mypy, the exact flags from the subject
uv run pytest  # 39 tests
```

Points to make:

- The eight models are in `src/models.py` and match the subject exactly:
  `MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `RagDataset`,
  `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`,
  `StudentSearchResultsAndAnswer`.
- They are used for validation on input (`RagDataset.model_validate_json`)
  and for output (`model_dump_json`). So a malformed dataset or a wrong
  output shape fails early.
- Service classes such as `Index`, `BM25` and `Generator` are plain Python,
  which the subject allows.
- `src/__main__.py` catches `OSError` and `ValueError` and prints
  `error: ...` instead of a traceback. `KeyboardInterrupt` is caught too.
- Files are opened with `with open(...)`, so handles close even on error.
- `tqdm` wraps the long loops: file indexing, dataset search, dataset
  answering.
- The Makefile has `install`, `run`, `debug`, `clean`, `lint`, plus `test`.
- No file imports or calls the moulinette. The grep the corrector runs
  returns nothing.

### 3.2 Chunking (Q4)

Show `src/chunker.py` and point at the three steps:

1. `split_python` and `split_markdown` produce pieces at natural boundaries.
2. `cut_long_piece` cuts a piece longer than `max_size`, preferring the last
   newline that fits, and cutting mid-line only when one line is longer than
   the limit.
3. `pack` merges consecutive pieces while they still fit.

Two strategies:

- **Python:** a new piece starts at any line whose stripped text begins with
  `def `, `async def `, `class ` or `@`. The `@` handling keeps a decorator
  attached to the function under it. Indentation is ignored, so methods are
  separate pieces.
- **Markdown/text:** a new piece starts at every line beginning with `#`,
  except inside a fenced code block, where `#` is a comment. The code block
  state flips on every line starting with three backticks.

The default is 2000. It is configurable with `--max_chunk_size`, and
`index` rejects values outside 1 to 2000, so a source can never exceed the
moulinette's `max_context_length`.

The required question: **what happens if chunks are too small or too
large?**

- Too small: a chunk loses the surrounding explanation. A retrieved snippet
  may be a single import line or half a function, so it is hard for the
  model to use and less likely to overlap the reference span. You also get
  many more chunks, which makes the index bigger and the context fragmented.
- Too large: a chunk covers several topics, so its BM25 score is diluted by
  words that are irrelevant to the question. Fewer chunks fit in the token
  budget, each chunk costs more context tokens, and above 2000 characters
  the moulinette rejects the source. That invalidates the whole output.

2000 is the compromise: big enough to hold a function or a doc section,
small enough to stay under the grader's limit and fit ten chunks in context.

### 3.3 Retrieval (Q5)

Live demo:

```bash
uv run python -m src search "How to configure OpenAI server?" --k 10
```

Explain the output line by line: `file_path [first:last]`. The path is
relative to the project root and is compared verbatim by the grader. The
range is a Python slice, `last` exclusive, at most 2000 characters.

How the ranking works:

1. Tokenize the query. `openai` and `server` stay, `how`, `to` and `the` are
   stop words.
2. For each token, find the chunks in the postings list. Compute idf from
   how many chunks contain it, then for each chunk compute tf and the length
   normalization. Add the score.
3. Sort all scored chunks descending, return the top k.

Why the top result is a docs page: `openai_compatible_server.md` contains
the words many times and its path also contains `openai` and `server`, so
it wins on both body and path. Some example client files also match, which
is why they appear lower in the list.

If asked "why not always perfect": BM25 counts words, so a chunk that
repeats the query words can beat the chunk that actually answers the
question. That is why the right source is often at rank 2 or 3, and why
recall@5 (0.82 docs, 0.72 code) is much higher than recall@1 (0.62 and
0.44).

### 3.4 Answer generation (Q6)

Live demo:

```bash
uv run python -m src answer "How to configure OpenAI server?" --k 10
```

What happens:

1. Retrieve the top 10 sources with the same BM25 search.
2. `read_context` opens each file and slices the range, joining the chunks
   with blank lines. The maximum is 10 x 2000 = 20,000 characters, about
   5,000 tokens, inside the model's context window.
3. `Generator` loads `Qwen/Qwen3-0.6B` from Hugging Face. On this laptop it
   uses CUDA; on the 42 cluster it falls back to CPU with float32.
4. The prompt is a system message plus one user message with the context and
   the question. The system message tells the model to answer only from the
   context, to treat the first passages as the most relevant, to start with
   the direct answer, to name only commands, flags and values that appear in
   the context, to include every relevant item, and to say so when the
   context does not contain the answer. The Qwen chat template is applied
   with `enable_thinking=False`.
5. Generation is greedy (`do_sample=False`), at most 256 new tokens, inside
   `torch.inference_mode()`. `model.eval()` is called once at load.
6. The prompt tokens are dropped from the output before decoding, and
   special tokens are skipped.

The default model is exactly `Qwen/Qwen3-0.6B`. There are no other model
options, so there is nothing else to demonstrate.

The prompt was tuned on this exact demo question. The earlier prompt
("Answer only from the context") produced a client-only answer. The current
prompt makes the model start with `vllm serve <model>` and then mention the
OpenAI SDK configuration. It is coherent and grounded.

If a corrector still gets a weak answer, explain the separation: retrieval
found the source (the [VALID] marker proves it), generation is limited by a
0.6B model. The project's target is retrieval quality and grounding, not
perfect prose.

---

## 4. Answer quality (Q11)

The corrector lists questions whose sources were retrieved, picks three, and
runs them. The three tested locally:

1. **"What OpenAI-compatible APIs does vLLM support for serving models?"**
   Answer: the full list with endpoints, Completions, Chat, Embeddings,
   Transcriptions, Translation, Tokenizer, Pooling, Classification, Score and
   Rerank. Complete and grounded.
2. **"What Python versions are supported for vLLM CPU installation?"**
   Answer: "3.9, 3.10, 3.11, and 3.12." Exact.
3. **"What is the recommended value for num-redundant-experts in large scale
   vLLM expert parallel deployments?"**
   Answer: "32", with the matching sentence from the docs quoted. Exact.

All three were judged satisfactory, so 3/3 against a 2/3 requirement.

If a corrector picks a different question and the answer is weak, explain
the separation: retrieval found the source (the [VALID] marker proves it),
generation is limited by a 0.6B model. The project's target is retrieval
quality and grounding, not perfect prose.

---

## 5. Edge cases (Q14)

All four pass with no traceback. Know what each one does:

| Test | What happens |
|---|---|
| `search "" --k 10` | `tokenize("")` returns `[]`, BM25 has no scored chunks, so it prints "No matching sources found." |
| `search "asdfghjkl ..."` | The tokens are not in any postings list, so the score dictionary is empty and the same message is printed. |
| `answer "What is vLLM?" --k 0` | BM25 returns `[]`, the context is empty, the model still loads and answers. It does not crash, but it does load Qwen, so expect ~30s+ on CPU. |
| `search_dataset --dataset_path /nonexistent/dataset.json` | `open` raises `FileNotFoundError`, which is an `OSError`, caught in `__main__`, printed as `error: ...` with exit code 1. |

Also handled, though not in the script: malformed JSON raises Pydantic's
`ValidationError`, which is a `ValueError`, so it also becomes a short error
message. A missing index raises `FileNotFoundError` with the message "run
the `index` command first". A non-numeric `--k` raises `ValueError` from
`int(k)` in `BM25.search`.

The one fix made during preparation: `search lora --k abc` used to fail with
a `TypeError` traceback from the slice. `BM25.search` now starts with
`k = int(k)`, so the bad value becomes a `ValueError` and the normal error
handler catches it. The repo test suite covers this case.

---

## 6. Numbers and definitions to remember

| Thing | Value |
|---|---|
| Chunk size default | 2000 characters |
| BM25 `K1` | 1.5 |
| BM25 `B` | 0.75 |
| Default `k` | 10 |
| IoU match threshold | 0.05 |
| Max new tokens | 256 |
| Model | `Qwen/Qwen3-0.6B` |
| Docs threshold | Recall@5 >= 0.80 |
| Code threshold | Recall@5 >= 0.50 |
| Indexing limit | 300s |
| Throughput limit | 200 questions in 90s |
| Private docs recall@5 | 0.820 |
| Private code recall@5 | 0.720 |
| Public docs recall@5 | 0.830 |
| Public code recall@5 | 0.808 |
| Chunks in the index | about 13,000 |
| Index size | about 9 MB |
| Tests | 39 |

Definitions:

- **idf:** inverse document frequency. A word in few chunks gets a high
  weight, because matching it is informative.
- **tf:** term frequency. How many times the word appears in this chunk.
- **df:** document frequency. How many chunks contain the word at all.
- **norm:** `1 - B + B * (chunk_length / average_length)`. Long chunks get
  `norm > 1`, which lowers their score.
- **Recall@k:** for each question, the share of its reference sources that
  appear in the top k results. One reference source counts as found when a
  result in the same file overlaps its character range with IoU >= 0.05.
- **IoU:** intersection over union of two character ranges. If they overlap
  by 5% of their combined length, it counts.
- **Augmentation:** adding retrieved text to the prompt. It is the "A" in
  RAG.
- **Grounded:** the answer uses the retrieved context, not the model's
  memory.

---

## 7. Commands to have ready

Run from the student repository root. The evaluator provides `data/raw/`
and the private datasets.

```bash
# install
uv sync

# quality
make lint

# build the index (2s here)
uv run python -m src index --max_chunk_size 2000

# single query
uv run python -m src search "How to configure OpenAI server?" --k 10

# single answer
uv run python -m src answer "How to configure OpenAI server?" --k 10

# batch search
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

# batch answers
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions

# own recall check
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

# official recall check
./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000
```

The official scripts expect this workspace, not the repo itself:

```
correction-root/
  student/                 clone of the repository
  exams/                   from exams.zip
  moulinette               the binary
  data/datasets/private/   from datasets_private.zip
```

`exam_retrieval.sh` resolves `PROJECT_DIR` as the parent of `exams/scripts`,
so the private datasets must be at `data/datasets/private/`. The student
repo keeps its own `data/raw/` and `data/processed/`.

---

## 8. Recode instructions

The sheet says a small modification may be requested. Be ready to do these
in a few minutes, then rerun `make lint` and `uv run pytest`.

Likely asks and where to edit:

- **Change the default `k`.** `src/cli.py`, the `k: int = 10` defaults, or
  the `COMMANDS` functions.
- **Change the default chunk size.** `MAX_CHUNK = 2000` in `src/cli.py`.
  Remember the `0 < max_chunk_size <= MAX_CHUNK` guard.
- **Print the BM25 score next to each source.** Add the score to
  `BM25.search`'s return or expose a `search_scored` method, then print it
  in `print_sources`. Keep the source format for the JSON unchanged.
- **Add a field to the output.** Add it to `MinimalSearchResults` in
  `src/models.py` and fill it where the model is constructed in `cli.py`.
  Pydantic will serialize it.
- **Skip a file extension.** Edit `EXTENSIONS` in `src/indexer.py`.
- **Change stop words.** Edit `STOP_WORDS` in `src/bm25.py`, then rebuild
  the index, because stop words affect the stored postings.
- **Add an output option.** Add a parameter to the CLI function; Fire exposes
  it automatically.

Rules while editing: keep type hints, keep the `with open` blocks, do not
break the model field names, and rerun the tests. If the change affects the
index, rerun `index`.

---

Answer these without looking. If you can, you are ready.

1. What are the four stages of RAG?
2. What does `last_character_index` mean, and why is it exclusive?
3. Why does the index store chunk locations instead of chunk text?
4. What do `K1` and `B` do in BM25?
5. Why does the tokenizer keep both `max_num_seqs` and its three parts?
6. Why is `newline=""` used when reading corpus files?
7. Why is the index pickled, and what are the two risks?
8. What happens when `k=0`, and why is there no crash?
9. Why is docs recall higher than code recall?
10. What would you change first if you had another day?

### Pickle in simple terms

Pickle is Python's way of saving an object so it can be loaded again later.
It turns the `Index` object into binary data and writes it to a file.
That object contains chunk locations and the BM25 search data.
So it does not contain a separate copy of the full text.
Later, `pickle.load` rebuilds the object so searching can start without
rebuilding the index. Writing and reading binary is fast and avoids parsing JSON.