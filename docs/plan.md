# Plan: building RAG against the machine step by step

This plan walks through the project in the order the pieces depend on each
other. Each step ends with a check you can run, so you always know the step
works before moving on. File names match the finished code in `src/`.

The whole pipeline in one sentence: cut the vLLM files into chunks, build a
BM25 index over the chunks' words, find the best chunks for a question, and
hand them to Qwen3-0.6B to write the answer.

```
data/raw/ --chunker--> chunks --bm25--> data/processed/index.pkl
question --search--> top-k chunks --generator (Qwen)--> answer
```

---

## Step 0: Read the inputs before writing code

1. Read the whole subject. The pieces feed each other, and the output format
   (chapter VI.4 and VI.5) decides what every stage has to keep.
2. Open one dataset file in `docs/datasets_public/`. Notice:
   - every question has exactly **one** source;
   - a source is a `file_path` plus a character range `[first, last)`;
   - docs questions point at `.md` / `.txt` files, code questions at `.py`.
3. Read `docs/README.md` (the moulinette). The moulinette is how you are
   graded, so learn to run it early.

**Check:** you can say what "recall@5 = 0.8" means: for 80% of the
questions, one of your top 5 chunks overlaps the reference range.

## Step 1: Project skeleton

1. Lay out the folders the subject requires:
   ```
   src/__init__.py   src/__main__.py
   data/raw/vllm-0.10.1/          (copy of docs/vllm-0.10.1)
   data/datasets/AnsweredQuestions/   data/datasets/UnansweredQuestions/
   ```
   Add `data/` to `.gitignore`: the subject says no large data in the repo.
2. Write `pyproject.toml` with the dependencies (`fire`, `pydantic`, `tqdm`,
   `torch`, `transformers`) and a dev group (`flake8`, `mypy`, `pytest`).
   Run `uv sync`; it creates `uv.lock`, which you must commit.
3. Write the `Makefile` (install, run, debug, clean, lint, lint-strict).
4. Tell flake8, mypy and pytest to skip `data/` and `docs/`. They contain
   vLLM's own Python files and tests, which are not yours to check.

**Check:** `make install` and `make lint` both succeed on the empty project.

## Step 2: Data models (`src/models.py`)

Copy the pydantic models from subject section VI.4 exactly. They are the
contract with the moulinette, so field names must not change.

Two things worth understanding:
- `last_character_index` is **exclusive**: `text[first:last]` is the chunk.
- `RagDataset` holds `AnsweredQuestion | UnansweredQuestion`. Pydantic picks
  `AnsweredQuestion` when the JSON has `sources` and `answer`.

**Check:** in a Python shell, `RagDataset.model_validate_json(...)` loads
both dataset files without errors.

## Step 3: Chunking (`src/chunker.py`)

Goal: turn one file's text into a list of `(start, end)` spans, each at most
`max_size` characters (2000 by default, and never more: the moulinette
rejects longer sources).

Work in three small functions:

1. **Split into pieces** at natural boundaries. This is where the two
   required strategies differ:
   - Python: a new piece starts at a line beginning with `def`, `class`,
     `async def` or a decorator `@` (at any indentation, so methods count).
   - Markdown: a new piece starts at a heading line (`#`), but **not** inside
     a ```` ``` ```` code block, where `#` is a comment.
   Walk the text with `splitlines(keepends=True)` and keep a running
   character position: that is how you know where each piece starts.
2. **Cut pieces that are too long**, preferably just after a newline
   (`str.rfind("\n", ...)`), and mid-line only if one line is huge.
3. **Pack** consecutive pieces together while they still fit in `max_size`.
   Without this step you get thousands of tiny chunks (one per import line)
   that score badly.

Write the tests as you go (`tests/test_chunker.py`): pieces cover the whole
text with no gaps, no chunk exceeds `max_size`, and `#` inside a code block
does not start a section.

**Check:** `uv run pytest tests/test_chunker.py` passes.

## Step 4: Tokenizer and BM25 (`src/bm25.py`)

### 4a. Tokenizer

Turn text into lowercase words. Loop over characters: letters, digits and
`_` build a word; anything else ends it. Then the important trick for code:

- a question says "load a LoRA adapter"; the code says `load_lora_adapter`;
- so split identifiers on `_` and on camelCase boundaries, and keep **both**
  the parts (`load`, `lora`, `adapter`) and the whole name (to match a
  question that quotes the name exactly).

Drop a short list of stop words ("the", "how", "what", ... and "vllm",
which is in almost every question and tells nothing apart).

### 4b. BM25

Store an inverted index: `word -> [(chunk_number, count), ...]` plus the
length (in words) of every chunk. To score a query, for each query word:

```
idf   = log((N - df + 0.5) / (df + 0.5) + 1)     N = chunks, df = chunks with the word
score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / avg_length))
```

with `K1 = 1.5` and `B = 0.75` (standard values). Read it as: rare words
count more (idf), repeating a word helps but less and less (K1), and long
chunks are pulled down a bit (B). Sort the scores, keep the top k.

**Check:** `tests/test_bm25.py`: the chunk about "scheduler" wins for the
query "scheduler request"; an unknown word returns nothing.

## Step 5: Indexer (`src/indexer.py`)

1. Walk `data/raw/` with `os.walk`, keep `.py`, `.md`, `.txt`, sort the list
   (so two runs give identical indexes).
2. Read each file with `open(path, encoding="utf-8", errors="replace",
   newline="")`. `newline=""` matters: without it Python turns `\r\n` into
   `\n` and every character index after it shifts by one.
3. Chunk each file, and index each chunk's words **plus the words of its
   file path** (`docs/features/lora.md` gives `features`, `lora`, `md`).
   This is the cheapest big win for docs questions.
4. Save with `pickle` to `data/processed/index.pkl`, load it back with
   `Index.load`.
5. `file_path` in results must be exactly `data/raw/vllm-0.10.1/...`.
   Since you walk the relative folder `data/raw`, that is what you get.

**Check:** indexing the full corpus takes seconds (limit: 5 minutes).

## Step 6: CLI and the retrieval commands (`src/cli.py`, `src/__main__.py`)

1. Write each command as a plain function: `index`, `search`,
   `search_dataset`, `evaluate`. `fire.Fire({"index": index, ...})` turns
   their parameters into `--options` automatically.
2. Error handling, the part reviewers will poke at. Fire passes whatever was
   typed, so validate `k` yourself (`k=0`, `k=abc`). Then wrap
   `fire.Fire(...)` in `__main__.py`:
   ```python
   try:
       fire.Fire(COMMANDS)
   except (OSError, ValueError) as error:
       sys.exit(f"error: {error}")
   ```
   `OSError` covers missing files, `ValueError` covers bad options **and**
   malformed JSON, because pydantic's `ValidationError` is a `ValueError`.
3. `evaluate`: for each question, a reference source is found if one of
   your top-k results is in the same file with IoU >= 0.05, where
   IoU = overlap length / union length of the two ranges.

**Check:** run the real loop and compare with the moulinette:

```
uv run python -m src index
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

Your `evaluate` must print the same numbers as the moulinette. Targets:
recall@5 >= 0.80 on docs and >= 0.50 on code.

## Step 7: Improve retrieval (only if below target)

Change one thing at a time and re-run the step 6 check after each. Ideas,
roughly by payoff: path words in each chunk, identifier splitting, stop
words, chunk size (try `--max_chunk_size 1000` and `1500`). Write down what
helped and what did not: the README asks for it.

A good habit: print the questions you miss with what you returned instead.
Patterns jump out quickly (for example, results landing in `tests/`).

## Step 8: Answer generation (`src/generator.py`)

1. Load `Qwen/Qwen3-0.6B` with `AutoTokenizer` and
   `AutoModelForCausalLM`. Use the GPU if `torch.cuda.is_available()`.
   Import torch inside the class: search commands should not pay the import
   time.
2. Build the context: read each source's text back from disk, prefix it with
   its file path, and stop adding snippets once a token budget is reached
   (count with the tokenizer; the "token budget" in the subject).
3. Prompt: a system message that says "answer only from the context, say so
   if it is not there", then the context and the question.
4. Call `apply_chat_template(..., enable_thinking=False)`. Qwen3 otherwise
   writes a long `<think>` section first and runs out of tokens.
5. `generate(..., do_sample=False)` for reproducible answers, then decode
   only the new tokens (the output starts with a copy of the prompt).
6. Add the `answer` and `answer_dataset` commands.

**Check:** `uv run python -m src answer "What HTTP endpoint is used to
dynamically load a LoRA adapter in vLLM?"` mentions `/v1/load_lora_adapter`.

## Step 9: Tests and edge cases (`tests/`)

Run the CLI as a subprocess on a tiny fake corpus (pytest's `tmp_path`) and
assert on the output. Cover: empty query, nonsense query, `k=0`, `k=abc`,
missing files, malformed JSON, `max_chunk_size` above 2000. For each, the
test checks there is **no "Traceback"** in stderr.

**Check:** `make test` and `make lint` pass.

## Step 10: README and defense prep

Write `README.md` with every section the subject lists (chapter VIII). Then,
for the defense, make sure you can explain without notes:

- why chunks are positions and not text;
- the BM25 formula, term by term;
- why identifiers are split and also kept whole;
- why `newline=""` when reading files;
- what `enable_thinking=False` does;
- how `evaluate` decides a source was found.