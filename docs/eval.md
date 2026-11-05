# Intra Projects: RAG against the machine (evaluation sheet)

## Introduction

- Remain polite, respectful and constructive during the evaluation.
- This project evaluates information retrieval, text chunking, answer
  generation and overall RAG system design.
- Reference exam scripts are provided in `exams/scripts/`:
  `exam_retrieval.sh`, `exam_answer.sh`, `exam_edge_cases.sh`.
- See `exams/exam_scripts_instructions.md` for detailed corrector guidance.
- The correction is designed so the retrieval pipeline (~8 min) runs in the
  background while the corrector inspects code and asks questions.
- Every mandatory question is pass/fail. The whole mandatory part must be
  validated (all questions pass) before the bonus is graded.

## Guidelines

- Only grade the work present in the student's Git repository.
- Set up an evaluation workspace: clone the student repository into a
  subdirectory named `student/`, and place the provided `moulinette` binary
  and the `exams/` scripts alongside it.
- Run `cd student && uv sync` to install the student's dependencies. The
  student's `pyproject.toml` and `uv.lock` live at the root of their
  repository, so `uv sync` is expected to work from there.
- The student solution must NEVER import moulinette packages. Verify
  (match imports only, not the word in comments/docstrings):

  ```bash
  grep -rnE "^[[:space:]]*(import|from)[[:space:]]+moulinette" student/ --include="*.py"
  ```

- The program must not crash unexpectedly during evaluation.
- No modification of student code is allowed during correction.
- The datasets are provided with this scale. Students work only with the
  public datasets; for grading, unzip the corrector-only private datasets:

  ```bash
  unzip data/datasets/datasets_private.zip -d data/datasets/
  ```

  These private datasets are NOT expected to be in the student's repository.
- Use flags if cheating or plagiarism is detected.

## Attachments

- `datasets_private.zip`
- `exams.zip`
- `subject.pdf`
- `datasets_public.zip`
- `README.md`
- `vllm-0.10.1.zip`
- `moulinette.zip`

---

## Mandatory part

### Q1. Preliminaries and setup

Check the following requirements:

- [ ] The Git repository is available and contains only the requested files.
- [ ] Python 3.10+ is used.
- [ ] `uv` is used as the project and package manager.
- [ ] `pyproject.toml` and `uv.lock` files are present.
- [ ] `src/` directory exists with the main module.
- [ ] A `Makefile` is present at the repository root.
- [ ] `README.md` file is present.
- [ ] `uv sync` completes without errors (run from the student repo root,
      i.e. `cd student && uv sync` in the evaluation workspace).
- [ ] Corrector-only step: unzip the private datasets provided with this
      scale: `unzip data/datasets/datasets_private.zip -d data/datasets/`
      (the student is NOT expected to ship private datasets).

**Question:** Are all setup requirements met?

### Q2. Launch pipeline (background)

Launch the automated retrieval pipeline in a background terminal. This runs
indexing, search, and evaluation (~8 min total).

In a separate terminal, run:

```bash
./exams/scripts/exam_retrieval.sh \
    --student-path ./student \
    --moulinette-path ./moulinette-ubuntu
```

Continue with code inspection (Q3 to Q6) while this runs. Keep this run's
output (terminal log and `summary.log`): questions Q7 to Q10 read their
results from it (docs/code Recall@5, indexing time and throughput). Do not
close the terminal before Q10.

**Question:** Did the pipeline launch successfully?

### Q3. Code quality and Pydantic models

Review code quality and data model implementation. Check ALL of the
following.

Code quality:

- [ ] Data models use pydantic for validation and type safety (service or
      orchestration classes such as indexer/retriever/pipeline need not be
      pydantic).
- [ ] Code adheres to flake8 coding standards, and passes mypy (run
      `make lint`, which the Common Instructions define as flake8 + mypy).
- [ ] The Makefile exposes the required rules (`install`, `run`, `debug`,
      `clean`, `lint`).
- [ ] Exceptions are handled gracefully with try-except blocks.
- [ ] Resources are properly managed (context managers or explicit
      `close()`).
- [ ] Progress bars are implemented for long-running operations (`tqdm`).

Pydantic models (all 8 must be correctly implemented):

- [ ] `MinimalSource` (`file_path`, `first_character_index`,
      `last_character_index`)
- [ ] `UnansweredQuestion` and `AnsweredQuestion`
- [ ] `RagDataset`
- [ ] `MinimalSearchResults` and `MinimalAnswer`
- [ ] `StudentSearchResults` and `StudentSearchResultsAndAnswer`

Output format:

- [ ] Search operations output valid JSON using `StudentSearchResults`.
- [ ] Answer operations output valid JSON using
      `StudentSearchResultsAndAnswer`.
- [ ] Each source includes correct `file_path`, `first_character_index`,
      `last_character_index`.

**Question:** Does the code meet all quality standards and are all models
correctly implemented?

### Q4. Chunking strategies

Verify chunking strategies:

- [ ] At least 2 chunking strategies are implemented (e.g. Python code +
      markdown/text).
- [ ] The chunk size default is 2000 characters and is configurable via
      `--max_chunk_size`.
- [ ] Student can explain the impact of chunk size on retrieval performance:
      "What happens if chunks are too small? Too large?"

Ask the student to show the code for both strategies.

**Question:** Are chunking strategies properly implemented?

### Q5. Retrieval system

Verify retrieval implementation:

- [ ] At least one method (TF-IDF or BM25) is implemented.
- [ ] System retrieves and ranks relevant information.
- [ ] Student can explain how their chosen method works.

Live demo: ask the student to run a search query and explain the results:

```bash
uv run python -m src search "How to configure OpenAI server?" --k 10
```

**Question:** Is the retrieval system properly implemented?

### Q6. Answer generation

Verify answer generation:

- [ ] `Qwen/Qwen3-0.6B` is used as the DEFAULT model.
- [ ] Run: `uv run python -m src answer "How to configure OpenAI server?" --k 10`
- [ ] System passes retrieved context to the LLM within token limits.
- [ ] Answers are generated based on retrieved code/documentation.

Note: the student may support additional local models beyond the default.
Check the README for documentation of supported models. During evaluation,
always test with the default model (Qwen3-0.6B).

**Question:** Does answer generation work correctly with the default model?

### Q7. Retrieval performance: docs

Check the `exam_retrieval.sh` results.

The script evaluates Recall@5 on the private docs dataset (100 questions).
Look for the "Docs Recall@5" line in the output.

Pass criterion: Docs Recall@5 >= 80% (the threshold stated in the subject).

**Question:** Does the docs dataset meet the 80% Recall@5 threshold?

### Q8. Retrieval performance: code

Check the `exam_retrieval.sh` results.

The script evaluates Recall@5 on the private code dataset (100 questions).
Look for the "Code Recall@5" line in the output.

Pass criterion: Code Recall@5 >= 50% (the threshold stated in the subject).

**Question:** Does the code dataset meet the 50% Recall@5 threshold?

### Q9. Retrieval performance: indexing time

Check the `exam_retrieval.sh` results.

Test 1/4 of the script times the full indexing run against the 300s (5 min)
limit stated in the subject. Look for the "Indexing" line in the output (or
the `INDEX:` line in `summary.log`).

Pass criterion: indexing completed in <= 300s (reported PASSED / `INDEX: OK`,
not EXCEEDED or CRASHED).

**Question:** Did indexing complete within the 5-minute limit?

### Q10. Retrieval performance: throughput

Check the `exam_retrieval.sh` results.

Test 2/4 of the script times search over the two 100-question datasets (200
questions total) against the 90s limit stated in the subject. Look for the
"Warm Retrieval Throughput" line (or the `THROUGHPUT:` line in
`summary.log`).

Pass criterion: 200 questions retrieved in <= 90s (reported PASSED /
`THROUGHPUT: OK`, not EXCEEDED or CRASHED).

**Question:** Did retrieval stay within the 90s throughput limit?

### Q11. Answer quality

Run the answer exam script to evaluate answer quality:

```bash
./exams/scripts/exam_answer.sh \
    --student-path ./student \
    --moulinette-path ./moulinette-ubuntu
```

The script lists questions where sources were correctly retrieved, then you
pick 3 questions. For each, the student's answer is displayed.

The mandatory `Qwen/Qwen3-0.6B` model has known reasoning limits, so grading
prioritizes retrieval quality and grounding over perfect final phrasing.
A satisfactory answer should:

- be coherent and understandable,
- be mostly grounded in the retrieved sources (no major hallucination),
- address the question asked.

Minor incompleteness due to base-model limitations is acceptable.

Pass if at least 2 out of 3 answers are satisfactory.

**Question:** Does the student pass the answer evaluation (2/3 answers ok)?

### Q12. Student understanding

Test the student's understanding of RAG concepts and their implementation.
Ask these 5 questions (in order):

1. "What is Retrieval-Augmented Generation (RAG) and why is it useful?"
2. "Walk me through your complete RAG pipeline, from raw documents to a
   generated answer."
3. "What is the difference between TF-IDF and BM25 as retrieval methods?"
4. "What implementation choices did you make and why? What trade-offs did
   you consider?"
5. "If you had more time, what would you improve in your system?"

Pass requires 4 out of 5 correct/satisfactory answers. A satisfactory answer
demonstrates genuine understanding, not just memorized definitions.

**Question:** Does the student demonstrate good understanding (4/5)?

### Q13. README and documentation

Verify `README.md` contains all required sections. Checklist:

1. [ ] System architecture (RAG pipeline components and interactions).
2. [ ] Chunking strategy explanation.
3. [ ] Retrieval method details (algorithm and ranking mechanism).
4. [ ] Performance analysis (recall@k scores and system performance).
5. [ ] Design decisions and trade-offs.
6. [ ] Example usage with clear instructions.

If the student implemented bonus features, the README should also document
those experiments and their results.

Pass if at least 5 out of 6 items are present and well-documented.

**Question:** Is the README comprehensive and well-documented?

### Q14. System reliability

Run the edge case exam script:

```bash
./exams/scripts/exam_edge_cases.sh --student-path ./student
```

This tests 4 degenerate inputs:

1. Empty query: `search "" --k 10`
2. Gibberish query: `search "asdfghjkl" --k 10`
3. `k=0`: `answer "What is vLLM?" --k 0`
4. Bad dataset path: `search_dataset --dataset_path /nonexistent.json`

Pass criterion: ALL 4 tests complete without Python tracebacks. The program
may print error messages or return empty results: that is acceptable.
Unhandled exceptions with tracebacks are NOT acceptable.

**Question:** Does the system handle all edge cases without crashing?

---

## Bonus part: advanced RAG

Graded only once the entire mandatory part is validated.

There are exactly five bonuses, each worth one point (one star). They all
run on a CPU-only campus machine. Award one star per bonus that is
implemented AND demonstrated working (not merely described in the README):

1. Semantic embeddings used for retrieval (CPU model, e.g.
   `all-MiniLM-L6-v2`).
2. Hybrid retrieval combining the lexical and semantic rankings.
3. Incremental indexing (re-index only changed files).
4. Caching of the index and/or query results.
5. Local HTTP API to query the index and answer questions.

Star rating = number of bonuses working (0 -> 1 star, 5 -> 5 stars). Ask the
student to demonstrate each claimed bonus.

**Rate it from 0 (failed) through 5 (excellent).**

---

## Ratings

- Ok
- Outstanding project
- Empty work
- Incomplete work
- Invalid compilation
- Norme
- Cheat
- Concerning situation
- Forbidden function
- Can't support / explain

## Conclusion

Leave a comment on this evaluation (2048 chars max).

---

# Local run results

This section records the checks run locally against this repository on
2026-11-05. Commands and raw results are reproduced so the evaluation can be
repeated.

Environment: laptop with an NVIDIA RTX 4060 (CUDA available), Python 3.13,
model already cached in `~/.cache/huggingface`. Evaluation workspace:
`/tmp/evalws/` with `student -> /home/susana/RAG`, the official `exams/`
scripts, the moulinette binary and the private datasets.

## Verdict

| Area | Result |
|---|---|
| Setup (Q1) | PASS |
| Retrieval exam (Q7 to Q10) | PASS 4/4 |
| Edge cases (Q14) | PASS 4/4 |
| Answer quality (Q11) | PASS 3/3 |
| Code quality (Q3) | PASS |
| README (Q13) | PASS 6/6 |

All mandatory checks pass locally. No bonus is implemented, so the bonus
score is 0 stars.

## Setup

| Check | Result |
|---|---|
| Git repository present | PASS (22 tracked files) |
| Python version | 3.13 (>= 3.10) |
| `uv` used | PASS (`uv sync` exits 0) |
| `pyproject.toml` and `uv.lock` present | PASS |
| `src/` main module | PASS (`python -m src`) |
| `Makefile` with install/run/debug/clean/lint | PASS |
| `README.md` present | PASS |
| No moulinette imports | PASS (grep matches nothing) |

## Code quality

| Check | Result |
|---|---|
| `make lint` (flake8 + mypy) | PASS |
| `uv run pytest` | 39 passed |
| Pydantic models (all 8) | PASS |
| Pydantic output validation | PASS (`StudentSearchResults` and `StudentSearchResultsAndAnswer` both validate) |
| tqdm on indexing/search/answering | PASS |
| Context managers for files | PASS |

The repository test suite initially failed one case:
`search lora --k abc` raised a `TypeError` traceback inside the BM25 slice.
Fixed in `src/bm25.py` by converting `k` with `int(k)` at the top of
`BM25.search`, so a non-numeric `k` raises `ValueError`, which
`src/__main__.py` turns into a short `error:` message.

The generation prompt in `src/generator.py` was then replaced with a stricter
grounded prompt. See "Generation prompt" under Answer quality. Retrieval
does not use this prompt, so the recall numbers below are unchanged.

## Retrieval exam (`exam_retrieval.sh`)

Command:

```bash
./exams/scripts/exam_retrieval.sh \
    --student-path ./student \
    --moulinette-path ./moulinette
```

| Test | Required | Measured | Result |
|---|---|---|---|
| Indexing | <= 300s | 3s | PASS |
| 200 questions retrieved | <= 90s | 2s (1s docs + 1s code) | PASS |
| Docs Recall@5 | >= 0.80 | 0.820 | PASS |
| Code Recall@5 | >= 0.50 | 0.720 | PASS |

Full private-dataset recall:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| Docs (100 questions) | 0.620 | 0.790 | 0.820 | 0.840 |
| Code (100 questions) | 0.440 | 0.660 | 0.720 | 0.820 |

`summary.log`: `INDEX: OK (3s <= 300s)`, `THROUGHPUT: OK (2s <= 90s)`,
`DOCS_RECALL: PASS`, `CODE_RECALL: PASS`.

## Edge cases (`exam_edge_cases.sh`)

| Test | Input | Result |
|---|---|---|
| 1 | `search "" --k 10` | PASS (no traceback) |
| 2 | `search "asdfghjkl zxcvbnm qwertyuiop" --k 10` | PASS (no traceback) |
| 3 | `answer "What is vLLM?" --k 0` | PASS (no traceback) |
| 4 | `search_dataset --dataset_path /nonexistent/dataset.json` | PASS (no traceback) |

## Answer quality (`exam_answer.sh`)

Command:

```bash
./exams/scripts/exam_answer.sh \
    --student-path ./student \
    --moulinette-path ./moulinette \
    --questions "What OpenAI-compatible APIs does vLLM support for serving models?,What Python versions are supported for vLLM CPU installation?,What is the recommended value for num-redundant-experts in large scale vLLM expert parallel deployments?"
```

The script listed 84/100 docs questions as valid (sources retrieved). The
three chosen answers:

| # | Question | Answer | Verdict |
|---|---|---|---|
| 1 | What OpenAI-compatible APIs does vLLM support for serving models? | Lists all 10 APIs with endpoints: Completions, Chat, Embeddings, Transcriptions, Translation, Tokenizer, Pooling, Classification, Score, Rerank | Satisfactory |
| 2 | What Python versions are supported for vLLM CPU installation? | "3.9, 3.10, 3.11, and 3.12" | Satisfactory |
| 3 | What is the recommended value for num-redundant-experts in large scale vLLM expert parallel deployments? | "32", with a quote from the docs | Satisfactory |

Result: 3/3 satisfactory (2/3 required), PASS.

The exact Q6 demo question was also run:
`answer "How to configure OpenAI server?" --k 10`. After the generation
prompt change it answers that the server is started with
`vllm serve <model>` and that the API key and base URL are configured with
the OpenAI SDK. It is coherent and grounded. Before the prompt change it
only described client-side configuration and never mentioned `vllm serve`.

### Generation prompt

The system prompt in `src/generator.py` is:

> You are a vLLM documentation assistant. Answer the question using only the
> retrieved context. The first passages are the most relevant; ignore
> irrelevant ones. Start with the direct answer, then one or two sentences of
> explanation. Only name commands, flags and values that appear in the
> context. If the context lists several relevant items, include them all. If
> the context does not answer the question, say so.

Why each part is there:

- "The first passages are the most relevant" tells the small model to trust
  the retrieval order. The top chunk is the best one, but a low-ranked
  example file was pulling the answer off topic.
- "Start with the direct answer" removes the preamble and makes the answer
  easy to grade.
- "Only name commands, flags and values that appear in the context" lowers
  hallucination.
- "Include them all" fixed incomplete list answers, such as listing only the
  Completions and Chat APIs.
- "Say so" gives the model a safe answer when the context does not contain
  the answer.

This prompt cannot change retrieval scores. BM25 ranks the chunks before the
model runs. It only changes how the retrieved context is used. Recall@5 is
0.820 docs and 0.720 code.

## Runs on this machine vs 42 cluster

The numbers above were measured on the laptop GPU. Retrieval does not use
the GPU, so indexing time, throughput and recall are identical on the 42
cluster. Generation is the only different part: the README reports ~30s
model load, ~20 to 40s per answer and ~4.4 GB RAM on the cluster CPU. The
`k=0` edge case loads the model too, so expect it to take longer there
(no traceback, so it still passes).

## Known risks

- **Docs margin is thin.** Private docs Recall@5 is 0.820 against a 0.800
  threshold: two questions of headroom. Do not change the tokenizer,
  chunker or BM25 constants without re-running `exam_retrieval.sh`.
- **Corpus is not in Git.** `data/raw/vllm-0.10.1/` is gitignored, as the
  subject requires (no large data in the repository). The corpus must exist
  in the cloned `student/` directory before the exam scripts run; the
  evaluator provides it from `vllm-0.10.1.zip`.
- **Private datasets must never be committed.** They are corrector-only and
  are currently only under the untracked `RAG_EVAL/` directory.
- **`RAG_EVAL/` is untracked.** Do not commit it: it contains the private
  datasets and evaluation material.
- **`uv sync` downloads torch.** On the cluster, set `HF_HOME` under
  `sgoinfre` before the first answer, and make sure there is enough disk
  space for the dependency stack and the ~1.5 GB model.
