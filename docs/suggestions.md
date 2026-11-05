# Suggestions

Found while benchmarking on the 42 cluster (see
[42performance.md](42performance.md)). Ranked by how much they matter.
Nothing here has been applied.

## Code

### 1. Use float32 on CPU (`src/generator.py`)

`torch_dtype="auto"` loads bfloat16. On a CPU without bfloat16
instructions (like the cluster's i5-10600) that is about 3x slower than
float32: ~75 s against ~30 s per question. Same answers on the questions
tested. The model is only 0.6B parameters, so float32 still fits easily
(about 2.4 GB).

`torch_dtype` is also deprecated in transformers 5 and prints a warning;
the new name is `dtype`.

```python
# bfloat16 is emulated (slow) on most CPUs, float32 is ~3x faster there.
dtype = "auto" if self.device == "cuda" else torch.float32
self.model: Any = AutoModelForCausalLM.from_pretrained(
    self.MODEL, dtype=dtype, device_map=self.device)
```

### 2. Run generation under `torch.inference_mode()` (`src/generator.py`)

`model.eval()` turns off dropout but autograd still records operations.
Wrapping `self.model.generate(...)` in `with torch.inference_mode():`
saved about 3 s per question on CPU. `torch` is imported inside
`__init__`, so `answer` would need its own local import (or keep a
reference on `self`).

### 3. `make test` fails

`tests/` is empty and listed in `.gitignore`, so `make test` exits with
pytest's error 5 ("no tests ran"). Either drop the `test` target, or add
a couple of small tests (for example `snake_camel_split`, `chunk_text`
never going over `max_chunk_size`, `iou`) and remove `tests/` from
`.gitignore` so evaluators see them.

### 4. `.PHONY` names a target that does not exist (`Makefile`)

`.PHONY` lists `lint-strict`, but there is no `lint-strict` rule. Remove
it from the list.

### 5. `read_context` and `read_file` open files differently

`read_file` in `src/indexer.py` uses `errors="replace"`; `read_context`
in `src/cli.py` does not. A file with invalid UTF-8 would index fine and
then crash `answer`. No file in vLLM 0.10.1 has invalid UTF-8 (checked),
so this cannot happen today. `read_context` could just call `read_file`
so the two can never disagree.

### 6. Fewer sources for answering (optional, changes behavior)

Generation time on CPU is almost all prompt reading: ~4000 tokens of
context for a 20 to 70 token answer. Answering from the top 5 sources
instead of 10 would roughly halve that time. Recall only goes from 0.83
to 0.86 (docs) between k=5 and k=10, so the answers would mostly see the
same right chunk. Only worth it if CPU answering matters to you.

## README

### 7. Code dataset question count

The retrieval table says "code (100 questions)", but only 99 have
reference sources, and moulinette prints "Questions evaluated: 99".

### 8. Say what hardware the numbers come from

"Indexing ... takes around 3 seconds" was measured on the laptop; on the
cluster it is 4 to 9 s. One line naming the machine (or a link to
`docs/42performance.md`) makes the numbers reproducible.

### 9. Mention CPU-only generation

The README never says the generator runs without a GPU or how slow it is.
A sentence under "Answers" or "Performance analysis" would help an
evaluator on a cluster machine: it works, model load takes ~30 s, and
each answer takes ~30 to 75 s on CPU.

### 10. Mention where the model is cached

The first `answer` downloads Qwen3-0.6B (~1.5 GB) into the Hugging Face
cache. On the cluster the home disk is small (4.7 GB, 93% full here), so
setting `HF_HOME` to a folder in `sgoinfre` before running avoids filling
it. Worth one line under Instructions.
