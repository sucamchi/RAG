# Performance on the 42 cluster

The project was developed on a laptop with an NVIDIA RTX 4060. These are
the same benchmarks re-run on a 42 Málaga cluster computer, measured on
2026-11-04 from commit `d1d6460`.

## Machine

| | 42 cluster |
|---|---|
| CPU | Intel Core i5-10600 @ 3.30GHz, 6 cores / 12 threads |
| GPU | none (`nvidia-smi` not found), so `Generator` runs on CPU |
| RAM | 7.6 GB, about 3.6 GB free during the runs |
| Python | 3.13.10 via `uv` |
| torch / transformers | 2.15.0+cu132 / 5.22.1 |
| torch threads | 6 |
| Model cache | `HF_HOME=/sgoinfre/students/scamlett/.cache/huggingface` |
| Code and data | on `/sgoinfre` (NFS) |

## Indexing

Command: `uv run python -m src index` (`max_chunk_size` 2000).

| Run | Reported time | Wall time | Peak memory |
|---|---|---|---|
| 1 (cold NFS cache) | 8.4 s | 9.5 s | 179 MB |
| 2 | 5.3 s | 5.8 s | 179 MB |
| 3 | 4.0 s | 4.4 s | 179 MB |

- 1969 files (38 MB of `data/raw`), 13,230 chunks.
- `data/processed/index.pkl` is 9.7 MB.
- Far under the 5 minute limit. The first run is slower because the files
  are read over NFS for the first time.

## Search

| Measurement | Time |
|---|---|
| `search "<query>" --k 5`, whole command | 0.41 to 0.46 s |
| `Index.load` (unpickle) | 0.17 s |
| One `Index.search` call, index already loaded (average of 100) | 1.4 ms |
| `search_dataset`, docs (100 questions, k=10) | 0.82 s |
| `search_dataset`, code (100 questions, k=10) | 0.93 s |

Most of a single `search` command is Python start-up and loading the
pickle. The search itself is about a millisecond.

## Retrieval quality

`search_dataset --k 10`, then both `evaluate` and the official
`./moulinette evaluate_student_search_results --k 10
--max_context_length 2000`. The two gave identical numbers.

| Dataset | Questions evaluated | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required @5 |
|---|---|---|---|---|---|---|
| docs | 100 | 0.650 | 0.790 | 0.830 | 0.860 | 0.80 |
| code | 99 | 0.495 | 0.737 | 0.808 | 0.848 | 0.50 |

These match the README exactly, which is expected: BM25 has no randomness
and does not depend on the hardware. Both datasets pass.

The code dataset has 100 questions but only 99 have reference sources, so
moulinette and `evaluate` both average over 99.

## Generation (Qwen3-0.6B on CPU)

With no GPU, `Generator` picks `device="cpu"`. `torch_dtype="auto"` loads
the weights in **bfloat16**, the model's stored type.

### Model load

| | Time | Peak memory |
|---|---|---|
| `Generator()` (weights already in the HF cache) | 31.8 s | 2.4 GB (whole process) |

### Per question, as the code is now (bfloat16)

First 3 questions of the docs search results, k=10 sources each:

| Prompt tokens (context) | Answer tokens | Time |
|---|---|---|
| ~3560 | 24 | 70.2 s |
| ~4410 | 68 | 82.1 s |
| ~4300 | 21 | 77.2 s |

About **75 s per question**. The answers are short (20 to 70 tokens), so
almost all the time is spent reading the ~4000 token prompt, not writing
the answer.

### `answer_dataset`, end to end (bfloat16)

Run on the first 5 questions of each dataset's search results, with the
unchanged command:

| Dataset | Questions | Wall time | Per question (after ~32 s model load) | Peak memory |
|---|---|---|---|---|
| docs | 5 | 348.9 s | ~63 s | 2.3 GB |
| code | 5 | 290.1 s | ~52 s | 3.3 GB |

Every answer was produced and the output files validated. The answers
look like the laptop's: short and usually quoting the right identifier.
One code answer was just `"Batched"`, and one said `trust_remote_code`
defaults to `True` (vLLM's default is `False`), so the small model still
gets things wrong when the context is ambiguous.

### Same questions in float32 (not in the code, test only)

Loading the model with `dtype=torch.float32` instead of `"auto"`:

| Question | bfloat16 | float32 |
|---|---|---|
| 1 | 70.2 s | 24.0 s |
| 2 | 82.1 s | 42.4 s |
| 3 | 77.2 s | 25.1 s |

float32 is about **3x faster** on this CPU and gave the same answers on
these questions. The i5-10600 has no native bfloat16 instructions, so
every bfloat16 operation is emulated. Wrapping the call in
`torch.inference_mode()` saved about 3 s more (21.3 s for question 1).

### Estimated time for a full dataset

| | per question | 100 questions |
|---|---|---|
| bfloat16 (current code) | ~50 to 75 s | ~1.5 to 2 h |
| float32 | ~30 s | ~50 min |

## Compared with the laptop

| | Laptop (RTX 4060) | 42 cluster (CPU only) |
|---|---|---|
| Indexing | ~3 s (README) | 4 to 9 s |
| Index size | 9 MB | 9.7 MB |
| Recall@k | see README | identical |
| Answer generation | GPU, not measured in the README | ~50 to 75 s per question |

Everything except generation behaves the same. Generation is the only
part that needs the GPU; on the cluster it works but is slow.
