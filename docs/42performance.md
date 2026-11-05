# Performance on the 42 cluster

The project was developed on a laptop with an NVIDIA RTX 4060. This page
compares it with a 42 Málaga cluster computer, which has no GPU, so
generation runs on its CPU. The laptop measurements were rerun on
2026-11-05 with the final code.

## Machines

| | Laptop | 42 cluster |
|---|---|---|
| GPU | NVIDIA RTX 4060 Laptop GPU, 8 GB | none |
| CPU | Intel Core i7-13620H, 10 cores / 16 threads, up to 4.9 GHz | Intel Core i5-10600, 6 cores / 12 threads |
| RAM | 16 GB | 7.6 GB|


The laptop is a hybrid CPU: 6 performance cores and 4 efficient cores,
with hyper-threading on the performance cores. The cluster CPU is older
and has fewer threads, which matters when generation runs without a GPU.

## How the numbers were measured

- **Indexing:** `uv run python -m src index --max_chunk_size 2000` on the
  full corpus, three times, with `data/processed/` deleted between runs.
  Both the timer printed by the code and the total wall time are reported.
- **Search, one query:** `uv run python -m src search ... --k 10`, three
  times. This includes Python startup and loading the index.
- **Search, 100 questions:** one process, index loaded once, private docs
  dataset, timing every `Index.search` call.
- **Recall:** `search_dataset` followed by `evaluate` on the public
  datasets, and the official exam on the private datasets.
- **Generation:** first 2 docs questions with 10 sources each, greedy
  decoding, 256 maximum new tokens. On the laptop GPU, one warm-up run and
  three timed runs per question.
- **Memory:** peak GPU memory from `torch.cuda.max_memory_allocated` and
  `max_memory_reserved`, and peak process memory from `ru_maxrss`.

## Summary

| Step | Laptop GPU | 42 cluster CPU |
|---|---|---|
| Indexing, full corpus | 2.5 to 2.7 s build, 2.8 to 3.0 s wall | 4 to 9 s |
| Search, one CLI query | 0.38 to 0.44 s | under 0.5 s |
| Search, 100 questions in one process | 0.30 s total, 3.0 ms per query | not measured separately |
| Docs Recall@5, private | 0.820 | 0.820 |
| Code Recall@5, private | 0.720 | 0.720 |
| Model load | 5.2 s | ~30 s |
| One answer, ~4k context | 0.67 to 0.96 s | 22.0 to 40.5 s |
| 100 answers, public docs | 123 s | ~50 min, estimated |
| Peak memory while answering | 1.70 GiB GPU, 2.37 GiB RAM | ~4.4 GB RAM |

Indexing, search and recall@k never use the GPU. BM25 only counts words,
so those numbers are identical on every machine. Generation is the step
where the hardware matters: the laptop GPU answers a 4k-token context in
under a second, while the cluster CPU needs 22 to 40 seconds.

## Indexing and search

All 1969 corpus files produce 13,230 chunks and a 9.2 MB index file.

Laptop indexing, three runs:

| Run | Build timer | Wall time | Peak RSS |
|---|---|---|---|
| 1 | 2.5 s | 2.84 s | 179 MiB |
| 2 | 2.5 s | 2.76 s | 179 MiB |
| 3 | 2.7 s | 2.99 s | 179 MiB |

The build timer is the time printed by the code. The wall time also
includes Python and `uv` startup. The index is small because it stores
only chunk locations and BM25 statistics, not the chunk text.

The cluster takes 4 to 9 seconds. The first run is the slow one (9 s)
because the corpus files are read over the network disk for the first
time. Later runs are closer to 4 seconds. The laptop reads from a local
disk, so its three runs are already warm and stay near 3 seconds.

Search is cheap because it only walks the postings lists of the query
words:

| Measurement | Result |
|---|---|
| Index load | 0.120 s |
| 100 questions, total | 0.298 s |
| Mean per question | 2.98 ms |
| Median per question | 2.79 ms |
| 95th percentile | 4.98 ms |
| Slowest question | 7.05 ms |
| Throughput | 336 questions/s |

A single CLI search takes 0.38 to 0.44 seconds because process startup
and loading the index cost more than the search itself. The same holds on
the cluster, where one query stays under 0.5 seconds. The official exam
on the laptop retrieved its 200 questions in about 2 seconds.

Recall is identical on both machines because the tokenizer, chunker and
BM25 code are deterministic and CPU-only:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required @5 |
|---|---|---|---|---|---|
| Public docs (100) | 0.650 | 0.790 | 0.830 | 0.860 | 0.80 |
| Public code (99) | 0.495 | 0.737 | 0.808 | 0.848 | 0.50 |
| Private docs (100) | 0.620 | 0.790 | 0.820 | 0.840 | 0.80 |
| Private code (100) | 0.440 | 0.660 | 0.720 | 0.820 | 0.50 |

## Answer generation

The first two docs questions have different context sizes, so the table
shows the token counts next to the times.

Laptop GPU, same model and same prompt:

| Question | Context tokens | Answer tokens | Time |
|---|---|---|---|
| vLLM metric for running requests | 3914 | 27 | 0.67 s |
| Structure of a rerank response | 4066 | 48 | 0.95 to 0.96 s |

The GPU runs were stable: 0.67, 0.67 and 0.67 seconds for the first
question, and 0.95, 0.96 and 0.96 seconds for the second. The second
answer is longer, and the extra time matches the extra output tokens.

The cluster's first answer took 22.0 seconds for a similar 4k-token
context, against 0.67 seconds here, so the laptop GPU is about 33 times
faster on that question. The cluster's second answer took 40.5 seconds
against 0.96 seconds here, about 42 times faster. Across the two
questions, the laptop GPU is roughly 30 to 40 times faster.

Model load is different. The laptop loads the model in 5.2 seconds. The
cluster takes about 30 seconds, which is probably dominated by disk and
environment startup rather than by the model itself.

Cluster generation:

| Context tokens | Answer tokens | Time |
|---|---|---|
| ~3560 | 24 | 22.0 s |
| ~4410 | 68 | 40.5 s |

The answers are short, so most of the CPU time goes into reading the
~4000 token context, not writing the answer. A longer context means a
slower answer. The laptop GPU removes almost all of that cost because the
matrix operations run in parallel on the GPU.

## A full dataset

The public docs dataset has 100 questions. Answering all of them on the
laptop GPU took 123 seconds of wall time, including the 5 second model
load. That is about 1.2 seconds per question, with peak process memory of
2.37 GiB.

The cluster estimate is about 50 minutes for its 100 questions. The GPU
brings the dataset from tens of minutes to about two minutes.

## Memory

| Setup | GPU memory | Process RAM |
|---|---|---|
| Laptop, GPU | 1.11 GiB after load, 1.65 GiB peak allocated, 1.70 GiB peak reserved | 2.37 GiB |
| 42 cluster, CPU only | none | ~4.4 GB |

On the laptop, the model weights and the working tensors live in the 8 GB
of VRAM, so the Python process keeps only about 2.4 GiB in RAM. The
cluster runs Qwen3-0.6B in float32 on the CPU, so the weights sit in
system RAM instead. Its 4.4 GB is a large share of the 7.6 GB machine. It
runs fine, but with a browser and other programs open the machine can
start swapping and get slower.

## What the GPU changes

- **Indexing, search and recall:** nothing. They are CPU-only and give
  the same results and nearly the same speed on both machines.
- **Answer generation:** the large difference. One answer goes from 22 to
  40.5 seconds on the cluster CPU to 0.67 to 0.96 seconds on the laptop
  GPU, about 30 to 40 times faster.
- **Model load:** the laptop loads in 5.2 seconds against about 30
  seconds on the cluster. The GPU is not the reason, since loading is
  mostly disk and startup.
- **Memory:** the GPU moves the model out of system RAM and into VRAM,
  which matters most on the 7.6 GB cluster machine.

