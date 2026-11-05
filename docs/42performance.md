# Performance on the 42 cluster

The project was developed on a laptop with an NVIDIA RTX 4060. This page
compares it with a 42 Málaga cluster computer, which has no GPU, so
everything runs on the CPU. Measured on 2026-11-05.

## Machines

| | Laptop | 42 cluster |
|---|---|---|
| GPU | NVIDIA RTX 4060 | none |
| CPU | | Intel Core i5-10600, 6 cores / 12 threads |
| RAM | | 7.6 GB, 2 to 3.6 GB free during the runs |

## Comparison

| | Laptop (GPU) | 42 cluster (CPU only) |
|---|---|---|
| Indexing | ~3 s | 4 to 9 s |
| Search, one query | under 0.5 s | under 0.5 s |
| Recall@k | see README | identical |
| Model load | | ~30 s |
| One answer | | ~20 to 40 s |
| Memory while answering | | ~4.4 GB |

Indexing, search and recall@k are not affected by the GPU. BM25 only
counts words, so it is fast on any CPU and gives the same results. The
first indexing run on the cluster is the slow one (9 s) because the files
are read over the network disk for the first time.

Only answer generation uses the GPU. On the cluster it still works, but
is slower.

## Answer generation on the cluster

First 2 questions of the docs search results, 10 sources each:

| Context tokens | Answer tokens | Time |
|---|---|---|
| ~3560 | 24 | 22.0 s |
| ~4410 | 68 | 40.5 s |

The answers are short, so most of the time goes into reading the ~4000
token context, not writing the answer. A longer context means a slower
answer.

A full dataset of 100 questions takes about 50 minutes on the cluster.

The answers are the same kind as on the laptop: short, and usually quoting
the right identifier or command when the right chunk was retrieved.

## Memory

Answering uses about 4.4 GB, which is a large share of the cluster's
7.6 GB. It runs fine, but with a browser and other programs open the
machine can start swapping and get slower.
