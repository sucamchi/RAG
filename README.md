*This project has been created as part of the 42 curriculum by scamlett.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation system that answers questions about
the vLLM codebase. It cuts the vLLM repository into chunks, indexes them
with BM25, retrieves the chunks most relevant to a question, and passes
them to the small local model Qwen3-0.6B to generate an answer.
Retrieval quality is measured with recall@k.

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

## Retrieval method

## Performance analysis

## Design decisions

## Challenges faced

## Resources

- [Qwen3 model](https://huggingface.co/Qwen/Qwen3-0.6B).
- [Hugging Face transformers: text generation](https://huggingface.co/docs/transformers/main_classes/text_generation).
- [Python Fire CLI](https://github.com/google/python-fire/blob/master/docs/guide.md).

AI was used in a responsible manner to help understand new concepts, draft a step-by-step plan, testing and edge case handling.