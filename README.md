# self-rag

This repository showcases the Self-RAG - advanced algorithm that combines the power of retrieval-based and generation-based approaches in natural language processing. It dynamically decides whether to use retrieved information and how to best utilize it in generating responses, aiming to produce more accurate, relevant, and useful outputs.

To start working with this repo, do the following

1. Restore uv project

```bash
uv sync
```

2. Create .env file in the root directory and set `OPENAI_API_KEY` and `OPENAI_API_BASE` (**if using AzureOpenAI**)

3. Open [main.ipynb](./main.ipynb) Jupyter notebook and follow along.

## Resources

- Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection - https://arxiv.org/abs/2310.11511
