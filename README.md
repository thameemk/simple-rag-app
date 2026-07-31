# Simple RAG App

A small FastAPI app that answers questions using your own documents (retrieval-augmented generation).

It embeds `apps/documents.txt` with a local sentence-transformer model, finds the most relevant chunks for a question, and sends them to an LLM (OpenAI or Hugging Face) to generate an answer.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (or pip, if you prefer)

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Copy the env file and fill in your keys:

   ```bash
   cp .env.example .env
   ```

   Edit `.env`:

   ```
   LLM_PROVIDER=openai   # or huggingface

   OPENAI_API_KEY=your_key_here
   OPENAI_MODEL=gpt-4o-mini

   HF_TOKEN=your_hf_token_here
   HF_MODEL=meta-llama/Llama-3.1-8B-Instruct:novita
   ```

   You only need to fill in the keys for the provider you're using.

   - OpenAI key: https://platform.openai.com/api-keys
   - Hugging Face token (free): https://huggingface.co/settings/tokens

## Run

```bash
cd apps
uv run uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## Try it

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "your question here"}'
```

## Switching providers

Change `LLM_PROVIDER` in `.env` to `openai` or `huggingface` and restart the server. No code changes needed.

## Adding your own documents

Edit `apps/documents.txt`. Separate each chunk/document with a blank line.
