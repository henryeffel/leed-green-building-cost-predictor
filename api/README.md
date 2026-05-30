# Optional FastAPI Backend Extension

## Purpose

This prototype extends the existing Streamlit MVP with a lightweight FastAPI backend. It exposes the same core prediction and minimum-cost recommendation workflow through API endpoints without replacing the Streamlit app.

## Core Design

ML and optimization functions calculate all numerical outputs. The optional LLM layer only explains validated JSON outputs from the backend.

This separation keeps score, cost, target-gap, and recommendation calculations deterministic and reduces hallucination risk.

## Run

```bash
uvicorn api.main:app --reload
```

## Endpoints

- `GET /health`
- `POST /predict`
- `POST /recommend`

## Example `.env`

```bash
OPENAI_API_KEY=...
```

The OpenAI API key is optional. If it is missing or the OpenAI request fails, `/recommend` returns a deterministic fallback report.

## Limitations

- Not an official LEED tool.
- Does not replace USGBC review.
- Uses a synthetic dataset.
- OpenAI API is optional.
- The LLM does not perform numerical calculation.
