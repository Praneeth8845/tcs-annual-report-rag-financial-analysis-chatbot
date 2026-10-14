# TCS Annual Report - RAG-Based Financial Analysis Chatbot

A Flask API that retrieves relevant passages from a locally indexed TCS annual
report and can use OpenAI to generate an answer grounded in those passages.
Without an OpenAI API key, the API returns the matching document excerpts.

## Requirements

- Python 3.10 or later
- A TCS annual report PDF that you are permitted to use
- An OpenAI API key for generated answers (optional)

## Setup

Run these commands from the project directory in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the PDF at `documents/tcs.pdf`, then build the local FAISS index:

```powershell
python batchjob.py
```

The PDF and generated index are intentionally excluded from Git. Obtain and
place the report locally before building the index.

## Run the API

Optionally set your OpenAI API key and model in the environment. Do not commit
API keys or other secrets.

```powershell
$env:OPENAI_API_KEY = "your-api-key"
python app.py
```

The service listens on `http://127.0.0.1:5000`. Send a `POST` request to
`/tcs` with a JSON body:

```json
{
  "question": "What was TCS's revenue?"
}
```

The response includes `response` and `response_con`; `answer_type` indicates
whether it is an AI-generated answer or document excerpts. The API also accepts
`tcs_question` in place of `question`. A `GET /tcs?question=...` request is
supported as well.
