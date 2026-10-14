"""Flask API that answers questions using the TCS document
        1.create a flask function that takes  that takes  quesiton from postion
        2.read the question from json
        3.load local fiass db 
        4.ask the question  to fiass db and get the relevant chunks
        5.make a prompt with question and chunk
        6.pass prompt to apen ai llm
        7.get response from oprn ai llm and return to postman or ui in json format
."""

from functools import lru_cache
import os
from pathlib import Path

from flask import Flask, jsonify, request
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

app = Flask(__name__)

@lru_cache(maxsize=1)
def get_vector_store():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    index_path = Path(__file__).resolve().parent / "faiss_index"
    return FAISS.load_local(
        str(index_path),
        embedding_model,
        allow_dangerous_deserialization=True,
    )


@app.route("/tcs", methods=["GET", "POST"])
def tcs_chatbot_api():
    if request.method == "GET":
        question = request.args.get("tcs_question", request.args.get("question"))
    else:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400
        question = data.get("tcs_question", data.get("question"))

    if not isinstance(question, str) or not question.strip():
        return (
            jsonify({"error": "A non-empty 'tcs_question' string is required."}),
            400,
        )

    vector_store = get_vector_store()
    relevant_chunks = vector_store.similarity_search(question.strip(), k=5)
    context = [chunk.page_content for chunk in relevant_chunks]

    def return_excerpts(warning):
        return (
            jsonify(
                {
                    "response": "\n\n".join(context),
                    "response_con": context,
                    "answer_type": "document_excerpts",
                    "warning": warning,
                }
            ),
            200,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return return_excerpts(
            "OPENAI_API_KEY is not set; returning matching document passages instead of an AI-generated answer."
        )

    prompt = (
        "Answer the user's question using only the context below. "
        "If the context does not contain the answer, say you could not find it "
        "in the TCS document.\n\n"
        f"Context:\n{chr(10).join(context)}\n\n"
        f"Question: {question.strip()}"
    )

    try:
        client = OpenAI(api_key=api_key)
        completion = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=prompt,
        )
    except AuthenticationError as error:
        app.logger.exception("OpenAI rejected the API key")
        return return_excerpts(
            "OpenAI rejected the API key; check OPENAI_API_KEY. Returning matching document passages instead."
        )
    except RateLimitError:
        app.logger.exception("OpenAI rate or quota limit reached")
        return return_excerpts(
            "OpenAI rate or quota limit reached; returning matching document passages instead."
        )
    except APIConnectionError:
        app.logger.exception("Could not connect to OpenAI")
        return return_excerpts(
            "Could not connect to OpenAI; returning matching document passages instead."
        )
    except APIStatusError as error:
        app.logger.exception("OpenAI request failed with status %s", error.status_code)
        return return_excerpts(
            f"OpenAI returned HTTP {error.status_code}; returning matching document passages instead."
        )
    except OpenAIError:
        app.logger.exception("OpenAI request failed")
        return return_excerpts(
            "OpenAI could not generate an answer; returning matching document passages instead."
        )

    return jsonify(
        {
            "response": completion.output_text,
            "response_con": context,
            "answer_type": "generated",
        }
    )

if __name__ == "__main__":
    app.run()