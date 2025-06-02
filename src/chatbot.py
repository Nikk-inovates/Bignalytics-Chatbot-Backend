import os
import requests
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError("❌ OPENROUTER_API_KEY not found in .env file.")

MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

DEBUG = False

def setup_deepseek():
    return MODEL_NAME

def ask_question(model_name, context_chunks, user_question):
    if not context_chunks:
        return "⚠️ No context available to answer the question."

    context = "\n\n".join(context_chunks)
    prompt = (
        "You are a helpful AI assistant. Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_question}"
    )

    if DEBUG:
        print("Prompt to DeepSeek:\n", prompt)

    try:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that answers based on given context."},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(
            url=OPENROUTER_API_URL,
            headers=HEADERS,
            json=payload
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ DeepSeek API error {response.status_code}: {response.text}"

    except Exception as e:
        return f"❌ Unexpected error: {e}"

def search_chunks(embedding_model, index, chunks, query, top_k=3):
    if not query.strip():
        return []

    try:
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        D, I = index.search(query_embedding, top_k)
        top_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
        return top_chunks

    except faiss.FaissException as faiss_err:
        print(f"❌ FAISS error during search: {faiss_err}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error in search_chunks: {e}")
        return []

def log_chat_to_history(question, answer, filename="logs/chat_history.json"):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        chat_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer
        }

        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                history = []
        else:
            history = []

        history.append(chat_entry)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

    except Exception as e:
        print(f"⚠️ Failed to log chat history: {e}")

def log_feedback_to_file(feedback_data, filename="logs/feedback_logs.jsonl"):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_data) + "\n")
    except Exception as e:
        print(f"⚠️ Failed to log feedback: {e}")
