from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime
import os
import shutil
import logging
import re
import textwrap

from src.load_pdf import load_pdf_text
from src.embed_text import split_text, embed_chunks, save_faiss_index, load_faiss_index
from src.chatbot import (
    ask_question as ask_llm_question,
    search_chunks,
    log_chat_to_history,
    log_feedback_to_file,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PDF_DIR = "data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"
FAISS_INDEX_PATH = "embeddings/faiss_index.faiss"  # Adjust path as per your save_faiss_index implementation
CHUNKS_SAVE_PATH = "embeddings/chunks.pkl"        # If you save chunks separately; adjust accordingly
PDF_PATH = os.path.join(PDF_DIR, "knowledge.pdf")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs("embeddings", exist_ok=True)  # Ensure embeddings folder exists

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend/dist"))
if os.path.isdir(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")
    app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIST, "static")), name="static")


def clean_text(text: str) -> str:
    """
    Clean raw LLM output by:
    - Removing markdown formatting characters like **, *, #
    - Replacing dashes used as list markers with bullets (•)
    - Preserving hyphens inside words (e.g. real-world)
    - Normalizing multiple newlines
    """
    # Remove markdown bold/italic
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    # Remove markdown headers
    text = re.sub(r"#+\s*(.*)", r"\1", text)
    # Remove markdown links
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Replace dashes only when used as list markers at line starts with bullet points
    # (?m) enables multiline mode to match start of each line
    text = re.sub(r"(?m)^\s*-\s+", "• ", text)

    # Normalize multiple blank lines to two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def format_bullet_hanging_indent(text: str, max_width=70) -> str:
    """
    Format bullets so multiline bullet points align properly with hanging indent.
    Uses textwrap to ensure wrapped lines start under the text, not the bullet.
    """
    lines = []
    for paragraph in text.split('\n'):
        # Only process bullet points that start with "• "
        if paragraph.startswith("• "):
            bullet = "• "
            indent = " " * (len(bullet)+1)
            formatted = textwrap.fill(
                paragraph[len(bullet):],  # Remove bullet for wrapping
                width=max_width,
                initial_indent=bullet,
                subsequent_indent=indent
            )
            lines.append(formatted)
        else:
            lines.append(paragraph)
    return "\n".join(lines)


@app.on_event("startup")
def startup_event():
    try:
        # Check if FAISS index file exists, else create it from local PDF
        if not os.path.exists(FAISS_INDEX_PATH):
            logger.info("FAISS index not found. Creating index from local PDF...")

            # Load PDF text
            text = load_pdf_text(PDF_PATH)
            logger.info(f"Loaded PDF with {len(text)} characters")

            # Split text into chunks
            chunks = split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")

            # Embed chunks and build FAISS index
            model, index, embeddings, chunk_list = embed_chunks(chunks, EMBEDDING_MODEL_NAME)

            # Save FAISS index and chunk list
            save_faiss_index(index, chunk_list)
            logger.info("FAISS index and chunks saved successfully.")

        else:
            logger.info("FAISS index found. Skipping creation on startup.")

    except Exception as e:
        logger.error(f"Failed to create FAISS index on startup: {e}")


@app.get("/")
def root():
    return {"message": "🚀 PDF Chatbot API is running!"}


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(PDF_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = load_pdf_text(file_path)
        chunks = split_text(text)
        model, index, embeddings, chunk_list = embed_chunks(chunks)
        save_faiss_index(index, chunk_list)

        return {"message": "✅ PDF uploaded and processed successfully!"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"❌ Failed to process PDF: {str(e)}"})


@app.post("/ask-question/")
async def ask_question_api(question: str = Form(...)):
    try:
        index, chunk_list = load_faiss_index()

        if not index or not chunk_list:
            return JSONResponse(
                status_code=400,
                content={"error": "❌ No index or chunks found. Upload a PDF first."}
            )

        top_chunks = search_chunks(embedding_model, index, chunk_list, question)

        prompt_instructions = (
            "Respond clearly and professionally in fluent, natural English. "
            "Use normal sentence case (capitalize only the first letter of each sentence and proper nouns). "
            "Use proper punctuation and spacing. "
            "Format lists with bullet points (•) followed by a tab or space. "
            "If a bullet point wraps onto more than one line, indent all lines after the first so that they align with the start of the text, not the bullet (hanging indent). "
            "Use section headings in plain sentence case without markdown, surrounded by blank lines. "
            "Avoid markdown syntax, emojis, filler, informal language, or feedback phrases. "
            "Make the response look like well-structured business or marketing copy — polished, clear, and easy to read."
        )

        context_text = "\n\n".join(top_chunks)
        full_prompt = f"{prompt_instructions}\n\nContext:\n{context_text}\n\nQuestion:\n{question}"

        raw_response = ask_llm_question(LLM_MODEL_NAME, top_chunks, full_prompt)

        clean_response = clean_text(raw_response)

        # Hanging indent formatting added here:
        clean_response = format_bullet_hanging_indent(clean_response)

        log_chat_to_history(question, clean_response)

        return {"answer": clean_response}

    except Exception as e:
        logger.error(f"Failed to answer question: {e}")
        return JSONResponse(status_code=500, content={"error": f"❌ Failed to get answer: {str(e)}"})


@app.post("/log-feedback/")
async def log_feedback_api(request: Request):
    try:
        feedback_data = await request.json()
        feedback_data["timestamp"] = datetime.utcnow().isoformat()
        log_feedback_to_file(feedback_data)
        return {"message": "✅ Feedback saved successfully."}
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return JSONResponse(status_code=500, content={"error": f"❌ Failed to save feedback: {str(e)}"})


@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    index_path = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    else:
        return JSONResponse(status_code=404, content={"error": "Frontend not found."})
