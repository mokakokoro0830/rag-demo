import os, shutil, tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from rag import ingest, ask

load_dotenv()

# APIキーの末尾の改行・空白を除去（Railway環境での貼り付けミス対策）
if os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", tempfile.gettempdir())

@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        with open("index.html", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>RAG API is running</h1>"

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        chunks = ingest(path)
        return {"ok": True, "chunks": chunks, "filename": file.filename}
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "detail": traceback.format_exc()}, status_code=500)

@app.post("/ask")
async def ask_question(body: dict):
    question = body.get("question", "")
    if not question:
        return JSONResponse({"error": "質問が空です"}, status_code=400)
    answer = ask(question)
    return {"answer": answer}
