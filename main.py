import os, shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from rag import ingest, ask

load_dotenv()

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    path = f"uploads/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks = ingest(path)
    return {"ok": True, "chunks": chunks, "filename": file.filename}

@app.post("/ask")
async def ask_question(body: dict):
    question = body.get("question", "")
    if not question:
        return JSONResponse({"error": "質問が空です"}, status_code=400)
    answer = ask(question)
    return {"answer": answer}
