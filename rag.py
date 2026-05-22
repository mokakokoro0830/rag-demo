import os, shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

load_dotenv()

# インメモリ保持（Railwayはディスク書き込み不可のため）
_db = None

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""あなたは不動産文書Q&Aアシスタントです。
以下の【文書抜粋】だけを根拠に質問に答えてください。

【文書抜粋】
{context}

【質問】
{question}

ルール：
- 文書に記載がある場合は、その内容を分かりやすく説明してください。
- 文書に記載がない場合は「この文書には該当する記載がありませんでした。」と答えてください。
- 推測や一般論は付け加えないでください。

回答："""
)

def ingest(pdf_path: str):
    """PDFを読み込んでインメモリベクトルDBに保存"""
    global _db
    print(f"📄 読み込み中: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    print(f"✂️  {len(chunks)}チャンクに分割")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # persist_directory を指定しない → インメモリ動作
    _db = Chroma.from_documents(chunks, embeddings)
    print(f"✅ インメモリDB構築完了")
    return len(chunks)

def ask(question: str) -> str:
    """質問に答える"""
    global _db
    if _db is None:
        return "まだPDFがアップロードされていません。"

    retriever = _db.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    result = chain.invoke({"query": question})
    return result["result"]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("使い方:")
        print("  python rag.py ingest <PDFパス>  ← PDFを読み込む")
        print("  python rag.py ask <質問>        ← 質問する")
    elif sys.argv[1] == "ingest":
        ingest(sys.argv[2])
    elif sys.argv[1] == "ask":
        answer = ask(sys.argv[2])
        print(f"\n💬 回答: {answer}")
