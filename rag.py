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

# PDF が読み込まれている場合のプロンプト（PDF抜粋 + 物件コンテキストを参照）
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""あなたは不動産物件Q&Aアシスタントです。

【PDF文書の抜粋】
{context}

【質問（物件情報を含む場合があります）】
{question}

回答ルール：
- 上記の PDF 抜粋に記載がある内容は、それを優先して分かりやすく説明してください。
- 質問の中に物件情報（ペット・賃料・設備など）が含まれている場合はその情報も参照して答えてください。
- どちらにも記載がない場合は「詳細な物件資料のご確認が必要です。」と答えてください。
- 推測や一般論は付け加えないでください。
- マークダウン記法（**、##、- など）は使わず、普通のテキストで回答してください。

回答："""
)

# PDF なしの場合のプロンプト（質問内の物件コンテキストだけで答える）
PROMPT_NO_PDF = PromptTemplate(
    input_variables=["question"],
    template="""あなたは不動産物件Q&Aアシスタントです。

{question}

上記の物件情報をもとに質問に答えてください。
物件情報に記載がない場合は「詳細は担当スタッフにお問い合わせください。」と答えてください。
推測や一般論は付け加えないでください。
マークダウン記法（**、##、- など）は使わず、普通のテキストで回答してください。

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
    _db = Chroma.from_documents(chunks, embeddings)
    print(f"✅ インメモリDB構築完了")
    return len(chunks)

def ask(question: str) -> str:
    """質問に答える"""
    global _db
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if _db is None:
        # PDF未アップロード → 質問内の物件コンテキストでLLMが直接回答
        chain = PROMPT_NO_PDF | llm
        result = chain.invoke({"question": question})
        return result.content

    # PDF 読込済み → RAG で PDF 抜粋を引いて回答
    retriever = _db.as_retriever(search_kwargs={"k": 4})
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
