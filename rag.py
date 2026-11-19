"""Core RAG pipeline: load docs -> chunk -> embed -> store -> retrieve -> generate."""
from pathlib import Path
from importlib import import_module
from typing import Any

# ---------------- Config ----------------
DOCS_DIR = Path("docs")          # put your PDFs / .txt files here
DB_DIR = "chroma_db"             # vector DB is saved here
COLLECTION = "knowledge_base"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"           # any model you have pulled in Ollama
CHUNK_SIZE = 800                 # characters per chunk
CHUNK_OVERLAP = 150
TOP_K = 4                        # chunks retrieved per question
MAX_DISTANCE = 0.75              # cosine distance cutoff (lower = more similar)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions ONLY using the provided "
    "context. If the answer is not in the context, say: "
    "\"I couldn't find that in the documents.\" Be concise and accurate. "
    "Mention the source file and page when relevant."
)

_embedder = None



def get_embedder() -> Any:
    global _embedder
    if _embedder is None:
        sentence_transformers = import_module("sentence_transformers")
        _embedder = sentence_transformers.SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_client():
    chromadb = import_module("chromadb")
    return chromadb.PersistentClient(path=DB_DIR)


def get_collection():
    return get_client().get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )


# ---------------- Loading & chunking ----------------
def load_documents():
    """Yield (source, page, text) for every PDF/TXT in DOCS_DIR."""
    DOCS_DIR.mkdir(exist_ok=True)
    for path in sorted(DOCS_DIR.rglob("*")):
        if path.suffix.lower() == ".pdf":
            pypdf = import_module("pypdf")
            reader = pypdf.PdfReader(str(path))
            for i, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    yield path.name, i, text
        elif path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                yield path.name, 1, text


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # try to cut at a sentence/word boundary
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


# ---------------- Indexing ----------------
def build_index() -> int:
    """(Re)build the vector index from everything in DOCS_DIR. Returns chunk count."""
    client = get_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, texts, metas = [], [], []
    for source, page, text in load_documents():
        for j, chunk in enumerate(chunk_text(text)):
            ids.append(f"{source}-p{page}-c{j}")
            texts.append(chunk)
            metas.append({"source": source, "page": page})

    if not texts:
        return 0

    embeddings = get_embedder().encode(texts, show_progress_bar=True).tolist()
    for i in range(0, len(texts), 500):  # add in batches
        col.add(
            ids=ids[i:i + 500],
            documents=texts[i:i + 500],
            embeddings=embeddings[i:i + 500],
            metadatas=metas[i:i + 500],
        )
    return len(texts)


# ---------------- Retrieval & generation ----------------
def retrieve(question: str, k: int = TOP_K):
    col = get_collection()
    if col.count() == 0:
        return []
    q_emb = get_embedder().encode([question]).tolist()
    res = col.query(query_embeddings=q_emb, n_results=k)
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        if dist <= MAX_DISTANCE:
            hits.append({"text": doc, "source": meta["source"],
                         "page": meta["page"], "distance": dist})
    return hits


def build_messages(question: str, hits, history=None):
    context = "\n\n".join(
        f"[Source: {h['source']}, page {h['page']}]\n{h['text']}" for h in hits
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in (history or [])[-6:]:  # last few turns for follow-ups
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}",
    })
    return messages


def stream_answer(question: str, hits, history=None):
    """Yield answer tokens from the local LLM."""
    if not hits:
        yield "I couldn't find that in the documents."
        return
    messages = build_messages(question, hits, history)
    ollama = import_module("ollama")
    for part in ollama.chat(model=LLM_MODEL, messages=messages, stream=True):
        yield part["message"]["content"]