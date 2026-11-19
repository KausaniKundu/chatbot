from importlib import import_module

try:
    st = import_module("streamlit")
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Streamlit is required. Install it with: python -m pip install streamlit"
    ) from exc

import rag

st.set_page_config(page_title="Document Q&A Chatbot", page_icon="📚")
st.title("📚 Document Q&A Chatbot")
st.caption("Ask questions about your documents. Answers come only from your files.")

# ---------------- Sidebar: upload + index ----------------
with st.sidebar:
    st.header("Knowledge base")
    uploads = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if uploads:
        rag.DOCS_DIR.mkdir(exist_ok=True)
        for f in uploads:
            (rag.DOCS_DIR / f.name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploads)} file(s) to {rag.DOCS_DIR}/")

    if st.button("Build / rebuild index", type="primary"):
        with st.spinner("Reading, chunking and embedding documents..."):
            n = rag.build_index()
        if n:
            st.success(f"Indexed {n} chunks.")
        else:
            st.warning("No documents found. Add files to the docs/ folder.")

    try:
        st.metric("Chunks in index", rag.get_collection().count())
    except Exception:
        pass

    top_k = st.slider("Chunks to retrieve", 1, 8, rag.TOP_K)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Chat ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"**{s['source']}** (page {s['page']})")
                    st.caption(s["text"][:400] + "...")

question = st.chat_input("Ask a question about your documents...")
if question:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        hits = rag.retrieve(question, k=top_k)
        answer = st.write_stream(rag.stream_answer(question, hits, history))
        if hits:
            with st.expander("Sources"):
                for s in hits:
                    st.markdown(f"**{s['source']}** (page {s['page']})")
                    st.caption(s["text"][:400] + "...")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": hits}
    )