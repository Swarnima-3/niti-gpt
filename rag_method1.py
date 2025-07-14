import os
import shutil
import streamlit as st
import csv
import re
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, WebBaseLoader
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def get_embedding_model():
    return AzureOpenAIEmbeddings(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    )

def _split_and_load_docs(docs):
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_documents(docs)

def create_faiss_from_documents(chunks, collection_name):
    persist_dir = f"faiss_dbs/{collection_name}"
    os.makedirs(persist_dir, exist_ok=True)
    db = FAISS.from_documents(chunks, embedding=get_embedding_model())
    db.save_local(persist_dir)

def load_vector_db(collection_name):
    persist_dir = f"faiss_dbs/{collection_name}"
    index_path = os.path.join(persist_dir, "index.faiss")
    embedding = get_embedding_model()
    if os.path.exists(index_path):
        db = FAISS.load_local(persist_dir, embeddings=embedding, allow_dangerous_deserialization=True)
        st.session_state.vector_db = db
    else:
        st.warning(f"⚠️ Vector DB for '{collection_name}' not found. Upload documents first.")
        st.session_state.vector_db = None

def extract_policy_metadata(text):
    return {
        "scheme_name": re.search(r"(Scheme Name|Title):\s*(.+)", text, re.I),
        "ministry": re.search(r"(Ministry|Department):\s*(.+)", text, re.I),
        "budget": re.search(r"(Budget|Funds):\s*Rs\.\s?([\d,]+)", text, re.I),
        "launch_date": re.search(r"(Launch Date|Start Date):\s*([\d-]+)", text, re.I),
    }

def load_doc_to_db(files, collection_name):
    docs = []
    for file in files:
        os.makedirs("temp_docs", exist_ok=True)
        path = os.path.join("temp_docs", file.name)
        with open(path, "wb") as f:
            f.write(file.getvalue())

        if file.type == "application/pdf":
            loader = PyPDFLoader(path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            loader = TextLoader(path)

        loaded = loader.load()
        docs += loaded
        if loaded:
            meta = extract_policy_metadata(loaded[0].page_content)
            st.session_state.last_uploaded_metadata = {k: v.group(2) if v else "Not Found" for k, v in meta.items()}

    shutil.rmtree("temp_docs", ignore_errors=True)

    if docs:
        chunks = _split_and_load_docs(docs)
        create_faiss_from_documents(chunks, collection_name)
        st.toast("✅ Documents embedded & stored.")

def load_url_to_db(url, collection_name):
    loader = WebBaseLoader(url)
    docs = loader.load()
    chunks = _split_and_load_docs(docs)
    create_faiss_from_documents(chunks, collection_name)

def get_conversational_rag_chain(llm):
    if not st.session_state.vector_db:
        raise ValueError("Vector DB not loaded — cannot build RAG chain.")

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Generate a search query based on this conversation.")
    ])

    retriever_chain = create_history_aware_retriever(
        llm,
        retriever=st.session_state.vector_db.as_retriever(),
        prompt=retriever_prompt
    )

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a policy assistant for helping create the Ministry Of Electronics & IT policies for startups in India.

Based on the following context, provide:
(1) Strengths
(2) Gaps
(3) Suggestions

Context: {context}
"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}")
    ])

    stuff_chain = create_stuff_documents_chain(llm, answer_prompt)
    return create_retrieval_chain(retriever_chain, stuff_chain)

def stream_llm_response(llm, messages):
    full = ""
    for chunk in llm.stream(messages):
        full += chunk.content
        yield chunk.content
    st.session_state.messages.append({"role": "assistant", "content": full})

def stream_llm_rag_response(llm, messages):
    if not st.session_state.get("vector_db"):
        yield "⚠️ Vector database not loaded. Please upload a document or select a collection first."
        return
    rag_chain = get_conversational_rag_chain(llm)
    full = ""
    for chunk in rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        full += chunk
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": full})

def log_feedback(user_input, model_output, rating, comment, file="feedback_log.csv"):
    with open(file, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([user_input, model_output, rating, comment])

def load_txt_files_from_folder(folder_path, collection_name):
    import glob
    docs = []
    file_paths = glob.glob(os.path.join(folder_path, "*.txt")) + glob.glob(os.path.join(folder_path, "*.md"))
    if not file_paths:
        st.warning(f"⚠️ No .txt or .md files found in {folder_path}")
        return
    for file_path in file_paths:
        loader = TextLoader(file_path)
        docs += loader.load()
    if docs:
        chunks = _split_and_load_docs(docs)
        create_faiss_from_documents(chunks, collection_name)
        print(f"✅ Loaded {len(docs)} documents into collection '{collection_name}'")
