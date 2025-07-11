import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import csv
import re
from dotenv import load_dotenv
load_dotenv()
from chromadb.config import Settings


def get_embedding_model():
    return AzureOpenAIEmbeddings(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    )

def _split_and_load_docs(docs):
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200).split_documents(docs)

def create_chroma_from_documents(chunks, collection_name):
    persist_dir = f"chroma_dbs/{collection_name}"
    os.makedirs(persist_dir, exist_ok=True)
    settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)

    db = Chroma.from_documents(
    chunks,
    embedding=get_embedding_model(),
    collection_name=collection_name,
    persist_directory=persist_dir,
    client_settings=settings  # ✅ prevents use of sqlite
    )

    db.persist()

def load_vector_db(collection_name):
    persist_dir = f"chroma_dbs/{collection_name}"

    settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)

    db = Chroma(
    persist_directory=persist_dir,
    collection_name=collection_name,
    embedding_function=get_embedding_model(),
    client_settings=settings
    )
    st.session_state.vector_db = db

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
        os.remove(path)

        # Metadata
        meta = extract_policy_metadata(loaded[0].page_content)
        st.session_state.last_uploaded_metadata = {k: v.group(2) if v else "Not Found" for k, v in meta.items()}

    if docs:
        chunks = _split_and_load_docs(docs)
        create_chroma_from_documents(chunks, collection_name)
        st.toast("✅ Documents embedded & stored.")

def load_url_to_db(url, collection_name):
    loader = WebBaseLoader(url)
    docs = loader.load()
    chunks = _split_and_load_docs(docs)
    create_chroma_from_documents(chunks, collection_name)

def get_conversational_rag_chain(llm):
    retriever_chain = create_history_aware_retriever(
        llm,
        retriever=st.session_state.vector_db.as_retriever(),
        prompt=ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
            ("user", "Generate a query to fetch relevant policy context.")
        ])
    )

    response_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a policy evaluation assistant for deep tech startup programs.

Use the retrieved context and your knowledge to answer questions in a structured format:
1. Strengths  
2. Gaps  
3. Suggestions

Focus on: TRL support, IP facilitation, funding access, commercialization, long-term impact.

Context: {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}")
    ])

    doc_chain = create_stuff_documents_chain(llm, response_prompt)
    return create_retrieval_chain(retriever_chain, doc_chain)

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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
        prompt=retriever_prompt  # ✅ not ...
    )

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a policy assistant. Based on the context, provide:
(1) Strengths
(2) Gaps
(3) Suggestions

Focus on deep tech startup policy metrics like TRL, IP, funding, commercialization, and impact.

Context: {context}
"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}")
    ])

    stuff_chain = create_stuff_documents_chain(llm, answer_prompt)
    return create_retrieval_chain(retriever_chain, stuff_chain)

def load_txt_files_from_folder(folder_path, collection_name):
    from langchain_community.document_loaders import TextLoader
    import glob

    docs = []
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        loader = TextLoader(file_path)
        loaded = loader.load()
        docs += loaded

    if docs:
        chunks = _split_and_load_docs(docs)
        create_chroma_from_documents(chunks, collection_name)
        print(f"✅ Loaded {len(docs)} text files from {folder_path} into collection '{collection_name}'")
