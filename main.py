import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()
import requests

from rag_method1 import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
    log_feedback,
    load_vector_db,
)

load_dotenv()

# Azure credentials
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Init model
llm_stream = AzureChatOpenAI(
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    deployment_name=AZURE_DEPLOYMENT,
    openai_api_version=AZURE_VERSION,
    temperature=0.3,
    streaming=True,
)

def download_github_policy_docs(repo_url, local_dir="data/txt_policies"):
    os.makedirs(local_dir, exist_ok=True)

    api_url = repo_url.replace("github.com", "api.github.com/repos").replace("/tree/", "/contents/")
    response = requests.get(api_url)
    files = response.json()

    collection_names = []

    for file in files:
        if file["name"].endswith((".txt", ".md")):
            file_url = file["download_url"]
            local_path = os.path.join(local_dir, file["name"])

            if not os.path.exists(local_path):
                content = requests.get(file_url).content
                with open(local_path, "wb") as f:
                    f.write(content)

            # Strip extension for collection name
            collection_name = file["name"].rsplit(".", 1)[0]
            collection_names.append(collection_name)

    return collection_names

# Streamlit UI
st.set_page_config(page_title="PolicyGPT", layout="wide")
st.title("NītiGPT")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None    

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Policy Collection (from GitHub)")

    github_folder = "https://github.com/Swarnima-3/niti-gpt/tree/main/data/txt_policies"
    available_collections = download_github_policy_docs(github_folder)
    available_collections = ["All"] + sorted(available_collections)

    selected = st.selectbox("Select Policy Collection", available_collections)

    if selected == "All":
        from langchain_community.vectorstores import FAISS
        all_dbs = []
        for name in available_collections[1:]:
            path = f"faiss_dbs/{name}"
            if not os.path.exists(path):
                from rag_methods import load_txt_files_from_folder
                load_txt_files_from_folder("data/txt_policies", name)

            db = FAISS.load_local(path, embeddings=get_embedding_model(), allow_dangerous_deserialization=True)
            all_dbs.append(db)

        if all_dbs:
            merged = all_dbs[0]
            for db in all_dbs[1:]:
                merged.merge_from(db)
            st.session_state.vector_db = merged
            st.session_state.collection_name = "All"
            st.success("✅ Merged all GitHub policies.")
        else:
            st.warning("⚠️ No valid collections found.")
    else:
        from rag_methods import load_txt_files_from_folder
        load_txt_files_from_folder("data/txt_policies", selected)
        load_vector_db(selected)
        st.session_state.collection_name = selected
        st.success(f"✅ Loaded: {selected}")

    # Upload and URL input stay the same
    st.markdown("---")
    st.markdown("### 📄 Upload Your Own Policy File")
    uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    if uploaded_files:
        temp_collection = f"user_upload_{st.session_state.session_id}"
        load_doc_to_db(uploaded_files, temp_collection)
        st.session_state.collection_name = temp_collection
        load_vector_db(temp_collection)
        st.success("✅ Uploaded policy indexed.")

    st.markdown("---")
    st.markdown("### 🌐 Analyze Policy from URL")
    url = st.text_input("Enter policy URL", key="url_key")
    if st.button("🌐 Load URL") and url:
        temp_collection = f"url_upload_{st.session_state.session_id}"
        load_url_to_db(url, temp_collection)
        st.session_state.collection_name = temp_collection
        load_vector_db(temp_collection)
        st.success("✅ URL policy indexed.")

    st.toggle("Use RAG", value=True, key="use_rag")
    st.button("🧹 Clear Chat", on_click=lambda: st.session_state.messages.clear())


# Main chat
from rag_method1 import load_txt_files_from_folder

# Example usage
collection_name = "DeepTech_Startup_Policies"
folder_path = "data/txt_policies"  # Your local folder path

load_txt_files_from_folder(folder_path, collection_name)
load_vector_db(collection_name)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the uploaded policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]
        if st.session_state.use_rag:
            st.write_stream(stream_llm_rag_response(llm_stream, messages))
        else:
            st.write_stream(stream_llm_response(llm_stream, messages))

        # Feedback
        st.divider()
        feedback = st.radio("Was this helpful?", ["👍", "👎"], horizontal=True)
        comment = st.text_input("Suggestions?")
        if st.button("📩 Submit Feedback"):
            log_feedback(prompt, st.session_state.messages[-1]["content"], feedback, comment)
            st.success("Thanks for your feedback!")

#comment
