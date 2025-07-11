import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

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
    st.markdown("### 📁 Policy Collection")
    # Load existing collections
    available_collections = os.listdir("faiss_dbs") if os.path.exists("faiss_dbs") else []
    new_policy_mode = st.checkbox("➕ Add New Policy Collection")

    if new_policy_mode:
        collection_name = st.text_input("Enter New Policy Name", key="new_policy_name")
        if collection_name and collection_name not in available_collections:
            st.success(f"✅ New collection ready: {collection_name}")
    else:
        collection_name = st.selectbox("Select Existing Policy", available_collections)

    # Load vector DB if name is set
    if collection_name:
        st.session_state.collection_name = collection_name
        load_vector_db(collection_name)
        st.success(f"✅ Using memory: {collection_name}")

    files = st.file_uploader("📄 Upload Docs", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    if st.button("📥 Upload & Index") and collection_name:
        load_doc_to_db(files, collection_name)

    url = st.text_input("🌐 Enter Policy URL", key="url_key")
    if st.button("🌐 Load URL") and url and collection_name:
        load_url_to_db(url, collection_name)

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
