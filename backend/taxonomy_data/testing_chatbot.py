import os
import json
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain




# -----------------------------
# Config
# -----------------------------
st.set_page_config(layout="wide")

os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["OPENAI_API_KEY"]

JSONL_PATH = "data/merged_taxonomic_chunks.jsonl"
DB_FAISS_PATH = "vectorstore/db_faiss_taxo"

# -----------------------------
# Load JSONL as LangChain Docs
# -----------------------------
def load_jsonl_documents(jsonl_path: str):
    documents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            text = obj.get("text", "").strip()
            if not text:
                continue

            metadata = {
                "species_name": obj.get("species_name"),
                "genus": obj.get("genus"),
                "chunk_type": obj.get("chunk_type"),
                "title": obj.get("title"),
                "section": obj.get("section"),
                "authority": obj.get("authority"),
                "year": obj.get("year"),
                "source_file": obj.get("source_file"),
            }

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )
    return documents

# -----------------------------
# Vectorstore (load or build)
# -----------------------------
@st.cache_resource
def get_vectorstore():
    embeddings = OpenAIEmbeddings()

    if Path(DB_FAISS_PATH).exists():
        return FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    docs = load_jsonl_documents(JSONL_PATH)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(DB_FAISS_PATH)
    return vectorstore

vectorstore = get_vectorstore()

# -----------------------------
# RAG Chain
# -----------------------------
llm = ChatOpenAI(
    temperature=0.0,
    model_name="gpt-4o"
)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 25}
    ),
    return_source_documents=True
)

# -----------------------------
# Chat logic
# -----------------------------
def conversational_chat(query):
    result = chain({
        "question": query,
        "chat_history": st.session_state["history"]
    })
    st.session_state["history"].append((query, result["answer"]))
    return result["answer"]

# -----------------------------
# Session state
# -----------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if "generated" not in st.session_state:
    st.session_state["generated"] = [
        "Hello! Ask me anything about Glyceridae 🤗"
    ]

if "past" not in st.session_state:
    st.session_state["past"] = ["Hey! 👋"]

# -----------------------------
# UI
# -----------------------------
st.title("TAXObot")

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown(
        """
        <div style="background-color:#f0f0f0;padding:20px;border-radius:10px;">
        <p>
        <b>TAXObot</b> is an AI assistant designed for Marine Taxonomists.  
        It uses taxonomic descriptions of <b>Glyceridae (Polychaeta)</b> from Indian waters.
        You can:
        <ul>
            <li>Identify species from morphological characters</li>
            <li>Retrieve characters for a given species</li>
            <li>Ask general taxonomic questions</li>
        </ul>
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input(
                "Query:",
                placeholder="Describe features or ask about a species",
                key="input"
            )
            submit_button = st.form_submit_button("Send")

        if submit_button and user_input:
            if not is_query_valid(user_input):
                st.stop()

            output = conversational_chat(user_input)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        with response_container:
            for i in range(len(st.session_state["generated"])):
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:flex-end;margin-bottom:10px;">
                        <div style="background:#e6f0ff;padding:10px;border-radius:5px;max-width:45%;">
                        {st.session_state["past"][i]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:flex-start;margin-bottom:20px;">
                        <div style="background:#f8f9fa;padding:10px;border-radius:5px;max-width:60%;">
                        {st.session_state["generated"][i]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
