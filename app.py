import os
import streamlit as st
import sys, pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# --- Configuración página ---
st.set_page_config(page_title="nicorl-bot RAG", page_icon="🤖")
st.title("🤖 Chatbot local con Hugging Face y FAISS")

# --- Embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persist_dir = "chromadb"

# --- Crear o cargar vectorstore ---
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    # Cargar documentos
    docs = []
    doc_folder = "docs"
    for file in os.listdir(doc_folder):
        if file.endswith(".txt"):
            with open(os.path.join(doc_folder, file), "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read()))

    # Dividir en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Crear vectorstore
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    vectorstore.persist()

# --- Modelo Hugging Face ---
model_id = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": device})
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device=0 if device=="cuda" else -1)
llm = HuggingFacePipeline(pipeline=pipe)

# --- Configurar RAG ---
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# --- Historial chat ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Entrada usuario
user_input = st.text_input("Escribe tu pregunta:")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    respuesta = qa.run(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": respuesta})

# Mostrar historial
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**Tú:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
