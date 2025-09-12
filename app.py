import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Configuración página ---
st.set_page_config(page_title="nicorl-bot RAG", page_icon="🤖")
st.title("🤖 Chatbot local con Hugging Face y FAISS")

# --- Cargar embeddings y FAISS ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings)

# --- Modelo local Hugging Face ---
model_id = "tiiuae/falcon-7b-instruct"  # modelo gratuito, descarga local
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")  # GPU recomendado
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
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
