import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Carpeta con documentos
doc_folder = "docs"
docs = []

for file in os.listdir(doc_folder):
    if file.endswith(".txt"):
        with open(os.path.join(doc_folder, file), "r", encoding="utf-8") as f:
            docs.append(Document(page_content=f.read()))

# Dividir en fragmentos (chunks)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Crear embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Crear FAISS vectorstore
vectorstore = FAISS.from_documents(chunks, embeddings)

# Guardar índice para futuras ejecuciones
vectorstore.save_local("faiss_index")
