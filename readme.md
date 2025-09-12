# nicorl-bot RAG con Hugging Face y Streamlit


🤖 Chatbot RAG que responde a preguntas usando **tus propios documentos**.  
No requiere OpenAI, funciona 100% con modelos gratuitos de Hugging Face.


---


## Estructura del proyecto


```


├── app.py                # interfaz Streamlit
├── only-one.py           # script de preparación de embeddings
├── docs/                 # documentos de referencia (txt)
├── faiss\_index/          # índice FAISS generado
├── requirements.txt
└── README.md


````


---


## Requisitos


- Python ≥ 3.10
- GPU opcional (CPU funciona pero más lento)
- Token gratuito de Hugging Face (para descargar el modelo)


Instalar dependencias:


```bash
pip install -r requirements.txt
````


---


## Preparación del índice FAISS


Si no lo has hecho:


```bash
python only-one.py
```


Esto creará la carpeta `faiss_index/` con los embeddings de tus documentos.


---


## Configuración de secretos


* Crea un token en Hugging Face: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
* En **Streamlit Cloud** o **Hugging Face Spaces**, añade `HF_TOKEN` como secreto.


---


## Ejecución local


```bash
streamlit run app.py
```


---


## Despliegue en la nube


### Opción A: Streamlit Cloud


1. Subir todo el proyecto a GitHub (incluyendo `faiss_index/` y `docs/`).
2. Crear un proyecto en Streamlit Cloud y conectar el repo.
3. Configurar `HF_TOKEN` en *Secrets Manager*.
4. Deploy automático → obtienes una URL pública.


### Opción B: Hugging Face Spaces


1. Crear un Space con framework Streamlit.
2. Subir repo o ficheros directamente.
3. Configurar `HF_TOKEN` en *Settings → Secrets*.
4. Deploy automático → URL pública en Hugging Face.


---


## Uso


* Abrir la URL pública o ejecutar localmente.
* Escribir la pregunta en la caja de texto.
* El bot responde usando los documentos de `docs/`.
* Historial de chat guardado mientras dure la sesión.


````


---


## 2️⃣ Pasos prácticos después de `only-one.py`


1. **Verifica que `faiss_index/` existe**  
   - Debe contener archivos tipo `index.faiss` y `store.pkl`.  


2. **Sube todo a GitHub**  
   - Incluye:  
     - `app.py`  
     - `only-one.py`  
     - `docs/`  
     - `faiss_index/`  
     - `requirements.txt`  


   ```bash
   git init
   git add .
   git commit -m "RAG bot ready"
   git branch -M main
   git remote add origin <tu-repo-url>
   git push -u origin main
````


3. **Elige plataforma de despliegue**


   **Streamlit Cloud (recomendado):**


   * Conecta tu repo GitHub.
   * Añade secret `HF_TOKEN`.
   * Deploy → URL pública.


   **Hugging Face Spaces (alternativa):**


   * Crear Space, seleccionar Streamlit.
   * Subir repo o ficheros.
   * Configurar secret `HF_TOKEN`.
   * Deploy → URL pública.


4. **Probar el chatbot**


   * Abre la URL o ejecuta local:


     ```bash
     streamlit run app.py
     ```
   * Escribe preguntas → responde usando tus `.txt`.


5. **Mantenimiento / actualización**


   * Si agregas documentos nuevos → ejecutar `only-one.py` de nuevo → actualizar `faiss_index/` → push a GitHub → deploy automático.


-


