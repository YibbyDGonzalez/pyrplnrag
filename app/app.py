import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
 
 
# ============================
# 0. CONFIGURACIÓN
# ============================
 
ARTICLES_PATH = "data/processed/articulos_total.csv"
EMBEDDINGS_PATH = "data/processed/models/embeddings_total.npy"
FAISS_PATH = "data/processed/models/faiss_index_total.bin"
 
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
 
 
# ============================
# 1. CARGA DE MODELOS Y DATOS
# ============================
 
@st.cache_resource
def load_resources():
    df = pd.read_csv(ARTICLES_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    index = faiss.read_index(FAISS_PATH)
 
    encoder = SentenceTransformer(EMBED_MODEL_NAME)
 
    groq_api_key = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", None))
    if groq_api_key is None:
        raise ValueError("No se encontró GROQ_API_KEY en variables de entorno ni en st.secrets.")
 
    groq_client = Groq(api_key=groq_api_key)
 
    return df, embeddings, index, encoder, groq_client
 
 
df, embeddings, index, encoder, groq_client = load_resources()
 
 
# ============================
# 2. RAG
# ============================
 
def buscar_articulos(query, top_k=4):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(q_emb).astype("float32"), top_k)
 
    resultados = df.iloc[idxs[0]].copy()
    resultados["score"] = scores[0]
    return resultados
 
 
def call_groq(prompt: str) -> str:
 
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un experto en Medicina Basada en la Evidencia (MBE). "
                    "Tu tarea es responder preguntas clínicas de manera clara, precisa y concisa, "
                    "utilizando ÚNICAMENTE la información proporcionada en los textos de contexto. "
 
                    "Reglas estrictas: "
                    "- No uses conocimiento externo. "
                    "- No inventes información. "
                    "- Si la respuesta no está en el contexto, responde: "
                    "'No hay suficiente información en los textos proporcionados para responder la pregunta.' "
                    "- Prioriza información relevante y directamente relacionada con la pregunta. "
                    "- Resume y sintetiza, no copies textualmente a menos que sea necesario. "
 
                    "Formato de respuesta: "
                    "- Respuesta clara y directa. "
                    "- Si aplica, incluye un breve soporte citando el fragmento del texto."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
 
    return completion.choices[0].message.content
 
 
def rag_responder(query: str, top_k: int = 4):
 
    articulos = buscar_articulos(query, top_k=top_k)
 
    contexto = ""
    for _, row in articulos.iterrows():
        contexto += f"ARTÍCULO {row['id_articulo']} - {row['titulo']}\n{row['texto']}\n\n"
 
    prompt = f"""
PREGUNTA:
{query}
 
PAGINA Y LIBRO RELEVANTES (EXTRACTOS DE INFORMACIÓN):
{contexto}
 
Instrucciones:
- Responde de manera clara y pedagógica.
- Indica explícitamente qué paginas usas.
- Si no hay información suficiente, respóndelo.
"""
 
    respuesta = call_groq(prompt)
    return respuesta
 
 
 
# ============================
# 3. INTERFAZ STREAMLIT
# ============================
 
st.set_page_config(page_title="Asistente educativo Javeriana", page_icon="🧪")
 
st.title("🔬 Asistente MBE • Facultad de medicina PUJ")
 
# --- Párrafo objetivo ---
with st.expander("ℹ️ ¿Qué es este asistente?", expanded=False):
    st.markdown(
        """
        Este asistente ha sido diseñado como una herramienta de apoyo académico para estudiantes
        de medicina que cursan la materia de Medicina Basada en la Evidencia. A través de un sistema
        de recuperación de información, permite consultar conceptos, metodologías y principios
        fundamentales de la MBE, facilitando la comprensión y aplicación de sus herramientas
        en el contexto clínico y académico.
 
        Este asistente se basa en literatura académica reconocida en Medicina Basada en la Evidencia, incluyendo:
       
        🔹Painless Evidence-Based Medicine — Antonio L. Dans, Leonila F. Dans, Maria Asuncion A. Silvestre
        🔹Users' Guides to the Medical Literature: A Manual for Evidence-Based Medicine — Gordon Guyatt, Drummond Rennie, Maureen O. Meade, Deborah J. Cook
        """
    )
 
st.divider()
 
# --- Preguntas sugeridas ---
st.subheader("Preguntas sugeridas:")
col1, col2, col3 = st.columns(3)
 
q1 = "¿Qué es la medicina basada en la evidencia y cuáles son sus pasos principales?"
q2 = "¿Cuál es la diferencia entre un estudio observacional y un ensayo clínico aleatorizado?"
q3 = "¿Qué significa el nivel de evidencia de un estudio y cómo se clasifica?"
 
if col1.button(q1):
    st.session_state["pregunta"] = q1
 
if col2.button(q2):
    st.session_state["pregunta"] = q2
 
if col3.button(q3):
    st.session_state["pregunta"] = q3
 
# --- Input de pregunta ---
pregunta = st.text_input(
    "Haz una pregunta sobre tratamientos, estudios, diagnósticos, etc:",
    value=st.session_state.get("pregunta", "")
)
 
if st.button("Consultar"):
    if pregunta.strip() == "":
        st.warning("Por favor ingresa una pregunta.")
    else:
        with st.spinner("Generando respuesta..."):
            respuesta = rag_responder(pregunta)
 
        st.subheader("📌 Respuesta")
        st.write(respuesta)
 
# --- Créditos al pie ---
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.82em; padding: 10px 0'>
        <b>Autores:</b> Yibby Gonzalez · Juan Ruiz &nbsp;|&nbsp;
        <b>Profesores:</b> Juan Pablo Páramo · Fabián Gil <br>
        📧 Contacto: <a href='mailto:gonzalez_yibby@javeriana.edu.co' style='color: gray;'>gonzalez_yibby@javeriana.edu.co</a>
    </div>
    """,
    unsafe_allow_html=True
)