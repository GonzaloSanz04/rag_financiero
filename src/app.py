import streamlit as st
import requests
import os
from langchain_ollama import OllamaLLM

# IMPORTANTE: Los nombres deben coincidir con logic.py
from logic import get_portfolio_agent, classify_query, index_financial_documents

# --- BLOQUE DE CALENTAMIENTO ---
@st.cache_resource
def preload_model():
    """Envía una petición en blanco a Ollama al iniciar la app para cargar el modelo en RAM"""
    try:
        requests.post(
            "http://ollama:11434/api/generate", 
            json={"model": "qwen2.5:7b"}, 
            timeout=5
        )
    except Exception:
        pass
preload_model()

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente Financiero Dual", layout="wide")
st.title("📈 Asistente RAG: Análisis de Cartera")

# Panel lateral para la ingesta de documentos
with st.sidebar:
    st.header("📂 Ingesta de Documentos")
    uploaded_file = st.file_uploader("Sube un informe o folleto (PDF)", type="pdf")
    doc_name_input = st.text_input("Nombre del activo (ej. SXR8):")
    
    if st.button("Procesar e Indexar"):
        if uploaded_file and doc_name_input:
            with st.spinner("Aplicando triple fragmentación e indexando..."):
                temp_path = os.path.join("data", uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Usamos el nombre correcto de la función
                    num_chunks = index_financial_documents(temp_path, doc_name_input)
                    st.success(f"¡Éxito! {num_chunks} fragmentos indexados en Elasticsearch.")
                except Exception as e:
                    st.error(f"Error durante la indexación: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        else:
            st.warning("Por favor, sube un PDF y añade el nombre del activo.")

@st.cache_resource
def get_agent():
    return get_portfolio_agent("data/prueba.csv")

agent = get_agent()

# Interfaz de chat simple
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu consulta (ej. '¿Qué es un ETF?')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar y mostrar respuesta (SOLO EL ROUTER)
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                # 1. El semáforo decide la ruta
                ruta = classify_query(prompt)
                
                # 2. Ejecutar la herramienta adecuada
                if "PANDAS" in ruta:
                    st.toast("Analizando los números de la cartera...", icon="📊")
                    response = agent.invoke(prompt)
                    final_answer = response.get("output", "No pude generar una respuesta matemática.")
                else:
                    st.toast("Buscando en la base documental...", icon="📄")
                    final_answer = "Has elegido la ruta Elasticsearch. (¡Próximamente conectaremos la búsqueda!)"
                
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            except Exception as e:
                st.error(f"Error en el enrutamiento: {e}")