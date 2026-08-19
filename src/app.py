import streamlit as st
from langchain_ollama import OllamaLLM

from logic import get_portfolio_agent

# Configuración básica de la interfaz
st.set_page_config(page_title="Asistente Financiero Dual", layout="wide")
st.title("📈 Asistente RAG: Análisis de Cartera")

# Inicializamos el LLM. 
# Importante: Usamos http://ollama:11434 porque la app y Ollama comparten la red interna de Docker
@st.cache_resource
def get_agent():
    # Asegúrate de que la ruta coincida con donde guardaste el CSV
    return get_portfolio_agent("data/prueba.csv")

agent = get_agent()

# Interfaz de chat simple
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu consulta (ej. '¿Qué es un ETF?')..."):
    # Guardar pregunta del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar y mostrar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando la cartera..."):
            try:
                # Usamos el agente en lugar del llm básico
                response = agent.invoke(prompt)
                
                # LangChain devuelve un diccionario, extraemos la respuesta final
                final_answer = response.get("output", "No pude generar una respuesta.")
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            except Exception as e:
                st.error(f"Error en el análisis: {e}")