# 📈 Asistente Financiero Dual (RAG + Pandas Agent)

Este proyecto implementa un asistente de Inteligencia Artificial generativa 100% local y privado, diseñado para la gestión y análisis de carteras de inversión. 

Utiliza una arquitectura de **Enrutamiento Dinámico (Router)** que clasifica la intención del usuario y deriva la consulta a uno de sus dos motores principales: un agente matemático para analizar transacciones en formato tabular o un sistema RAG avanzado para consultar folletos financieros en PDF.

---

## Arquitectura del Sistema

El núcleo del asistente está orquestado con **LangChain** y se divide en tres componentes clave:

1. **El Enrutador (Router LLM):** Un clasificador semántico que lee el *prompt* del usuario y decide en milisegundos si la pregunta requiere cálculos sobre la cartera o teoría financiera.
2. **Pandas DataFrame Agent:** Se encarga de la ruta analítica. Lee exportaciones en CSV de plataformas de inversión (ej. Trade Republic), normaliza los datos y escribe/ejecuta código Python en segundo plano para responder preguntas matemáticas (coste total, rentabilidad, filtrado de activos como SXR8).
3. **Motor RAG Documental (Elasticsearch):** Se encarga de la ruta teórica. Permite subir informes anuales o folletos (KID) en PDF a través de la interfaz. Implementa una estrategia de **Recuperación Padre-Hijo** mediante una triple fase de fragmentación:
   * *Fase Estructural:* Conversión de PDF a Markdown al vuelo y división por cabeceras.
   * *Fase Semántica:* Agrupación por coherencia de significado usando embeddings locales.
   * *Fase Recursiva:* Ajuste estricto al límite de tokens del modelo.

---

## Stack Tecnológico

* **Interfaz:** Streamlit
* **Orquestación:** LangChain & LangChain Experimental
* **Modelos Locales (Ollama):** 
  * LLM: `qwen2.5:7b` (Equilibrio óptimo entre lógica matemática y consumo de RAM).
  * Embeddings: `nomic-embed-text`
* **Base de Datos Vectorial:** Elasticsearch (Dockerizado)
* **Procesamiento de Datos:** Pandas, PyMuPDF4LLM

---

## Requisitos Previos

* **Docker & Docker Compose** instalados.
* **Memoria RAM:** Se recomienda asignar al menos 8-10 GB de RAM al subsistema de Docker (WSL2 en Windows) para garantizar una ejecución fluida del modelo de 7B parámetros.

---

## Instalación y Despliegue

1. **Clonar el repositorio y levantar la infraestructura:**
   ```bash
   docker-compose up --build -d
