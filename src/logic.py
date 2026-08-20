import pandas as pd
import pymupdf4llm
import os

from langchain_ollama import OllamaLLM
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:7b")
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL", "http://elasticsearch:9200")

def load_and_clean_portfolio(csv_path: str) -> pd.DataFrame:
    """
    Lee y normaliza el CSV del bróker para el Agente Pandas.
    """
    # 1. Cargar el CSV
    df = pd.read_csv(csv_path)
    
    # 2. Renombrar columnas a un estándar en inglés (ayuda mucho al LLM)
    # Ajusta los nombres originales (izquierda) según los que vengan de tu bróker
    column_mapping = {
        'Fecha': 'Date',
        'Activo': 'Ticker',
        'Tipo de operación': 'Type',
        'Títulos': 'Shares',
        'Precio': 'Price',
        'Total': 'Total_Amount'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # 3. Forzar el formato de fecha ISO 8601
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    
    # 4. Limpiar números (quitar símbolos de moneda y convertir comas a puntos)
    for col in ['Shares', 'Price', 'Total_Amount']:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('€', '').str.replace('$', '')
            df[col] = df[col].str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 5. Rellenar valores nulos (ej. dividendos que no tienen "Shares")
    df.fillna(0, inplace=True)
    
    return df

def get_portfolio_agent(csv_path: str):
    """
    Limpia el CSV y crea un agente de LangChain capaz de ejecutar 
    consultas sobre los datos usando Ollama.
    """
    # 1. Usamos tu función anterior para limpiar los datos
    df = load_and_clean_portfolio(csv_path)
    
    # 2. Inicializamos el modelo (igual que en la interfaz, pero con temperature=0 
    # para que sea analítico y no invente datos)
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    
    # 3. Creamos el agente
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True, # Para ver en la consola cómo piensa y qué código escribe
        allow_dangerous_code=True, # Permite la ejecución de Pandas
        handle_parsing_errors=True
    )
    
    return agent

def get_financial_chunks(pdf_path: str, doc_name: str):
    """Aplica triple chunking (Estructural > Semántico > Recursivo) a un PDF financiero."""
    
    # 1. Convertir PDF a Markdown al vuelo
    markdown_content = pymupdf4llm.to_markdown(pdf_path)
    
    # 2. Inicializar chunkers
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    semantic_chunker = SemanticChunker(embeddings)
    recursive_chunker = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    
    # 3. Fase 1: División Estructural
    headers_to_split_on = [("#", "Seccion"), ("##", "Subseccion"), ("###", "Apartado")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    sections = markdown_splitter.split_text(markdown_content)
    
    chunks = []
    for doc in sections:
        parent_text = doc.page_content
        context_path = " > ".join([f"{v}" for k, v in doc.metadata.items()])
        
        # 4. Fase 2: División Semántica
        semantic_splits = semantic_chunker.split_text(parent_text)
        
        # 5. Fase 3: Refinamiento Recursivo
        for s in semantic_splits:
            final_s_chunks = recursive_chunker.split_text(s)
            for sub_chunk in final_s_chunks:
                # Modificamos el contexto para reflejar el activo financiero
                chunks.append({
                    "search_text": f"[Activo: {doc_name}] [{context_path}]\n{sub_chunk}",
                    "parent_context": f"[Activo: {doc_name}] [{context_path}]\n{parent_text}"
                })
    return chunks

def index_financial_documents(pdf_path: str, doc_name: str, index_name: str = "informes_financieros"):
    """
    Ejecuta la fragmentación avanzada y guarda los resultados en Elasticsearch.
    """
    # 1. Obtenemos los fragmentos usando tu función de triple fase
    raw_chunks = get_financial_chunks(pdf_path, doc_name)
    
    # 2. Convertimos los diccionarios en objetos Document de LangChain
    documents = []
    for chunk in raw_chunks:
        doc = Document(
            page_content=chunk["search_text"], # El texto hijo corto (ideal para búsqueda vectorial)
            metadata={
                "parent_context": chunk["parent_context"], # El texto padre completo (ideal para el LLM)
                "doc_name": doc_name
            }
        )
        documents.append(doc)
        
    # 3. Inicializar embeddings e indexar
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    
    ElasticsearchStore.from_documents(
        documents=documents,
        embedding=embeddings,
        es_url=ELASTICSEARCH_URL,
        index_name=index_name,
    )
    
    return len(documents)

def classify_query(query: str) -> str:
    """Clasifica la pregunta para decidir la ruta de ejecución."""
    # Usamos temperatura 0 para que no sea creativo, solo analítico
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    
    template = """You are an expert classifier for RAG systems. Your task is to read the user's question and decide which database to send it to.
    
    OPTIONS:
    - PANDAS: If the question requires mathematical calculations regarding the portfolio, Trade Republic transactions, purchases, sales, or the amount of money invested (e.g., "How much have I spent on SXR8?").
    - ELASTICSEARCH: If the question concerns financial theory, macroeconomic risks, or the textual content of prospectuses for funds and ETFs tracking the S&P 500 (e.g., "What is the risk associated with this asset?").
    
    User question: {query}
    
    Respond ONLY with the word PANDAS or ELASTICSEARCH. Do not add anything else.
    """
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    
    # Ejecutamos y limpiamos espacios o saltos de línea residuales
    decision = chain.invoke({"query": query}).strip().upper()
    return decision

def ask_elasticsearch(query: str, index_name: str = "informes_financieros") -> str:
    """Busca en Elasticsearch y responde usando los fragmentos recuperados."""
    # 1. Conexión a la base vectorial
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vector_store = ElasticsearchStore(
        embedding=embeddings,
        es_url=ELASTICSEARCH_URL,
        index_name=index_name,
    )
    
    # Configuramos el retriever para traer los 3 fragmentos más relevantes
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 2. Configurar el LLM
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    
    # 3. Crear el prompt estricto
    template = """Use the following context excerpts to answer the question at the end.
    If you do not know the answer or it is not in the context, simply state that you do not have that information. Do not invent data under any circumstances.
    
    Context:
    {context}
    
    Question: {question}
    
    Helpful and direct answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # 4. Formatear documentos aprovechando el contexto padre
    def format_docs(docs):
        # Si existe el parent_context lo usamos, si no, usamos el texto normal
        return "\n\n".join(doc.metadata.get("parent_context", doc.page_content) for doc in docs)
    
    # 5. Cadena LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Ejecutamos la cadena completa
    return rag_chain.invoke(query)