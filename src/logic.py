import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

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
    llm = OllamaLLM(model="qwen2.5:7b", base_url="http://ollama:11434", temperature=0)
    
    # 3. Creamos el agente
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True, # Para ver en la consola cómo piensa y qué código escribe
        allow_dangerous_code=True, # Permite la ejecución de Pandas
        handle_parsing_errors=True
    )
    
    return agent