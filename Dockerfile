# Usamos una imagen ligera de Python
FROM python:3.11-slim

# Evitamos que Python escriba archivos .pyc y forzamos el output en consola
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copiamos e instalamos dependencias primero para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código
COPY . .

# Exponemos el puerto de Streamlit
EXPOSE 8501

# Comando para arrancar la app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]