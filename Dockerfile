FROM python:3.12-slim

# Evita buffer no log
ENV PYTHONUNBUFFERED=1

# Diretório de trabalho
WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instala Poetry
RUN pip install --no-cache-dir poetry

# Copia arquivos de dependência primeiro (cache)
COPY pyproject.toml poetry.lock* /app/

# Configura Poetry para NÃO criar venv
RUN poetry config virtualenvs.create false

# Instala dependências
RUN poetry install --no-interaction --no-ansi

# Copia o restante do projeto
COPY . /app

# Porta da UI
EXPOSE 8501

# Comando padrão (UI)
CMD ["streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
