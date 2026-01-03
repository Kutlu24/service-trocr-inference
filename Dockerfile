FROM huggingface/transformers-pytorch-gpu:latest

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.hf_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app

# Install pip + uv
RUN pip install --upgrade pip uv

# Dependency installation
COPY pyproject.toml README.md ./
RUN uv pip compile pyproject.toml -o requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# copy App code
COPY app ./app

# Expose port 8000
EXPOSE 8000

# Start the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]