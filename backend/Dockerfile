FROM python:3.11-slim

# Install system dependencies for better PDF/Image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements first for better caching
COPY --chown=user requirements.txt .

# 1. Install regular dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 2. Install CPU-ONLY versions of AI libraries (This saves 1.5GB+)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    transformers \
    sentence-transformers

# Copy the rest of the app
COPY --chown=user . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

EXPOSE 10000

CMD ["python", "main.py"]
