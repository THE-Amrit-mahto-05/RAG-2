FROM python:3.9

# Create a non-root user for security (Requirement for Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements first for better caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the app
COPY --chown=user . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Hugging Face Spaces always listens on 7860
EXPOSE 7860

CMD ["python", "main.py"]
