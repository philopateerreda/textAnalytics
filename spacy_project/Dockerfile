FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create directories for output and database
RUN mkdir -p text_analysis_results db OldDBs

# Set the default config.ini
RUN echo "[paths]" > config.ini && \
    echo "db_path = db/text_analysis.db" >> config.ini && \
    echo "output_dir = text_analysis_results" >> config.ini && \
    echo "[settings]" >> config.ini && \
    echo "output_format = txt" >> config.ini && \
    echo "add_new_words = true" >> config.ini && \
    echo "use_lemmas = true" >> config.ini

# Set volume for the database and other directories
VOLUME ["/app/db", "/app/text_analysis_results", "/app/OldDBs"]

# Change from ENTRYPOINT to CMD or remove completely
# Option 1: Use CMD instead of ENTRYPOINT
CMD ["python"]

