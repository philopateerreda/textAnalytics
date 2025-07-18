FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential && \
    pip install --no-cache-dir -r requirements.txt
# Install a lightweight English model by default to keep the image small.
# The application will attempt to download a larger model at runtime if available.
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
# Keep the container running in the background so that we can exec into it
CMD ["sleep", "infinity"]

