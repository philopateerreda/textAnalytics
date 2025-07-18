# Text Analytics Application

This repository contains two versions of a text analytics application:

## 1. Basic Version (NLTK-based)

A lightweight text analytics tool that uses NLTK for natural language processing. This version is compatible with Python 3.13 and doesn't require Docker.

### Features
- Text cleaning and preprocessing
- Word frequency and keyword density analysis
- Flesch-Kincaid readability scoring
- Simple SQLite database for vocabulary tracking
- Sentence and paragraph analysis

### Requirements
- Python 3.x (including 3.13)
- NLTK library
- No Docker needed

### Usage
```bash
# Install requirements
pip install nltk

# Run the application
python basic/main.py
```

## 2. SpaCy Version (Docker-based)

A more advanced text analytics tool that uses spaCy for sophisticated NLP capabilities. This version is containerized with Docker for easy setup and consistent execution.

### Features
- **Advanced NLP:** Utilizes spaCy for robust text processing, including lemmatization, part-of-speech tagging, and named entity recognition.
- **Intelligent Vocabulary Building:** Identifies new words from a text and allows the user to save them to a persistent vocabulary database.
- **Database Management:** Uses SQLite to store and manage the vocabulary. Includes a feature to back up the database.
- **Configurability:** Project settings can be managed through a `config.ini` file.
- **Flexible Execution:** Can be run easily using Docker Compose or locally within a Python virtual environment.
- **Dynamic Model Loading:** Automatically downloads the required spaCy language model if it's not already installed.

### Requirements
- Docker and Docker Compose (for containerized execution)
- Python 3.9+ (for local execution)

### Usage

You can run the application using either Docker Compose or a local Python virtual environment.

#### Running with Docker Compose

This is the recommended method for a clean, isolated environment. The `Updated 2025-04-13` instructions are as follows:

1.  Navigate to the `spacy_project` directory where the `docker-compose.yml` file is located.
2.  Build and run the container. If you have changed the `Dockerfile`, use the `--build` flag.
    ```bash
    # If you have changed the Dockerfile or want to rebuild
    docker-compose up --build -d

    # If you just want to start the container
    docker-compose up -d
    ```
3.  Execute the analysis script using `analyze_text.bat`. This script passes commands into the running Docker container.

    ```bash
    # Analyze a file without saving new words to the database
    .\analyze_text.bat "path\to\your\file.txt" --save-words no

    # Analyze a file and save new words
    .\analyze_text.bat "path\to\your\file.txt" --save-words yes

    # Create a backup of the database
    .\analyze_text.bat "path\to\your\file.txt" --backupDB yes

    # Backup the database and save new words
    .\analyze_text.bat "path\to\your\file.txt" --backupDB yes --save-words yes
    ```

#### Running with a Python Virtual Environment

If you prefer to run the script locally without Docker:

1.  Ensure you have Python 3.9+ and `pip` installed.
2.  Create and activate a virtual environment.
    ```bash
    # From the spacy_project directory
    python -m venv .venv
    .\.venv\Scripts\activate.bat
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the script directly using Python.
    ```bash
    # Analyze a file and save new words
    python src\main.py "path\to\your\file.txt" --save-words yes

    # Analyze a file and back up the database
    python src\main.py "path\to\your\file.txt" --backupDB yes
    ```

## Key Differences

| Feature | Basic Version | SpaCy Version |
|---------|--------------|---------------|
| NLP Library | NLTK | spaCy |
| Docker Required | No | Yes |
| Python Version | Works with 3.13 | Uses 3.9 (via Docker) |
| Word Analysis | Surface forms | Lemmatization |
| Architecture | Procedural | Object-oriented |
| Configuration | Hardcoded | Config file |

## Purpose

Both applications analyze text to provide insights about vocabulary, readability, and linguistic patterns. The basic version is lightweight and easy to run on any modern Python installation, while the SpaCy version offers more advanced language processing capabilities through Docker containerization.
