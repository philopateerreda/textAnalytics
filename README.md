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

A more advanced text analytics tool that uses spaCy for sophisticated NLP capabilities including lemmatization. This version requires Docker.

### Features
- Advanced text processing with spaCy
- Lemmatization for more accurate word analysis
- Entity recognition and filtering
- Configurable via config.ini
- Enhanced database schema with lemmas, POS tagging
- Multiple output formats (txt, json)

### Requirements
- Docker and Docker Compose
- Python 3.9 (managed via Docker)
- spaCy and its dependencies (installed via Docker)

### Usage
```bash
# Build and run with Docker Compose
docker-compose up -d

# Use the Docker container
docker exec -it text-analytics-app python main.py --file /app/readT/your_file.txt
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
