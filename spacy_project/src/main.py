# Build and run with Docker Compose - Updated 2025-04-13
## first cd to the directory where docker-compose.yml is located
## then run if you change the Dockerfile:  docker-compose up --build -d
## then run if you don't:                  docker-compose up -d

# Use the Docker container
## to save words without backup DB:
###.\analyze_text.bat file_path --save-words yes
## to not save words:
###.\analyze_text.bat file_path --save-words no

## to backup the database:
###.\analyze_text.bat file_path --backupDB yes
###.\analyze_text.bat file_path --backupDB yes --save-words yes
###.\analyze_text.bat file_path --backupDB yes --save-words no

# use .venv
## first activate .venv by .\.venv\Scripts\activate.bat
## to save words without backup DB:
### python file_path  --save-words yes
## to backup the database:
### python file_path  --backupDB yes --save-words yes
### python file_path  --backupDB yes --save-words no

import re
import sqlite3
import time
import os
import json
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set
import spacy
import logging
from pathlib import Path
import argparse
from configparser import ConfigParser

# Set up logging as early as possible so that helper functions can use it
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Robust spaCy model loading ---
# Try to load progressively smaller English models, downloading them if necessary.
# Falls back to the smallest model if the larger ones are unavailable.
# This prevents the application from crashing when the requested model isn't pre-installed.

def _load_spacy_model() -> "spacy.language.Language":
    """Attempt to load an English spaCy model, downloading it if missing.

    Tries, in order: en_core_web_lg, en_core_web_md, en_core_web_sm.
    Returns the first successfully loaded model. Raises RuntimeError if none can
    be loaded.
    """
    from spacy.cli import download as spacy_download

    candidate_models = ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]
    for model in candidate_models:
        try:
            loaded_model = spacy.load(model)
            logger.info(f"Loaded spaCy model: {model}")
            return loaded_model
        except OSError:
            logger.warning(f"spaCy model '{model}' not found. Attempting download …")
            try:
                spacy_download(model)
                loaded_model = spacy.load(model)
                logger.info(f"Successfully downloaded and loaded spaCy model: {model}")
                return loaded_model
            except Exception as e:
                logger.error(f"Failed to download/load model '{model}': {e}")
                continue

    raise RuntimeError(
        "Unable to load any English spaCy model. Please install one of "
        "`en_core_web_sm`, `en_core_web_md`, or `en_core_web_lg`."
    )

# Load spaCy model globally so it is only loaded once per process
nlp = _load_spacy_model()


class TextAnalyzer:
    """A class to encapsulate text analysis logic and state."""
    
    # Entity types to exclude from vocabulary analysis
    EXCLUDED_ENTITY_TYPES = {
        'DATE', 'TIME', 'ORDINAL', 'CARDINAL', 'MONEY', 'PERCENT', 
        'QUANTITY', 'GPE', 'LOC', 'FAC', 'ORG', 'PRODUCT', 'EVENT'
    }
    
    def __init__(self, config_path: str = "config.ini"):
        """Initialize the analyzer with configuration."""
        self.config = self._load_config(config_path)
        self.db_path = Path(self.config["paths"]["db_path"])
        self.backup_dir = None
        self.output_dir = Path(self.config["paths"]["output_dir"])
        self.output_format = self.config["settings"]["output_format"]
        logger.info(f"Database path set to: {self.db_path.absolute()}")
        if not str(self.db_path).startswith(self.config["paths"]["output_dir"]):
            logger.warning("Database path is outside output_dir; it may not persist with volume mounts.")
        self.conn, self.cursor = self._init_db()

    def _load_config(self, config_path: str) -> Dict[str, Dict[str, str]]:
        """Load configuration from a file or set defaults."""
        config = ConfigParser()
        if os.path.exists(config_path):
            config.read(config_path)
        else:
            config["paths"] = {
                "db_path": "db/text_analysis.db",
                "output_dir": "text_analysis_results",
                "backup_dir": "backup",
            }
            config["settings"] = {
                "output_format": "txt",  # Options: txt, json
                "add_new_words": "true",
                "use_lemmas": "true",    # New setting for lemmatization
                "min_word_length": "3",  # Minimum length for words to consider
            }
            with open(config_path, "w") as f:
                config.write(f)
        return config

    def _init_db(self) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
        """Initialize SQLite database and create words table."""
        logger.info(f"Initializing database at: {self.db_path}")
        try:
            os.makedirs(self.db_path.parent, exist_ok=True)
            logger.info(f"Created directory: {self.db_path.parent}")
        except Exception as e:
            logger.error(f"Failed to create directory {self.db_path.parent}: {e}")
            raise
            
        if not os.access(self.db_path.parent, os.W_OK):
            logger.error(f"Cannot write to {self.db_path.parent}; check volume mount and permissions.")
            raise PermissionError(f"Cannot write to {self.db_path.parent}")
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Updated schema with additional fields
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS words (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT NOT NULL,
                    lemma TEXT NOT NULL,
                    pos TEXT,
                    definition TEXT,
                    examples TEXT,
                    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(lemma) ON CONFLICT IGNORE
                )
            """)
            
            # Check if we need to migrate existing data
            cursor.execute("PRAGMA table_info(words)")
            columns = [info[1] for info in cursor.fetchall()]
            
            if len(columns) > 0 and not all(col in columns for col in ['pos', 'definition', 'examples', 'date_added']):
                logger.info("Migrating existing word data to include new fields...")
                # Create temporary table with new schema
                cursor.execute("""
                    CREATE TABLE words_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        word TEXT NOT NULL,
                        lemma TEXT NOT NULL,
                        pos TEXT,
                        definition TEXT,
                        examples TEXT,
                        date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(lemma) ON CONFLICT IGNORE
                    )
                """)
                
                # Copy existing data
                cursor.execute("SELECT word, lemma FROM words")
                for row in cursor.fetchall():
                    word, lemma = row
                    cursor.execute(
                        "INSERT OR IGNORE INTO words_new (word, lemma) VALUES (?, ?)",
                        (word, lemma)
                    )
                
                # Replace old table with new one
                cursor.execute("DROP TABLE words")
                cursor.execute("ALTER TABLE words_new RENAME TO words")
                
                logger.info("Migration completed.")
            
            conn.commit()
            logger.info(f"Database initialized successfully at {self.db_path}")
            return conn, cursor
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def backup_db(self):
        """Create a backup of the SQLite database."""
        if not self.db_path.exists():
            logger.warning("Database does not exist, skipping backup.")
            return

        try:
            if self.backup_dir is None:
                self.backup_dir = Path(self.config["paths"]["backup_dir"])
            self.backup_dir.mkdir(exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            backup_file = self.backup_dir / f"{self.db_path.stem}_{timestamp}.db"
            
            # Connect to the source database and the backup database
            backup_conn = sqlite3.connect(backup_file)
            with backup_conn:
                self.conn.backup(backup_conn)
            backup_conn.close()
            
            logger.info(f"Database successfully backed up to {backup_file}")

        except Exception as e:
            logger.error(f"Failed to create database backup: {e}")

    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted patterns."""
        patterns = [
            r"\d+:\d{2}:\d{2},\d{3} --> \d{2}:\d{2},\d{3}",  # SRT timestamps
            r"^\d+$",  # Standalone numbers
            r"^\s*$\n",  # Empty lines
            r"</?i[^>]*>|<[^>]+>",  # HTML tags
            r"\d+",  # Numbers
            r"[^\w\s']",  # Special characters except apostrophes
            r"@\w+",  # Twitter handles
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",  # URLs
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email addresses
        ]
        for pattern in patterns:
            text = re.sub(pattern, " ", text, flags=re.MULTILINE if pattern.startswith("^") else 0)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def process_text(self, text: str) -> spacy.tokens.Doc:
        """Process text with spaCy for NLP analysis."""
        try:
            return nlp(text)
        except Exception as e:
            logger.error(f"spaCy processing failed: {e}")
            raise

    def is_valid_word(self, token: spacy.tokens.Token, excluded_entities: Set[int]) -> bool:
        """Check if a token should be considered for vocabulary analysis."""
        # Get minimum word length from config
        min_length = int(self.config["settings"].get("min_word_length", "3"))
        
        # Skip tokens that are part of excluded entity types
        if token.i in excluded_entities:
            return False
            
        # Skip tokens that are stopwords, punctuation, numbers, etc.
        if (token.is_punct or token.is_space or token.is_stop or 
            token.is_digit or token.like_num or token.like_url or 
            token.like_email or len(token.text) < min_length):
            return False
            
        # Skip common parts of speech we want to exclude
        if token.pos_ in ['PROPN', 'NUM', 'INTJ', 'X', 'SYM']:
            return False
            
        return True

    def extract_words_and_lemmas(self, doc: spacy.tokens.Doc) -> List[Tuple[str, str, str]]:
        """Extract words, lemmas and POS from the processed document, excluding entities we don't want."""
        word_lemma_pos_triples = []
        
        # Create a set of token indices that are part of excluded entity types
        excluded_token_indices = set()
        for ent in doc.ents:
            if ent.label_ in self.EXCLUDED_ENTITY_TYPES:
                excluded_token_indices.update(range(ent.start, ent.end))
        
        for token in doc:
            if self.is_valid_word(token, excluded_token_indices):
                # Create a clean version of the lemma (only alphabetic characters)
                clean_lemma = ''.join(c for c in token.lemma_.lower() if c.isalpha() or c == "'")
                
                if clean_lemma:  # Only add non-empty lemmas
                    word_lemma_pos_triples.append(
                        (token.text.lower(), clean_lemma, token.pos_)
                    )
                    
        return word_lemma_pos_triples

    def check_new_lemmas(self, word_lemma_pos_triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Identify words with lemmas not in the database."""
        unique_lemmas = {lemma for _, lemma, _ in word_lemma_pos_triples}
        existing_lemmas = set()
        
        try:
            self.cursor.execute("SELECT lemma FROM words")
            existing_lemmas = {row[0].lower() for row in self.cursor.fetchall()}
            new_triples = [(word, lemma, pos) for word, lemma, pos in word_lemma_pos_triples 
                         if lemma not in existing_lemmas]
            return new_triples
        except sqlite3.Error as e:
            logger.error(f"Error checking lemmas in database: {e}")
            return word_lemma_pos_triples  # Fallback to assume all are new

    def add_words_to_db(self, word_lemma_pos_triples: List[Tuple[str, str, str]], save_words_option="ask") -> int:
        """Add new words with lemmas to the database based on user confirmation."""
        if not word_lemma_pos_triples:
            return 0
        
        # Check if add_new_words is enabled in config
        if self.config["settings"]["add_new_words"].lower() != "true":
            logger.info("Adding new words is disabled in config")
            return 0
        
        # Group by lemma to avoid duplicates, keeping the most common POS
        lemma_groups = {}
        for word, lemma, pos in word_lemma_pos_triples:
            if lemma not in lemma_groups:
                lemma_groups[lemma] = {"word": word, "pos": pos}
        
        # Display new words found
        print("\n===== New Words Found =====")
        print(f"Found {len(lemma_groups)} new lemmas:")
        
        # Sort by alphabetical order for display
        sorted_lemmas = sorted(lemma_groups.items())
        for i, (lemma, info) in enumerate(sorted_lemmas[:15], 1):
            print(f"{i}. {info['word']} -> {lemma} ({info['pos']})")
        
        if len(lemma_groups) > 15:
            print(f"... and {len(lemma_groups) - 15} more.")
        
        # Process based on save_words_option
        if save_words_option == "no":
            logger.info("Not saving words (command-line option)")
            return 0
        elif save_words_option == "ask":
            try:
                response = input("\nDo you want to save these words to the database? (y/n): ").strip().lower()
                if response != 'y':
                    logger.info("User chose not to save new words")
                    return 0
            except Exception as e:
                logger.error(f"Error getting user input: {e}")
                logger.warning("Defaulting to not saving words")
                return 0
        
        try:
            self.cursor.executemany(
                "INSERT OR IGNORE INTO words (word, lemma, pos) VALUES (?, ?, ?)",
                [(info["word"], lemma, info["pos"]) for lemma, info in lemma_groups.items()],
            )
            self.conn.commit()
            return self.cursor.rowcount
        except sqlite3.Error as e:
            logger.error(f"Failed to add words to database: {e}")
            return 0

    def analyze_text(self, doc: spacy.tokens.Doc) -> Dict[str, float]:
        """Perform readability and structure analysis."""
        sentences = list(doc.sents)
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        total_words = len(words)
        
        if not sentences or not words:
            return {
                "sentence_count": 0,
                "paragraph_count": 0,
                "avg_sentence_length": 0.0,
                "flesch_kincaid_grade": 0.0,
                "sentiment": 0.0,
            }

        avg_sentence_length = total_words / len(sentences)
        syllables = sum(self._count_syllables(word) for word in words)
        flesch_kincaid = (
            0.39 * (total_words / len(sentences)) +
            11.8 * (syllables / total_words) - 15.59
        )
        sentiment = doc.sentiment if hasattr(doc, "sentiment") else 0.0  # Requires spaCy sentiment pipeline

        return {
            "sentence_count": len(sentences),
            "paragraph_count": doc.text.count("\n\n") + 1,
            "avg_sentence_length": round(avg_sentence_length, 2),
            "flesch_kincaid_grade": round(flesch_kincaid, 2),
            "sentiment": round(sentiment, 2),
        }

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Estimate syllable count for a word."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word and word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if word.endswith("le"):
            count += 1
        return max(count, 1)

    def calculate_keyword_density(self, doc: spacy.tokens.Doc) -> Tuple[Dict[str, Tuple[int, float]], int]:
        """Calculate word frequency and density using lemmas if configured."""
        # Create a set of token indices that are part of excluded entity types
        excluded_token_indices = set()
        for ent in doc.ents:
            if ent.label_ in self.EXCLUDED_ENTITY_TYPES:
                excluded_token_indices.update(range(ent.start, ent.end))
        
        use_lemmas = self.config["settings"].get("use_lemmas", "true").lower() == "true"
        min_length = int(self.config["settings"].get("min_word_length", "3"))
        
        words = []
        for token in doc:
            if self.is_valid_word(token, excluded_token_indices):
                if use_lemmas:
                    word = token.lemma_.lower()
                else:
                    word = token.text.lower()
                
                # Only add words that meet the length requirement
                if len(word) >= min_length:
                    words.append(word)
        
        total_words = len(words)
        if not total_words:
            return {}, 0
        
        word_counts = Counter(words)
        return (
            {word: (count, round((count / total_words) * 100, 2)) 
             for word, count in word_counts.most_common(100)},  # Limit to top 50 words
            total_words,
        )

    def group_by_frequency(self, keyword_density: Dict[str, Tuple[int, float]]) -> Dict[int, Dict[str, float]]:
        """Group words by their frequency of occurrence."""
        frequency_groups = {}
        for word, (count, density) in keyword_density.items():
            frequency_groups.setdefault(count, []).append((word, density))
        
        return {
            count: {
                "word_count": len(words),
                "total_words": count * len(words),
                "total_percentage": round(sum(density for _, density in words), 2),
                "words": [word for word, _ in sorted(words)],  # Include the actual words
            } for count, words in frequency_groups.items()
        }

    def extract_entities(self, doc: spacy.tokens.Doc) -> Dict[str, List[str]]:
        """Extract named entities from the text, filtering out unwanted types."""
        entities = {}
        for ent in doc.ents:
            # Skip entity types we want to exclude
            if ent.label_ not in self.EXCLUDED_ENTITY_TYPES:
                entities.setdefault(ent.label_, []).append(ent.text)
        return entities

    def save_results(self, results: Dict[str, any], output_file: Path) -> None:
        """Save analysis results in the specified format."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.output_format.lower() == "json":
            with open(output_file.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        else:  # Default to txt
            with open(output_file.with_suffix(".txt"), "w", encoding="utf-8") as f:
                self._write_txt_results(f, results)

    def _write_txt_results(self, f, results: Dict[str, any]) -> None:
        """Write results to a text file with improved formatting."""
        f.write("=" * 80 + "\n")
        f.write("VOCABULARY LEARNING ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("NEW WORDS TO LEARN:\n")
        f.write("-" * 80 + "\n")
        if results["new_words"]:
            # Group by POS tag for better organization
            pos_groups = {}
            for word, lemma, pos in results["new_words"]:
                pos_groups.setdefault(pos, []).append((word, lemma))
            
            # Display words by part of speech
            for pos, words in pos_groups.items():
                f.write(f"\n{pos} words:\n")
                f.write("-" * 40 + "\n")
                # Deduplicate words while preserving order
                seen = set()
                deduped_words = []
                for word, lemma in words:
                    if word not in seen:
                        seen.add(word)
                        deduped_words.append((word, lemma))
                for word, lemma in deduped_words:
                    if word == lemma:
                        f.write(f"• {word}\n")
                    else:
                        f.write(f"• {word} → {lemma}\n")
        else:
            f.write("No new words found to learn.\n")
        
        f.write("DATABASE STATISTICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"• Words in database before analysis: {results['database_stats']['total_words_in_db_before']}\n")
        f.write(f"• Known lemmas in text: {results['database_stats']['old_lemmas_in_text']}\n")
        f.write(f"• New lemmas found in text: {results['database_stats']['new_lemmas_found']}\n")
        f.write(f"• New lemmas added to database: {results['database_stats']['new_lemmas_added']}\n")
        f.write(f"• Total words in database after analysis: {results['database_stats']['total_words_in_db_after']}\n")
        f.write("\n")
        
        f.write("TEXT ANALYSIS:\n")
        f.write("-" * 80 + "\n")
        for key, value in results["text_analysis"].items():
            f.write(f"• {key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        
        f.write("WORD FREQUENCY ANALYSIS:\n")
        f.write("-" * 80 + "\n")
        f.write("Showing top 50 most frequent words:\n\n")
        f.write(f"{'Word':<20}{'Count':<10}{'Frequency (%)':<15}\n")
        f.write("-" * 45 + "\n")
        
        # Sort by count (descending) then alphabetically
        for word, (count, density) in sorted(
            results["keyword_density"].items(), 
            key=lambda x: (-x[1][0], x[0])
        ):
            f.write(f"{word:<20}{count:<10}{density:<15.2f}\n")
        f.write("\n")
        
        f.write("FREQUENCY GROUPS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Frequency':<15}{'Word Count':<15}{'Examples':<50}\n")
        f.write("-" * 80 + "\n")
        
        # Show frequency groups from most to least frequent
        for freq, details in sorted(results["frequency_groups"].items(), reverse=True):
            # Show a few example words
            example_words = details["words"][:5]
            if len(details["words"]) > 5:
                example_words.append(f"... ({len(details['words'])-5} more)")
            
            f.write(f"{freq:<15}{details['word_count']:<15}{', '.join(example_words)}\n")

    def backup_database(self, analyzed_filename: str) -> None:
        """
        Create a backup of the database after analyzing a file.
        
        Args:
            analyzed_filename: Name of the file that was analyzed (used for naming the backup)
        """
        # Create backup directory if it doesn't exist
        if self.backup_dir is None:
            self.backup_dir = Path(self.config["paths"]["backup_dir"])
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate backup filename with timestamp and analyzed filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename_stem = Path(analyzed_filename).stem  # Gets the filename without the extension
        backup_path = self.backup_dir / f"text_analysis_db_{filename_stem}_{timestamp}.db"
        
        try:
            # Connect to a new database file
            backup_conn = sqlite3.connect(backup_path)
            
            # Backup current database to the new connection
            with backup_conn:
                self.conn.backup(backup_conn)
            
            backup_conn.close()
            logger.info(f"Database backed up to {backup_path}")
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            logger.error(f"Error details: {str(e)}")  # Add more detailed error logging

    def analyze_file(self, input_file: str, save_words_option="ask") -> None:
        """Analyze a text file and save the results."""
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Input file '{input_path}' does not exist.")
            return

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()
            cleaned_text = self.clean_text(text)
            doc = self.process_text(cleaned_text)
            
            # Extract words and lemmas with POS
            word_lemma_pos_triples = self.extract_words_and_lemmas(doc)
            unique_lemmas = {lemma for _, lemma, _ in word_lemma_pos_triples}
            
            # Total words in database before
            self.cursor.execute("SELECT COUNT(*) FROM words")
            total_words_in_db_before = self.cursor.fetchone()[0] or 0
            
            # Check for new lemmas
            new_triples = self.check_new_lemmas(word_lemma_pos_triples)
            new_lemmas = {lemma for _, lemma, _ in new_triples}
            
            # Add new words to the database
            added_count = self.add_words_to_db(new_triples, save_words_option)
            if added_count:
                logger.info(f"Added {added_count} new lemmas to the database.")
            
            # Total words in database after
            total_words_in_db_after = total_words_in_db_before + added_count
            old_lemmas_in_text = len(unique_lemmas) - len(new_lemmas)

            # Calculate metrics
            keyword_density, total_words = self.calculate_keyword_density(doc)
            text_analysis = self.analyze_text(doc)
            frequency_groups = self.group_by_frequency(keyword_density)
            entities = self.extract_entities(doc)

            results = {
                "database_stats": {
                    "total_words_in_db_before": total_words_in_db_before,
                    "old_lemmas_in_text": old_lemmas_in_text,
                    "new_lemmas_found": len(new_lemmas),
                    "new_lemmas_added": added_count,
                    "total_words_in_db_after": total_words_in_db_after,
                },
                "new_words": new_triples,
                "text_analysis": text_analysis,
                "frequency_groups": frequency_groups,
                "keyword_density": keyword_density,
                "entities": entities,
                "total_words": total_words,
            }

            # Use the input filename (without extension) as part of the output filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename_stem = input_path.stem  # Gets the filename without the extension
            output_file = self.output_dir / f"{filename_stem}_{timestamp}"
            self.save_results(results, output_file)
            logger.info(f"Results saved to '{output_file}.{self.output_format}'")
            
            # Backup the database after analysis only if we're not using --save-words no
            if save_words_option != "no":
                self.backup_database(input_file)
            else:
                logger.info("Skipping database backup (--save-words no specified)")

        except Exception as e:
            logger.error(f"Analysis failed: {e}")

    def __del__(self):
        """Ensure database connection is closed."""
        if hasattr(self, "conn"):
            self.conn.close()

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Advanced Text Analysis Tool")
    parser.add_argument("input_path", type=str, help="Path to the input text file or directory")
    parser.add_argument("--config", type=str, default="config.ini", help="Path to configuration file")
    parser.add_argument("--save-words", choices=["yes", "no", "ask"], default="ask", 
                      help="Whether to save new words to database: yes, no, or ask (default)")
    parser.add_argument("--min-word-length", type=int, default=3,
                      help="Minimum length of words to include in analysis (default: 3)")
    parser.add_argument("--backupDB", choices=["yes", "no"], default="no",
                        help="Create a backup of the database? 'yes' or 'no' (default).")
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_args()
    
    # Create or update config file with command line arguments
    config = ConfigParser()
    if os.path.exists(args.config):
        config.read(args.config)
    else:
        config["paths"] = {
            "db_path": "db/text_analysis.db",
            "output_dir": "text_analysis_results",
        }
        config["settings"] = {
            "output_format": "txt",
            "add_new_words": "true",
            "use_lemmas": "true",
        }
    
    # Update config with command line args
    config["settings"]["min_word_length"] = str(args.min_word_length)
    
    with open(args.config, "w") as f:
        config.write(f)
    
    analyzer = TextAnalyzer(args.config)

    if args.backupDB == "yes":
        analyzer.backup_db()
    input_path = Path(args.input_path)

    if input_path.is_dir():
        logger.info(f"Processing directory: {input_path}")
        files_to_process = sorted(
            [p for p in input_path.glob("*") if p.suffix.lower() in {".txt", ".srt"}]
        )
        if not files_to_process:
            logger.warning(f"No .txt or .srt files found in {input_path}")
            return

        for file_path in files_to_process:
            logger.info(f"Analyzing file: {file_path}")
            try:
                analyzer.analyze_file(str(file_path), args.save_words)
            except Exception as e:
                logger.error(f"Failed to analyze {file_path}: {e}")
    elif input_path.is_file():
        analyzer.analyze_file(str(input_path), args.save_words)
    else:
        logger.error(f"Input path {input_path} is not a valid file or directory.")

if __name__ == "__main__":
    main()