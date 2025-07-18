import re
import sqlite3
import time
import os
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download necessary NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Function to initialize and connect to the SQLite database
def init_db():
    conn = sqlite3.connect("P:\\progs\\vsCode\\VSProjects\\yourTools\\text-analytics\\text_analysis.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT UNIQUE COLLATE NOCASE
        )
    """)
    conn.commit()
    return conn, cursor

# Function to check if a word or phrase is new or exists in the database
def check_new_words(words, cursor):
    new_words = []
    for word in set(words):
        cursor.execute("SELECT word FROM words WHERE word = ? COLLATE NOCASE", (word,))
        result = cursor.fetchone()
        if not result:
            new_words.append(word)
    return new_words

# Function to add a word or phrase to the database
def add_word_to_db(word, cursor, conn):
    try:
        cursor.execute("INSERT INTO words (word) VALUES (?)", (word.lower(),))
        conn.commit()
    except sqlite3.IntegrityError:
        pass

def get_keyword_density(text):
    words = [word for word in word_tokenize(text.lower()) if word.isalnum() and word not in set(stopwords.words('english'))]
    word_counts = Counter(words)
    total_words = len(words)

    keyword_density = {}
    if total_words > 0:
        for word, count in word_counts.items():
            density = (count / total_words) * 100
            keyword_density[word] = (count, round(density, 2))

    return keyword_density, total_words

def analyze_text(text):
    sentences = sent_tokenize(text)
    words = [word for word in word_tokenize(text) if word.isalnum()]

    num_words = len(words)
    num_sentences = len(sentences)

    avg_sentence_length = round(num_words / num_sentences, 2) if num_sentences > 0 else 0

    syllables = sum(count_syllables(word) for word in words)

    if num_words > 0 and num_sentences > 0:
        flesch_kincaid = 0.39 * (num_words / num_sentences) + 11.8 * (syllables / num_words) - 15.59
    else:
        flesch_kincaid = 0

    return {
        'sentence_count': num_sentences,
        'paragraph_count': len(re.split(r'\n\s*\n+', text.strip())),
        'avg_sentence_length': avg_sentence_length,
        'flesch_kincaid_grade': round(flesch_kincaid, 2)
    }

def count_syllables(word):
    word = word.lower()
    count = 0
    vowels = 'aeiouy'
    if word and word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    return max(1, count)

def group_words_by_frequency(keyword_density):
    frequency_groups = {}
    for word, (count, density) in keyword_density.items():
        if count not in frequency_groups:
            frequency_groups[count] = []
        frequency_groups[count].append((word, density))

    group_summaries = {}
    for count, words in frequency_groups.items():
        total_words_in_group = count * len(words)
        total_percentage = sum(density for _, density in words)
        group_summaries[count] = {
            'word_count': len(words),
            'total_words': total_words_in_group,
            'total_percentage': round(total_percentage, 2)
        }

    return group_summaries

def clean_text(text):
    # Remove SRT formatting artifacts (timestamps and subtitle numbers)
    text = re.sub(r'\d+:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', text)
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)  #standalone numbers (subtitle numbers)
    text = re.sub(r'^\s*$\n', '', text, flags=re.MULTILINE)  # empty lines

    text = re.sub(r'</?i[^>]*>', '', text)  # Remove <i> and </i> tags and their variants

    text = re.sub(r'<[^>]+>', '', text)  #all HTML-like tags

    text = re.sub(r'\d+', '', text)  # Remove all numbers

    return text.strip()

def main():
    # Initialize the SQLite database
    conn, cursor = init_db()

    # Path to the text file (replace with the path to your file).
    
    text_file_path =  "P:\\progs\\vsCode\\VSProjects\\yourTools\\text-analytics\\readT\\other\\NLP1.txt"


    # Read the content of the text file.
    try:
        with open(text_file_path, "r", encoding="utf-8") as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{text_file_path}'")
        return

    text = clean_text(text)

    words = [word.lower() for word in word_tokenize(text) if word.isalnum()]
    new_words = check_new_words(words, cursor)

    # combine new words.
    all_new_words = set(new_words)

    # If new words are found, ask the user if they want to add them to the database.
    if all_new_words:

        choice = input("Do you want to add these new words to the database? (yes/no): ").lower()
        if choice == 'yes':
            for word in all_new_words:
                add_word_to_db(word, cursor, conn)

    keyword_density, total_words = get_keyword_density(text)
    text_analysis = analyze_text(text)
    frequency_groups = group_words_by_frequency(keyword_density)

    # Fixed directory for saving the file
    output_directory = "P:\\progs\\vsCode\\VSProjects\\yourTools\\text-analytics\\text_analysis_results"

    # Create the directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    #unique filename using a timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_directory, f"text_analysis_{timestamp}.txt")

    # writing output to a new text file
    with open(filename, "w", encoding="utf-8") as f:
        if all_new_words:
            f.write("New words found in the text:\n")
            for word in all_new_words:
                f.write(f"- {word}\n")
            f.write("\n")

        f.write("Text Analysis:\n")
        f.write(f"Total words: {total_words}\n")
        f.write(f"Sentence count: {text_analysis['sentence_count']}\n")
        f.write(f"Paragraph count: {text_analysis['paragraph_count']}\n")
        f.write(f"Average sentence length: {text_analysis['avg_sentence_length']} words\n")
        f.write(f"Flesch-Kincaid Grade Level: {text_analysis['flesch_kincaid_grade']}\n")

        f.write("\nFrequency Group Analysis:\n")
        f.write("Frequency\tWords\tTotal Words\tTotal Percentage\n")
        f.write("-" * 60 + "\n")
        for frequency, summary in sorted(frequency_groups.items(), reverse=True):
            words_str = ", ".join([word for word, _ in summary.items()])
            f.write(f"{frequency}\t\t{summary['word_count']}\t\t{summary['total_words']}\t\t{summary['total_percentage']}%\n")

        f.write("\nKeyword Density Analysis:\n")
        f.write("Word\t\tCount\t\tDensity (%)\n")
        f.write("-" * 40 + "\n")
        for word, (count, density) in sorted(keyword_density.items(), key=lambda x: x[1][0], reverse=True):
            f.write(f"{word:<15}\t\t{count:<10}\t\t{density}\n")

    conn.close()

    print(f"Text analysis has been saved in '{filename}'")

if __name__ == "__main__":
    main()