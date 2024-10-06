import re
from rapidfuzz import fuzz
from inflect import engine
import nltk
from textblob import Word

nltk.download('wordnet')
nltk.download('omw-1.4')

inflector = engine()

def normalize_word(word):
    """Normalize the word by converting to lowercase and removing non-alphanumeric characters."""
    return re.sub(r'[^a-zA-Z0-9]', '', word.lower())

def singularize_and_lemmatize_word(word):
    """Singularize the word and convert to its base form."""
    if word:
        singular_form = inflector.singular_noun(word)#to have the singulars
        word = singular_form if singular_form else word
        word_blob = Word(word)
        base_form = word_blob.lemmatize("v")  # To have the base tense of keywords
        return base_form
    return word

def generate_keyword_variants(keyword):
    """Generate possible variants of a hyphenated keyword."""
    if '-' in keyword:
        parts = keyword.split('-')
        combined = ''.join(parts)
        spaced = ' '.join(parts)
        return [keyword, combined, spaced]
    return [keyword]

def is_multi_word_match(keyword, sentence):
    keyword_parts = [singularize_and_lemmatize_word(normalize_word(part)) for part in keyword.split()]
    sentence_parts = [singularize_and_lemmatize_word(normalize_word(part)) for part in re.split(r'[\s,-]+', sentence)]
    keyword_variants = generate_keyword_variants(keyword)
    for kw in keyword_variants:
        start_index = 0  # cheking the order of multi words
        for part in keyword_parts:
            try:
                start_index = sentence_parts.index(part, start_index) + 1
            except ValueError:
                return False
    return True

def is_keyword_in_sentence(sentence, keyword, threshold=95):
    """Check if the keyword is present in the sentence with a certain confidence level."""
    non_flexible_keywords = ["shirt", "t-shirt"]
    stop_words = ['i', 'me', 'my', 'we', 'you', 'he', 'she', 'it', 'they', 'is', 'are', 'was', 'be', 'have', 'do', 'the', 'and', 'a', 'an', 'in', 'to', 'of', 'for', 'on', 'with', 'at', 'by', 'from', 'about', 'this', 'does', 'has', 'have']

    temp_keyword = inflector.singular_noun(keyword) or keyword
    if (temp_keyword not in non_flexible_keywords) and (' ' in keyword or '-' in keyword):
        if is_multi_word_match(keyword, sentence):
            return True, keyword, 100
        else:
            return False, None, 0

    words = [w.strip() for w in re.split(r'[\s,]+', sentence) if w]

    keyword_singular = singularize_and_lemmatize_word(normalize_word(keyword))

    if non_flexible_keywords and keyword_singular in [normalize_word(kw) for kw in non_flexible_keywords]:
        for word in words:
            if word in stop_words:
                continue
            word_singular = singularize_and_lemmatize_word(normalize_word(word))
            if word_singular == keyword_singular:
                return True, word, 100
        return False, None, 0

    best_match, score = None, 0
    for word in words:
        if len(word) < 3 or word in stop_words:
            continue
        keyword_singular = singularize_and_lemmatize_word(normalize_word(keyword))
        word_singular = singularize_and_lemmatize_word(normalize_word(word))
        match_score = fuzz.partial_ratio(keyword_singular, word_singular)
        if match_score > score:
          if len(keyword_singular) == len(word_singular):
            best_match, score = word, match_score

    if score >= threshold:
        return True, best_match, score
    else:
        return False, None, score

# Example SWER5W  1qaed34 QA
title = "This a Tank Top "
keyword = "tank top"

present, match, confidence = is_keyword_in_sentence(title, keyword)

if present:
    print(f"Keyword '{keyword}' is present in the title as '{match}' with confidence {confidence}.")
else:
    print(f"Keyword '{keyword}' is not present in the title.")
