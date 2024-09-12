import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Extract keywords using spaCy (faster)
def extract_keywords_spacy(page_content):
    doc = nlp(page_content)
    # Filter out single letters and stop words, and keep only nouns and proper nouns
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] 
                and not token.is_stop and len(token.text) > 1]
    return list(set(keywords))  # Return unique keywords

# Main function to extract keywords
def extract_and_refine_keywords(page_content):
    return extract_keywords_spacy(page_content)
