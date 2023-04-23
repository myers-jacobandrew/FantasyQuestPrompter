import json
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter

# Load JSON data
with open('fantasy_prompts.json', 'r') as f:
    data = json.load(f)

# Define NLP pipeline
nlp = spacy.load('en_core_web_sm')

# Define function to preprocess text
def preprocess(text):
    doc = nlp(text)
    # Remove stop words and punctuation, and lemmatize tokens
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    # Count word frequencies and return as dictionary
    word_freq = dict(Counter(tokens))
    return word_freq

# Preprocess each prompt and store results in a list
processed_data = []
for prompt in data:
    processed_prompt = {
        'prompt': prompt['prompt'],
        'keywords': preprocess(prompt['prompt']),
        'difficulty': prompt['difficulty']
    }
    processed_data.append(processed_prompt)

# Save processed data to a JSON file
with open('fantasy_prompts_processed.json', 'w') as f:
    json.dump(processed_data, f)
