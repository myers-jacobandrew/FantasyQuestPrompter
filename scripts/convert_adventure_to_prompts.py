import nltk
from nltk.tokenize import sent_tokenize
from prompt_generator import PromptGenerator

# Download necessary NLTK resources (only need to do this once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Open the file and read in the text
with open('my_dnd_adventure.txt', 'r') as f:
    text = f.read()

# Tokenize the text into sentences
sentences = sent_tokenize(text)

# Define a function to filter out non-prompt sentences
def is_prompt(sentence):
    # Identify keywords using Named Entity Recognition (NER)
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    chunked = nltk.ne_chunk(tagged)
    keywords = set(' '.join(i[0] for i in t.leaves()) for t in chunked if hasattr(t, 'label') and t.label() == 'PERSON')
    
    # Define keywords to look for in a sentence
    prompt_keywords = ['go', 'explore', 'find', 'retrieve', 'defeat']
    
    # Return True if the sentence contains a prompt keyword and a named entity
    return any(keyword in sentence.lower() for keyword in prompt_keywords) and any(keyword in sentence.lower() for keyword in keywords)

# Filter the list of sentences to only include prompts
prompts = [sentence for sentence in sentences if is_prompt(sentence)]

# Create an instance of the PromptGenerator class
prompt_generator = PromptGenerator('fantasy_prompts.json')

# Generate a prompt for each sentence in the list of prompts
for prompt in prompts:
    generated_prompt = prompt_generator.generate_prompt(prompt)
    print(generated_prompt)
