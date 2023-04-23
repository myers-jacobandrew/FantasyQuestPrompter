import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tag import pos_tag
from collections import defaultdict
from gensim import corpora, models, similarities

class PromptGenerator:
    def __init__(self, prompts_path):
        self.prompts = self.load_prompts(prompts_path)
        self.corpus = self.build_corpus(self.prompts)
        self.dictionary = self.build_dictionary(self.corpus)
        self.model = self.build_model(self.corpus, self.dictionary)
        self.index = self.build_index(self.model, self.corpus)

    def load_prompts(self, prompts_path):
        with open(prompts_path, 'r') as f:
            prompts = f.read().splitlines()
        return prompts

    def build_corpus(self, prompts):
        corpus = []
        for prompt in prompts:
            tokens = self.tokenize(prompt)
            corpus.append(tokens)
        return corpus

    def build_dictionary(self, corpus):
        dictionary = corpora.Dictionary(corpus)
        return dictionary

    def build_model(self, corpus, dictionary):
        tfidf = models.TfidfModel(corpus)
        model = models.LsiModel(tfidf[corpus], id2word=dictionary, num_topics=10)
        return model

    def build_index(self, model, corpus):
        index = similarities.MatrixSimilarity(model[corpus])
        return index

    def tokenize(self, text):
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens]
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        tokens = pos_tag(tokens)
        tokens = [(word, self.get_wordnet_pos(tag)) for word, tag in tokens]
        tokens = [self.stem_lemmatize(word, tag) for word, tag in tokens]
        return tokens

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return 'a'
        elif treebank_tag.startswith('V'):
            return 'v'
        elif treebank_tag.startswith('N'):
            return 'n'
        elif treebank_tag.startswith('R'):
            return 'r'
        else:
            return None

    def stem_lemmatize(self, word, pos):
        porter = PorterStemmer()
        wordnet = WordNetLemmatizer()
        if pos:
            lemma = wordnet.lemmatize(word, pos=pos)
        else:
            lemma = wordnet.lemmatize(word)
        stemmed = porter.stem(lemma)
        return stemmed

    def generate_prompt(self, query):
        query_tokens = self.tokenize(query)
        query_bow = self.dictionary.doc2bow(query_tokens)
        query_lsi = self.model[query_bow]
        sims = self.index[query_lsi]
        index_sorted = sorted(enumerate(sims), key=lambda item: -item[1])
        prompt = self.prompts[index_sorted[0][0]]
        return prompt
