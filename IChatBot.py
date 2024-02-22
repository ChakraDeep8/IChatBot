import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

class Chatbot:
    def __init__(self):
        self.df = pd.read_csv('dialogs.txt', sep='\t', header=None, names=['Question', 'Answer'])
        self.tokenizer = RegexpTokenizer(r'\b\w+\b')  # Modified to include short words
        self.stop_words = set(stopwords.words('english')) - {'what', 'when', 'where', 'which', 'who', 'whom', 'why', 'how'}
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Question'].apply(self.preprocess_text))
        self.previous_response = None

    def preprocess_text(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)

    def generate_response(self, user_query, threshold=0.2):
        user_query = self.preprocess_text(user_query)
        user_tfidf = self.vectorizer.transform([user_query])
        similarities = cosine_similarity(user_tfidf, self.tfidf_matrix)
        similar_indices = similarities[0] > threshold
        if similar_indices.any():
            response_index = random.choice(similar_indices.nonzero()[0])
            self.previous_response = self.df.loc[response_index, 'Answer']
            return self.previous_response
        elif user_query.lower() in ['lol', 'haha', 'hehe']:  # Handling short expressions
            responses = [
                "That's funny!",
                "LOL!",
                "Haha, good one!",
                "Hehe, you got me!",
                "I chuckled!"
            ]
            return random.choice(responses)
        else:
            return "I'm sorry, I don't understand that."
