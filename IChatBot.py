import json
import random
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

'''# Download NLTK data and store it in the Streamlit app directory
nltk.download('punkt', download_dir='.')
nltk.download('stopwords', download_dir='.')
nltk.download('wordnet', download_dir='.')'''

# Add NLTK's data directory to the path
nltk.data.path.append('.')

class Chatbot:
    def __init__(self):
        self.df = pd.read_csv('dialogs.txt', sep='\t', header=None, names=['Question', 'Answer'])
        self.tokenizer = RegexpTokenizer(r'\b\w+\b')  # Modified to include short words
        self.stop_words = set(stopwords.words('english')) - {'what', 'when', 'where', 'which', 'who', 'whom', 'why', 'how'}
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Question'].apply(self.preprocess_text))
        self.previous_response = None

        with open('additional_response.json', 'r', encoding='utf-8') as f:
            additional_data = json.load(f)

        self.additional_jokes = additional_data['additional_jokes']
        self.additional_short_expressions = additional_data['additional_short_expressions']
        self.emoji_responses = additional_data['emoji_responses']

    def preprocess_text(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)

    def generate_response(self, user_query, threshold=0.2):
        # Check for emoji responses
        for emoji_code, responses in self.emoji_responses.items():
            if emoji_code in user_query:
                return random.choice(responses)

        # Process user query and check for matching responses
        user_query = self.preprocess_text(user_query)
        user_tfidf = self.vectorizer.transform([user_query])
        similarities = cosine_similarity(user_tfidf, self.tfidf_matrix)
        similar_indices = similarities[0] > threshold
        if similar_indices.any():
            response_index = random.choice(similar_indices.nonzero()[0])
            self.previous_response = self.df.loc[response_index, 'Answer']
            return self.previous_response

        # Check for additional short expressions
        for expression, responses in self.additional_short_expressions.items():
            if user_query.lower() == expression:
                return random.choice(responses)

        # Check for jokes
        if user_query.lower() == 'tell me a joke':
            joke, punchline = random.choice(self.additional_jokes)
            return f"{joke}\n{punchline}"

        # If no matching response is found, provide a generic response
        return "I'm sorry, I don't understand that."
