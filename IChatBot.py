import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Download NLTK data and store it in the Streamlit app directory
nltk.download('punkt', download_dir='.')
nltk.download('stopwords', download_dir='.')
nltk.download('wordnet', download_dir='.')

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

        self.additional_jokes = [
        ("Why don't scientists trust atoms?", "Because they make up everything!"),
        ("What do you call a fake noodle?", "An impasta!"),
        ("Why did the scarecrow win an award?", "Because he was outstanding in his field!"),
        ]
        self.additional_short_expressions = {
            'lol': ["That's funny!", "LOL!", "Haha, good one!", "Hehe, you got me!", "I chuckled!"],
            'haha': ["Haha!", "That's hilarious!", "Funny one!", "You cracked me up!"],
            'hehe': ["Hehe!", "That's amusing!", "Nice one!", "You got me giggling!"],
            'rofl': ["ROFL!", "Oh, that's too funny!", "I'm rolling on the floor laughing!", "LOL, you're killing me!"],
            'lmao': ["LMAO!", "I'm laughing my *ss off!", "Haha, that's hilarious!", "You got me in stitches!"],
            'omg': ["OMG!", "Oh my gosh!", "Wow, really?", "That's incredible!"],
            'brb': ["BRB!", "Sure, take your time!", "No problem, I'll be here!", "Take your time, I'll wait!"],
            'btw': ["BTW!", "By the way!", "Just a heads up!", "Oh, and by the way!"],
            'gtg': ["GTG!", "Got to go!", "I need to run!", "Catch you later!"],
            'idk': ["IDK!", "I don't know!", "Not sure about that!", "Hmm, that's a tough one!"],
            'hmm': ["Hmm...", "Interesting!", "Let me think about that.", "I'm pondering..."],
            'hm': ["Hm...", "Hmm...", "I see.", "Interesting point."],
            'hmmm': ["Hmmm...", "Deep in thought...", "Let me consider that.", "Interesting question!"],
            'hmmmm': ["Hmmm...", "Deep in thought...", "Let me consider that.", "Interesting question!"],
            'hmmmmm': ["Hm...", "Hmm...", "I see.", "Interesting point."],
            'wtf': ["WTF!", "What the heck?", "Seriously?", "I'm shocked!"],
            'fyi': ["FYI!", "Just so you know!", "For your information!", "By the way!"],
            'ttyl': ["TTYL!", "Talk to you later!", "Catch you later!", "Until next time!"],
            'lolwut': ["LOLWUT?", "What on earth?", "That's unexpected!", "You surprised me!"],
            'np': ["NP!", "No problem!", "You're welcome!", "Anytime!"],
        }
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
        elif user_query.lower() == 'tell me a joke':
            joke, punchline = random.choice(self.additional_jokes)
            return f"{joke}\n{punchline}"

        else:
            return "I'm sorry, I don't understand that."
