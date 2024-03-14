import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import emoji
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
            'ahaha': ["Ahaha!", "That's funny!", "Haha, good one!", "You got me laughing!"],
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
            'ohh': ["Ohh...", "I see...", "Got it!", "Interesting!"]
        }

        self.emoji_responses = {
            '😂': ["That's funny!", "LOL!", "Haha, good one!", "Hehe, you got me!", "I chuckled!"],
            '😊': ["I'm glad you're happy!", "That's wonderful!", "Smile, it's contagious!", "Spread the happiness!"],
            '🙂': ["I'm glad you're happy!", "That's wonderful!", "Smile, it's contagious!", "Spread the happiness!"],
            '😢': ["I'm sorry to hear that. Is there anything I can do?", "Sending hugs your way.",
                  "It's okay to feel sad sometimes.", "Things will get better!"],
            '😍': ["Wow, that's lovely!", "You're so in love!", "Heart eyes all the way!", "That's adorable!"],
            '😎': ["Cool as a cucumber!", "Looking sharp!", "Sunglasses emoji level cool!", "You're rocking it!"],
            '🤔': ["Hmm, let me think about that.", "Interesting question!", "Deep in thought...", "I ponder..."],
            '🥳': ["Woo-hoo, let's celebrate!", "Party time!", "Congratulations!", "That calls for a celebration!"],
            '👍': ["Thumbs up!", "Great job!", "You got it!", "Well done!"],
            '👏': ["Clap clap!", "Bravo!", "Well done!", "You're amazing!"],
            '🎉': ["Hooray!", "Let's celebrate!", "Congrats!", "Party time!"],
            '🤗': ["Big hug!", "Hugs!", "Sending you warm wishes!", "You're not alone!"],
            '😄': ["Smile, it's contagious!", "Cheer up!", "Be happy!", "Let's spread joy!"],
            '😤': ["Take a deep breath.", "Stay calm.", "Let's keep cool.", "Inhale, exhale."],
            '🙄': ["Rolling my eyes.", "Sigh...", "Hmm, interesting.", "What a surprise."],
            '😱': ["Oh no, that's scary!", "Yikes!", "That's terrifying!", "Take care!"],
            '🤣': ["ROFL!", "That's hilarious!", "HAHA!", "You're cracking me up!"],
            '😇': ["Angel emoji!", "You're too kind!", "Good vibes only!", "You're an angel!"],
            '🤩': ["Wow, that's amazing!", "You're awesome!", "Amazing!", "Fantastic!"],
            '😬': ["Oops!", "Awkward...", "My bad!", "Sorry about that!"],
            '🤫': ["Shh, it's a secret!", "Keep it quiet!", "Let's keep this between us!", "Confidential!"],
            # Add more emoji mappings and responses here
        }

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
