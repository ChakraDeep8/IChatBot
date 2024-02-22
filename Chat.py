import os
from dataclasses import dataclass
from typing import Literal
import streamlit as st
from IChatBot import Chatbot
import streamlit.components.v1 as components
import pandas as pd

static_path = os.path.join(os.path.dirname(__file__), "static")

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str

def load_css():
    css_file = os.path.join(static_path, "styles.css")
    with open(css_file, "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        st.session_state.conversation = Chatbot()
    if "human_prompt" not in st.session_state:
        st.session_state.human_prompt = ""  # Initialize human prompt

def on_click_callback():
    human_prompt = st.session_state.human_prompt
    llm_response = st.session_state.conversation.generate_response(human_prompt)
    st.session_state.history.append(Message("human", human_prompt))
    st.session_state.history.append(Message("ai", llm_response))
    # Clear the input prompt after submission
    st.session_state.human_prompt = ""

load_css()
initialize_session_state()

import streamlit as st

st.markdown("<h2 style='margin-bottom: -10px;'>Hi, I am Chatty 🤗</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='margin-top: -10px;'>Feel free to chat with me..</h3>", unsafe_allow_html=True)
st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)


# Load sample questions from dialog.txt using pandas
dialogs_df = pd.read_csv('dialogs.txt', sep='\t', header=None, names=['Question', 'Answer'])
sample_questions = dialogs_df['Question'].sample(n=20, random_state=42).tolist()

st.sidebar.title("Sample Questions")
st.sidebar.markdown("Select a question to suggest to the user:")
selected_question = st.sidebar.selectbox("", sample_questions)

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
credit_card_placeholder = st.empty()

with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
<div class="chat-row 
    {'' if chat.origin == 'ai' else 'row-reverse'}">
    <img class="chat-icon" src="app/static/{
        'ai_icon.png' if chat.origin == 'ai'
        else 'user_icon.png'}"
         width=42 height=42>
    <div class="chat-bubble
    {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
        &#8203;{chat.message}
    </div>
</div>
        """
        st.markdown(div, unsafe_allow_html=True)

    for _ in range(3):
        st.markdown("")

    for _ in range(3):
        st.markdown("")

with prompt_placeholder:
    st.markdown("**Chat**")
    cols = st.columns((6, 1))
    # Set the default value of the text input to the selected question
    cols[0].text_input(
        "Chat",
        value=selected_question,
        label_visibility="collapsed",
        key="human_prompt",
    )
    cols[1].form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback,
    )
    # Display toast message suggesting checking out sample questions
    if st.session_state.human_prompt == "":
        st.error("👈 Check out the sample questions on the sidebar for ideas!")
