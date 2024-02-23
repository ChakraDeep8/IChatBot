import os
from dataclasses import dataclass
from typing import Literal
import streamlit as st
from IChatBot import Chatbot
import streamlit.components.v1 as components
import pandas as pd
import random

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


ms = st.session_state
if "themes" not in ms:
    ms.themes = {"current_theme": "light",
                 "refreshed": True,

                 "light": {"theme.base": "dark",
                           "theme.backgroundColor": "black",
                           "theme.primaryColor": "#c98bdb",
                           "theme.secondaryBackgroundColor": "#14061E",
                           "theme.textColor": "white",
                           "theme.textColor": "white",
                           "button_face": "🌜"},

                 "dark": {"theme.base": "light",
                          "theme.backgroundColor": "white",
                          "theme.primaryColor": "#5591f5",
                          "theme.secondaryBackgroundColor": "#7bd5db",
                          "theme.textColor": "#0a1464",
                          "button_face": "🌞"},
                 }


def ChangeTheme():
    previous_theme = ms.themes["current_theme"]
    tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
    for vkey, vval in tdict.items():
        if vkey.startswith("theme"): st._config.set_option(vkey, vval)

    ms.themes["refreshed"] = False
    if previous_theme == "dark":
        ms.themes["current_theme"] = "light"
    elif previous_theme == "light":
        ms.themes["current_theme"] = "dark"


btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"][
    "button_face"]
st.button(btn_face, on_click=ChangeTheme)

if ms.themes["refreshed"] == False:
    ms.themes["refreshed"] = True
    st.rerun()

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

st.markdown("<h2 style='margin-bottom: -10px;'>Hi, I am Chatty 🤗</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='margin-top: -10px;'>Feel free to chat with me..</h3>", unsafe_allow_html=True)
st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;'>", unsafe_allow_html=True)

# Load sample questions from dialog.txt using pandas
dialogs_df = pd.read_csv('dialogs.txt', sep='\t', header=None, names=['Question', 'Answer'])
question_ends_with_question_mark = dialogs_df[dialogs_df['Question'].str.endswith('?')]['Question'].tolist()
selected_questions = random.sample(question_ends_with_question_mark, k=100)

st.sidebar.title("Sample Questions")
#st.sidebar.markdown("Select a question 👇")
selected_question = st.sidebar.selectbox("Select a question&nbsp;&nbsp;👇", selected_questions)

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

