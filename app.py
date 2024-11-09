import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
from pytube import Search
import random

# Load DialoGPT model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Set up Streamlit page configuration
st.set_page_config(page_title="Grief and Loss Support Bot", page_icon="🌿", layout="centered")
st.markdown("""
    <style>
    .css-1d391kg { background-color: #F3F7F6; }
    .css-ffhzg2 { font-size: 1.5em; font-weight: 500; color: #4C6D7D; }
    .stTextInput>div>div>input { background-color: #D8E3E2; }
    .stButton>button { background-color: #A9D0B6; color: white; border-radius: 5px; }
    .stButton>button:hover { background-color: #8FB79A; }
    .stTextInput>div>label { color: #4C6D7D; }
    </style>
""", unsafe_allow_html=True)

# Title and introduction to the bot
st.title("Grief and Loss Support Bot 🌿")
st.subheader("Your compassionate companion in tough times 💚")

# User input
user_input = st.text_input("Share what's on your mind...", placeholder="Type here...", max_chars=500)

# Store previous responses to check for repetition
if 'previous_responses' not in st.session_state:
    st.session_state.previous_responses = []

# Function to generate a more empathetic and focused response using DialoGPT
def generate_response(user_input):
    # Encode the input text and generate a response
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id, temperature=0.7, top_k=50, repetition_penalty=1.2)
    
    # Decode the response to text
    chat_history_ids = chat_history_ids[:, bot_input_ids.shape[-1]:]  # remove the input from the response
    bot_output = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
    
    # Build a more empathetic and thoughtful response
    response = f"I’m really sorry you're feeling like this. It’s completely normal to feel overwhelmed when you're facing a heavy workload. It’s important to acknowledge how you feel and not keep it bottled up. Sometimes, stress and emotional exhaustion can build up, and it’s okay to let yourself feel those emotions."

    # Add coping strategies based on the situation
    if "workload" in user_input.lower():
        response += "\n\nWhen the workload feels too heavy, it can be helpful to break tasks down into smaller, more manageable steps. Focus on one thing at a time, and remember that it’s okay to take breaks when needed. Asking for support from colleagues or friends is also a good way to lighten the load."

    # Add general supportive message
    response += "\n\nYou're doing your best, and that’s all anyone can ask for. Please take care of yourself and know that it’s okay to take a step back when things feel too much. Your well-being is the most important thing."

    # Suggest a productive activity based on detected keywords
    if any(keyword in user_input.lower() for keyword in ["lonely", "lost", "sad", "overwhelmed"]):
        st.info("Here's a suggestion to help you cope:")

        # List of activities
        hobbies = ["journaling", "yoga", "painting", "exercise", "meditation"]
        activity = st.selectbox("Choose an activity you'd like to try:", hobbies)
        
        # Search YouTube for videos related to the selected activity
        try:
            search = Search(activity)
            search_results = search.results[:3]  # limit results to 3 videos
            st.write(f"Searching for videos related to {activity}...")  # Debugging line
            
            if not search_results:
                st.write(f"No results found for '{activity}'. Please try again.")
            else:
                st.write(f"Found {len(search_results)} video(s) related to '{activity}'!")
                for video in search_results:
                    st.write(f"[{video.title}]({video.watch_url})")
        except Exception as e:
            st.write(f"An error occurred while searching for videos: {str(e)}")
            st.write("Sorry, I couldn't fetch videos at the moment.")

    # Crisis resources
    crisis_keywords = ["help", "suicide", "depressed", "emergency", "hurt", "lost"]
    if any(keyword in user_input.lower() for keyword in crisis_keywords):
        st.warning("It seems like you might be in distress. Please reach out to a crisis hotline or a trusted individual.")
        st.write("[Find emergency resources here](https://www.helpguide.org/find-help.htm)")

    return response

# Check if the user has typed something
if user_input:
    # Generate the empathetic response
    response = generate_response(user_input)
    
    # Store and show the new response
    st.session_state.previous_responses.append(response)
    st.text_area("Bot's Response:", response, height=250)

    # Text-to-speech output (optional)
    tts = gTTS(response, lang='en')
    audio_file = "response.mp3"
    tts.save(audio_file)
    st.audio(audio_file, format="audio/mp3")
