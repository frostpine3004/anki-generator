import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import genanki
import random

# Page config
st.set_page_config(
    page_title="AI Flashcard Generator",
    page_icon="🃏",
    layout="centered"
)

# Header
st.title("🃏 AI Flashcard Generator")
st.subheader("Turn any webpage into Anki flashcards instantly!")

# Security notice
st.info("🔒 Your API key is never stored. It's only used for your current session.")

# API key input
api_key = st.text_input(
    "Enter your OpenAI API key",
    type="password",
    placeholder="sk-proj-..."
)

# URL input
url = st.text_input(
    "Enter a webpage URL",
    placeholder="https://en.wikipedia.org/wiki/..."
)

# Number of cards
num_cards = st.slider("Number of flashcards", min_value=3, max_value=20, value=5)

# Difficulty level
difficulty = st.selectbox(
    "Difficulty level",
    ["Beginner", "Intermediate", "Advanced"]
)

# Deck name
deck_name = st.text_input("Deck name", value="My AI Flashcards")

# Generate button
if st.button("🚀 Generate Flashcards"):
    if not api_key:
        st.error("Please enter your OpenAI API key!")
    elif not url:
        st.error("Please enter a URL!")
    else:
        with st.spinner("Downloading webpage..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs])

        with st.spinner("Generating flashcards with AI..."):
            client = OpenAI(api_key=api_key)
            prompt = f"""Create {num_cards} flashcards at {difficulty} level from this content.
Format each flashcard exactly like this:
Q: question here
A: answer here

Content: {content[:4000]}"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates educational flashcards."},
                    {"role": "user", "content": prompt}
                ]
            )
            flashcards_text = response.choices[0].message.content

        # Parse flashcards
        flashcards = []
        lines = flashcards_text.split('\n')
        question = None
        for line in lines:
            if line.startswith("Q:"):
                question = line[2:].strip()
            elif line.startswith("A:") and question:
                answer = line[2:].strip()
                flashcards.append((question, answer))
                question = None

        # Show flashcards
        st.success(f"✅ Generated {len(flashcards)} flashcards!")
        for i, (q, a) in enumerate(flashcards, 1):
            with st.expander(f"Flashcard {i}: {q[:50]}..."):
                st.write(f"**Question:** {q}")
                st.write(f"**Answer:** {a}")

        # Create Anki deck
        model_id = random.randrange(1 << 30, 1 << 31)
        model = genanki.Model(
            model_id,
            'Simple Model',
            fields=[{'name': 'Question'}, {'name': 'Answer'}],
            templates=[{
                'name': 'Card 1',
                'qfmt': '{{Question}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
            }]
        )
        deck_id = random.randrange(1 << 30, 1 << 31)
        deck = genanki.Deck(deck_id, deck_name)
        for question, answer in flashcards:
            note = genanki.Note(model=model, fields=[question, answer])
            deck.add_note(note)

        # Save and download
        filename = f"{deck_name}.apkg"
        genanki.Package(deck).write_to_file(filename)
        with open(filename, "rb") as f:
            st.download_button(
                label="⬇️ Download Anki Deck",
                data=f,
                file_name=filename,
                mime="application/octet-stream"
            )