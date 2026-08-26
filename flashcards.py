import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import genanki
import random
import PyPDF2
import io


def scrape_url(url):
    """Fetch a webpage and return its main text, stripped of navigation and scripts."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/91.0.4472.124 Safari/537.36'

                      
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    paragraphs = soup.find_all('p')
    return ' '.join([p.get_text() for p in paragraphs])

def parse_page_range(text, total_pages):
    """Turn '1, 5-8, 27-30' into a list of page indices. Ignores anything out of range."""
    pages = set()

    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            for p in range(int(start), int(end) + 1):
                if 1 <= p <= total_pages:
                    pages.add(p - 1)
        else:
            p = int(part)
            if 1 <= p <= total_pages:
                pages.add(p - 1)

    return sorted(pages)


def extract_pdf(uploaded_file, page_range=None):
    """Read an uploaded PDF and return its text. page_range is a string like '1, 5-8'."""
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    total = len(reader.pages)

    if page_range:
        indices = parse_page_range(page_range, total)
    else:
        indices = range(total)

    pages = []
    for i in indices:
        text = reader.pages[i].extract_text()
        if text:
            pages.append(text)

    return ' '.join(pages)

def extract_pdf_pages(uploaded_file, page_range=None):
    """Like extract_pdf, but returns a list of (page_number, text) instead of one string."""
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    uploaded_file.seek(0)
    total = len(reader.pages)

    if page_range:
        indices = parse_page_range(page_range, total)
    else:
        indices = range(total)

    result = []
    for i in indices:
        text = reader.pages[i].extract_text()
        if text:
            result.append((i + 1, text))

    return result

def count_pdf_pages(uploaded_file):
    """Return the number of pages in an uploaded PDF."""
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    uploaded_file.seek(0)
    return len(reader.pages)

def generate_cards(content, api_key, num_cards, difficulty):
    """Send content to OpenAI and return a list of (question, answer) pairs."""
    client = OpenAI(api_key=api_key)

    prompt = f"""Create exactly {num_cards} flashcards at {difficulty} level from this content.

You MUST use this EXACT format for every card, no exceptions:
Q: your question here
A: your answer here

Rules:
- Never use 'Question:' or 'Answer:' or numbers
- Always start with Q: and A: only
- One blank line between each flashcard
- Questions should test understanding, not ask about URLs or sources
- Base questions only on the main content, not links or references

Content: {content[:20000]}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates educational flashcards."},
            {"role": "user", "content": prompt}
        ]
    )

    return parse_cards(response.choices[0].message.content)


def parse_cards(text):
    """Turn the model's text response into a list of (question, answer) pairs."""
    flashcards = []
    question = None

    for line in text.split('\n'):
        line = line.strip()
        if line.startswith("Q:"):
            question = line[2:].strip()
        elif line.startswith("A:") and question:
            answer = line[2:].strip()
            flashcards.append((question, answer))
            question = None

    return flashcards


def build_deck(flashcards, deck_name):
    """Build an Anki deck, write it to a .apkg file, and return the filename."""
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
        deck.add_note(genanki.Note(model=model, fields=[question, answer]))

    filename = f"{deck_name}.apkg"
    genanki.Package(deck).write_to_file(filename)
    return filename