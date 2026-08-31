import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import genanki
import random
import PyPDF2
import io
import os
import feedparser

HEADERS = {
    'User-Agent': 'AnkiFlashcardGenerator/1.0 (https://github.com/frostpine3004/anki-generator)'
}

def scrape_url(url):
    """Fetch a webpage and return its main text, stripped of navigation and scripts."""
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    paragraphs = soup.find_all('p')
    return ' '.join([p.get_text() for p in paragraphs])

def sample_content(content, limit=20000):
    """Take evenly spaced chunks across the whole text instead of just the start."""
    if len(content) <= limit:
        return content

    chunks = 6
    size = limit // chunks
    step = len(content) // chunks
    parts = []
    for i in range(chunks):
        start = i * step
        parts.append(content[start:start + size])
    return "\n[...]\n".join(parts)

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
    uploaded_file.seek(0)
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    uploaded_file.seek(0)
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
    uploaded_file.seek(0)
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
    uploaded_file.seek(0)
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    uploaded_file.seek(0)
    return len(reader.pages)

def fetch_podcast_transcript(rss_url):
    """Fetch RSS feed, find the latest episode, and return its transcript if available.

    Returns (transcript_text, info_string). transcript_text is None if nothing was found.
    """
    try:
        resp = requests.get(rss_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return None, f"Could not fetch the feed: {e}"

    feed = feedparser.parse(resp.content)

    if not feed.entries:
        return None, "No episodes found in this RSS feed."

    latest = feed.entries[0]
    title = latest.get("title", "Unknown episode")

    transcript_url = None
    if hasattr(latest, "podcast_transcript"):
        transcript_url = latest.podcast_transcript.get("url")

    for tag in latest.get("tags", []):
        if "transcript" in tag.get("term", "").lower():
            candidate = tag.get("scheme")
            if candidate and candidate.startswith("http"):
                transcript_url = candidate
                break

    if transcript_url:
        try:
            resp = requests.get(transcript_url, headers=HEADERS, timeout=10)
            if resp.ok:
                if "html" in resp.headers.get("content-type", ""):
                    text = BeautifulSoup(resp.text, "html.parser").get_text(separator=" ")
                else:
                    text = resp.text
                text = " ".join(text.split())
                if len(text) > 200:
                    return text, title
        except Exception:
            pass

    summary = latest.get("summary", "")
    if summary and len(summary) > 2000:
        clean = BeautifulSoup(summary, "html.parser").get_text(separator=" ").strip()
        return clean, f"{title} (show notes only — no transcript found)"

    return None, (
        f"No transcript found for '{title}'. "
        "About 30–40% of podcasts publish transcripts. "
        "You can paste the transcript manually below."
    )

def load_prompt(name):
    """Read a prompt file from the prompts folder."""
    path = os.path.join(os.path.dirname(__file__), "prompts", f"{name}.txt")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()
    
def extract_concepts(content, api_key, count):
    """Ask the model for the key concepts in the text, one to three words each."""
    client = OpenAI(api_key=api_key)
    rules = load_prompt("concepts")

    prompt = f"""List the {count} most important concepts in the text below.

{rules}

Text:
{sample_content(content)}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    concepts = []
    for line in response.choices[0].message.content.split('\n'):
        line = line.strip().lstrip("0123456789.-• ")
        if line:
            concepts.append(line)

    return concepts

def generate_cloze_from_concepts(concepts, content, api_key):
    """Write one cloze card per concept, with the concept as the blank."""
    client = OpenAI(api_key=api_key)
    concept_list = "\n".join(concepts)
    rules = load_prompt("cloze")

    prompt = f"""Write one cloze card for each concept listed below, using the source text.

{rules}

Concepts:
{concept_list}

Source text:
{content[:20000]}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return parse_cards(response.choices[0].message.content)

def generate_cards(content, api_key, num_cards, difficulty, card_type="Q/A cards"):
    """Send content to OpenAI and return a list of (question, answer) pairs."""
    client = OpenAI(api_key=api_key)

    if card_type == "Cloze cards":
        concepts = extract_concepts(content, api_key, num_cards * 2)
        return generate_cloze_from_concepts(concepts, content, api_key)

    prompt_files = {
        "Q/A cards": "qa",
        "Cloze cards": "cloze",
        "Both": "both",
    }
    card_instruction = load_prompt(prompt_files.get(card_type, "qa"))
    shared_rules = load_prompt("shared_rules")

    prompt = f"""Create exactly {num_cards * 2} flashcards at {difficulty} level from this content.

{card_instruction}
{shared_rules}

Content: {sample_content(content)}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates educational flashcards."},
            {"role": "user", "content": prompt}
        ]
    )

    return parse_cards(response.choices[0].message.content)

def parse_cards(text):
    """Parse both Q/A and cloze formats from the model's response."""
    flashcards = []
    question = None
    cloze_sentence = None

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith("Q:"):
            question = line[2:].strip()

        elif line.startswith("A:") and question:
            answer = line[2:].strip()
            flashcards.append((question, answer))
            question = None

        elif line.startswith("CARD:"):
            cloze_sentence = line[5:].strip()

        elif line.startswith("ANSWER:") and cloze_sentence:
            flashcards.append((cloze_sentence, line[7:].strip()))
            cloze_sentence = None

    return flashcards

def answer_leaks(sentence, answer):
    """True if the answer, or a significant word from it, appears in the sentence."""
    sentence_lower = sentence.lower()
    for word in answer.lower().split():
        if len(word) > 3 and word in sentence_lower:
            return True
    return False


def review_cards(cards, api_key, card_type="Q/A cards"):
    """Ask the model to check its own cards and drop the weak ones."""
    if not cards:
        return cards, 0

    client = OpenAI(api_key=api_key)
    card_list = "\n".join(f"{i+1}. {q} → {a}" for i, (q, a) in enumerate(cards))

    if card_type == "Cloze cards":
        criteria = """- The answer appears in the sentence, including as part of a name or compound term
- The answer can be guessed from grammar or sentence structure alone
- The blank covers only part of a multi-word term
- The answer is a single common word that could be guessed without reading the source"""
    else:
        criteria = """- The question is open-ended and could be answered many different ways
- The question asks "what happened" or "what challenges" rather than about a specific concept
- The answer merely restates the question without adding information
- The answer is a vague quantifier or relation such as "a major factor" rather than a specific thing
- The answer could be guessed from general knowledge alone"""

    prompt = f"""Review these flashcards and decide which to keep.

Reject a card if:
{criteria}
- The answer is a publication year, book title or other bibliographic detail
- The question or answer is empty or a placeholder

Reply with ONLY the numbers of the cards to keep, separated by commas.
Example: 1, 3, 4

Cards:
{card_list}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    keep = []
    for part in response.choices[0].message.content.split(','):
        part = part.strip().rstrip('.')
        if part.isdigit():
            index = int(part) - 1
            if 0 <= index < len(cards):
                keep.append(cards[index])

    if card_type == "Cloze cards":
        keep = [(q, a) for q, a in keep if not answer_leaks(q, a)]

    return keep, len(cards) - len(keep)


def fix_grammar(cards, api_key):
    """Check each cloze sentence reads correctly when the blank is filled."""
    if not cards:
        return cards

    client = OpenAI(api_key=api_key)
    card_list = "\n".join(f"{i+1}. {q} ||| {a}" for i, (q, a) in enumerate(cards))

    prompt = f"""Each line below is a sentence with a blank marked [...], followed by ||| and the word that belongs in the blank.

For each line, give the correct inflected form of that word for the position of
the blank in that sentence. If the word is already correct, return it unchanged.

Return only the word itself. Do not return the sentence. Do not explain.

Return one line per card, in the same order:
number. word

Cards:
{card_list}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    fixed = []
    for i, line in enumerate(response.choices[0].message.content.split('\n')):
        line = line.strip().lstrip("0123456789. ")
        if not line or i >= len(cards):
            continue
        fixed.append((cards[i][0], line))

    return fixed if len(fixed) == len(cards) else cards

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