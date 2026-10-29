import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import genanki
import random
import re
import PyPDF2
import io
import os
import feedparser
import subprocess
import tempfile
from collections import Counter

# Steps of the last generation run, shown in the app when debugging is on
debug_log = []

def debug(step, text):
    debug_log.append((step, text))

WHISPER_COST_PER_MINUTE = 0.006
WHISPER_MAX_BYTES = 25 * 1024 * 1024

# Choosing cloze sentences and reviewing cards takes judgment that gpt-4o-mini lacks: it
# chose a sentence the prompt gives as a bad example, and the review kept it. These prompts
# are short, so a stronger model costs little. Other steps still use gpt-4o-mini.
# It thinks before answering, and at the default effort a run takes minutes, so the effort
# is set low. A model that doesn't reason, such as gpt-4o-mini, rejects this setting.
JUDGE_MODEL = "gpt-5-mini"
JUDGE_EFFORT = "low"

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
    return "\n\n".join([p.get_text() for p in paragraphs])

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

def filter_by_concept(text, concept, min_stem=4, context=0):
    """Keep only paragraphs matching any of the search terms.

    Matching is done on truncated stems so that inflected forms
    of the same word are treated as equivalent.

    Returns: (filtered_text, matched_count, total_count)
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    total = len(paragraphs)

    # Commas separate alternative terms, spaces separate words within a term.
    terms = [t.strip() for t in concept.split(",") if t.strip()]
    stems = []
    for term in terms:
        for word in term.split():
            if len(word) >= min_stem:
                stems.append(word[:min_stem].lower())

    if not stems:
        return text, total, total

    keep = set()
    for i, paragraph in enumerate(paragraphs):
        lowered = paragraph.lower()
        if any(stem in lowered for stem in stems):
            for j in range(i - context, i + context + 1):
                if 0 <= j < total:
                    keep.add(j)

    filtered = "\n\n".join(paragraphs[i] for i in sorted(keep))
    return filtered, len(keep), total

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

    return "\n\n".join(pages)

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

def parse_duration(value):
    """Turn an iTunes duration ('3600' or '01:02:03') into seconds. 0 if unknown."""
    value = str(value).strip()
    if not value:
        return 0
    if ':' in value:
        seconds = 0
        for part in value.split(':'):
            if not part.isdigit():
                return 0
            seconds = seconds * 60 + int(part)
        return seconds
    return int(value) if value.isdigit() else 0


def fetch_episodes(rss_url, limit=10):
    """Return recent episodes as a list of dicts with title, audio_url and duration."""
    try:
        resp = requests.get(rss_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return [], f"Could not fetch the feed: {e}"

    feed = feedparser.parse(resp.content)
    if not feed.entries:
        return [], "No episodes found in this RSS feed."

    episodes = []
    for entry in feed.entries[:limit]:
        audio_url = None
        for enc in entry.get("enclosures", []):
            if "audio" in enc.get("type", ""):
                audio_url = enc.get("href") or enc.get("url")
                break
        if not audio_url:
            continue
        episodes.append({
            "title": entry.get("title", "Unknown episode"),
            "audio_url": audio_url,
            "duration": parse_duration(entry.get("itunes_duration", "")),
        })

    if not episodes:
        return [], "No downloadable audio found in this feed."
    return episodes, f"Found {len(episodes)} episodes."

def trim_audio(path, out_dir, start_min=0, limit_min=0):
    """Cut an audio file to a time range. Returns the new path."""
    out = os.path.join(out_dir, "trimmed.mp3")
    cmd = ["ffmpeg", "-i", path]
    if start_min:
        cmd += ["-ss", str(int(start_min * 60))]
    if limit_min:
        cmd += ["-t", str(int(limit_min * 60))]
    cmd += ["-c", "copy", out]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is required for trimming but is not installed.")

    return out if os.path.exists(out) else path

    
def split_audio(path, out_dir, minutes=10):
    """Split an audio file into chunks small enough for the Whisper API."""
    if os.path.getsize(path) <= WHISPER_MAX_BYTES:
        return [path]

    pattern = os.path.join(out_dir, "part%03d.mp3")
    try:
        subprocess.run(
            ["ffmpeg", "-i", path, "-f", "segment",
             "-segment_time", str(minutes * 60), "-c", "copy", pattern],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is required for episodes over 25 minutes but is not installed.")

    parts = sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir) if f.startswith("part")
    )
    return parts or [path]

def transcribe_episode(audio_url, api_key, progress=None, start_min=0, limit_min=0):
    """Download an episode and transcribe it with Whisper. Returns the full text."""
    client = OpenAI(api_key=api_key)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "episode.mp3")
        with requests.get(audio_url, headers=HEADERS, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for block in r.iter_content(chunk_size=1 << 20):
                    f.write(block)

        if start_min or limit_min:
            path = trim_audio(path, tmp, start_min, limit_min)

        parts = split_audio(path, tmp)

        texts = []
        for i, part in enumerate(parts):
            if progress:
                progress(i, len(parts))
            with open(part, "rb") as f:
                result = client.audio.transcriptions.create(model="whisper-1", file=f)
            texts.append(result.text)

    return " ".join(texts)

def load_prompt(name):
    """Read a prompt file from the prompts folder."""
    path = os.path.join(os.path.dirname(__file__), "prompts", f"{name}.txt")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()
    
def extract_concepts(content, api_key, count, exclude=None):
    """Ask the model for the key concepts in the text, one to three words each."""
    client = OpenAI(api_key=api_key)
    rules = load_prompt("concepts")
    avoid = ""
    asked = count
    if exclude:
        avoid = "\nDo not include these concepts or any form of them:\n" + "\n".join(exclude) + "\n"
        # The model often lists excluded concepts anyway, so ask for extra and drop them below
        asked += len(exclude)

    prompt = f"""List the {asked} most important concepts in the text below.

{rules}
{avoid}

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

    # The prompt asks the model to skip the main topic, but it doesn't reliably,
    # so drop it here by counting how often the words occur in the text.
    sentences = split_sentences(sample_content(content))
    kept = []
    for c in concepts:
        if any(answer_matches_concept(c, e) or answer_matches_concept(e, c) for e in exclude or []):
            debug("Concepts", f"DROPPED (already tried): {c}")
        elif is_main_topic(c, sentences):
            debug("Concepts", f"DROPPED (main topic): {c}")
        elif len(kept) < count:
            debug("Concepts", f"kept: {c}")
            kept.append(c)
    return kept

def source_sentences(text):
    """Split text into sentences, keeping the original wording.

    Citation markers such as [54] are removed first. Wikipedia puts them right
    after the full stop, as in "signatures).[54] While", which hides the break.
    """
    text = re.sub(r"\[(?:\d+|[a-z]|note \d+|citation needed)\]", "", text, flags=re.IGNORECASE)
    # PDFs sometimes run a heading into the previous sentence, as in "legislation.Digital".
    # An abbreviation such as "U.S." or "i.e." doesn't end a sentence.
    parts = re.split(r"(?<=[.!?])(?<!\.\w\.)\s+|(?<=[a-z][.!?])(?=[A-Z])|\n\s*\n", text)
    sentences = [" ".join(p.split()) for p in parts if p.strip()]
    # PDF extraction leaves a space before punctuation, as in "In theory , electronic money"
    sentences = [re.sub(r"\s+([,.;:!?)])", r"\1", s) for s in sentences]
    return rejoin_split_words(sentences)

def rejoin_split_words(sentences):
    """Join words that PDF extraction split in two, as in "anonymit y" and "isomo rphism".

    A pair is joined only if the result is a word used elsewhere in the text and one
    part occurs nowhere else. A lone last letter is also joined if the part before it
    never occurs on its own, unless the letter is used as a variable elsewhere: in a
    math text "a system of n equations" or "a polynomial of degree n" must stay apart.
    """
    # Links are left out, or "Fox News" would be joined because of "www.foxnews.com"
    joined = re.sub(r"\S*(?:https?://|www\.)\S*", " ", " ".join(sentences))
    counts = Counter(w.lower() for w in re.findall(r"[^\W\d_]+", joined))
    alone = {w.lower() for w in re.findall(r"\b[^\W\d_]+\b(?! [b-hj-z]\b)", joined)}
    variables = {letter for part, letter in re.findall(r"\b([^\W\d_]+) ([b-hj-z])\b", joined)
                 if part.lower() in alone and (part + letter).lower() not in counts}

    def rejoin(match):
        first, second = match.groups()
        whole = (first + second).lower()
        if len(first) < 3:
            join = False
        elif len(second) == 1:
            join = second in "bcdefghjklmnopqrstuvwxyz" and (
                whole in counts or (first.lower() not in alone and second not in variables))
        else:
            join = whole in counts and 1 in (counts[first.lower()], counts[second.lower()])
        return first if join else first + " "

    # Only the first word and the space are replaced, so each word is also checked with the next
    return [re.sub(r"\b([^\W\d_]+) (?=([^\W\d_]+)\b)", rejoin, s) for s in sentences]

def split_sentences(text, min_words=6):
    """Split text into sentences, each a list of lowercase words.

    Fragments shorter than min_words are left out. PDFs are full of headings,
    captions and references, which would otherwise make every word look rare.
    """
    sentences = [re.findall(r"\w+", p.lower()) for p in source_sentences(text)]
    return [s for s in sentences if len(s) >= min_words]

def shares_stem(a, b):
    """True if two words share most of their beginning, so inflections and compounds match.

    "rahanpesudirektiivi" shares a stem with "rahanpesun", "money" does not with "monetary".
    """
    shortest = min(len(a), len(b))
    if shortest < 4:
        return False
    needed = min(shortest, max(5, int(shortest * 0.8)))
    return len(os.path.commonprefix([a, b])) >= needed

def is_main_topic(concept, sentences, share=0.3):
    """True if most of the concept's words occur in a large share of the sentences.

    Those words are the text's main topic, which appears nearly everywhere and
    cannot be hidden in a cloze card. "anti-money laundering" is dropped from a
    text about money laundering, but "electronic money" is kept.
    """
    # Leave out abbreviations in parentheses, as in "Anti-Money Laundering (AML)"
    plain = re.sub(r"\(.*?\)", "", concept.lower())
    words = [w for w in re.findall(r"\w+", plain) if len(w) >= 4]
    if not words or not sentences:
        return False

    frequent = 0
    for word in words:
        hits = sum(1 for s in sentences if any(shares_stem(word, other) for other in s))
        if hits >= share * len(sentences):
            frequent += 1
    return frequent * 2 > len(words)

def find_concept(sentence, concept):
    """Return (start, end) of the concept in the sentence, or None.

    Each word may carry an inflected ending, so "rahanpesun" is found for
    "Rahanpesu" and "politically exposed persons" for "Political Exposed Persons".
    """
    concept_words = re.findall(r"[\w-]+", concept.lower())
    tokens = list(re.finditer(r"[\w-]+", sentence))
    for i in range(len(tokens) - len(concept_words) + 1):
        window = tokens[i:i + len(concept_words)]
        if all(t.group().lower() == c or shares_stem(t.group().lower(), c)
               for t, c in zip(window, concept_words)):
            return window[0].start(), window[-1].end()
    return None

def usable_sentence(sentence, span):
    """True if the sentence can become a card once the concept at span is blanked.

    Rejects reference list entries with links, fragments such as
    'Jurado-Rodriguez specialized in "smurfing".' that leave too little context,
    and glossary lines such as "Transaction laundering: when a merchant...".
    """
    if re.search(r"https?://|www\.", sentence):
        return False
    # A title from the reference list, as in '"The Rise in Popularity of Cryptocurrencies".'
    if re.match(r"^[\"“].*[\"”]\.?$", sentence):
        return False
    rest = sentence[:span[0]] + sentence[span[1]:]
    if len(re.findall(r"\w+", rest)) < 6:
        return False
    # A heading before a colon, as in "Structuring: Often known as smurfing, is..."
    if re.match(r"^[\w\s-]{1,40}:\s", sentence):
        return False
    # "one of these sites", "this method" or "in this case" points back to an earlier sentence
    if re.search(r"\b(these|those)\b|\bthis (method|process|practice|technique|scheme|way|type|kind|case)\b", sentence, re.IGNORECASE):
        return False
    # So does a subject such as "This scalar" or "It", which a cut can't leave out
    if re.match(r"(this|such|it|its|another|other|he|she|they|their)\b", sentence, re.IGNORECASE):
        return False
    # A symbol such as V, f or v1 is defined in an earlier sentence. "n" is allowed, as in
    # "a system of n linear equations", and so are "Cramer's" and abbreviations such as "U.S."
    plain = re.sub(r"['’]s\b|\b(?:\w\.){2,}", "", sentence)
    if re.search(r"\b(?![aAIn]\b)[^\W\d_]\b|\b[a-z]\d+\b", plain):
        return False
    # "known as smurfing" names the concept instead of using it
    if re.search(r"\b(known as|called|termed|referred to as)\W*$", sentence[:span[0]], re.IGNORECASE):
        return False
    return not sentence[span[1]:].lstrip().startswith((":", "–", "—"))

def concept_candidates(concepts, content, max_words=80, per_concept=5):
    """For each concept, the shortest usable source sentences that contain it.

    Long sentences are included too, since the model may cut a shorter part out of them.
    """
    sentences = [s for s in source_sentences(content) if len(s.split()) <= max_words]
    candidates = []
    for concept in concepts:
        found = []
        for sentence in sentences:
            span = find_concept(sentence, concept)
            if span and usable_sentence(sentence, span):
                found.append((sentence, span))
        found.sort(key=lambda f: len(f[0]))
        if not found:
            debug("Cloze cards", f"no usable source sentence for: {concept}")
        candidates += [(concept, sentence, span) for sentence, span in found[:per_concept]]
    return candidates

def blanked(sentence, span):
    return sentence[:span[0]] + "[...]" + sentence[span[1]:]

# Words that bind to the word after them, so a cut right after one breaks a phrase
PHRASE_WORDS = {"a", "an", "the", "of", "with", "in", "on", "by", "for", "to", "from",
                "at", "as", "into", "than", "their", "its", "his", "her", "our", "my"}

# Words that start a new clause, so a cut may end right before one
CLAUSE_WORDS = {"and", "but", "or", "while", "whereas", "although", "because", "which", "where", "when"}

# Words that point back to an earlier sentence, so a card must not start with one
BACK_WORDS = {"this", "these", "those", "such", "it", "he", "she", "they", "its", "their", "his", "her",
              "another", "additionally", "however", "moreover", "furthermore", "also", "thus",
              "therefore", "hence", "equivalently", "similarly", "conversely"}
BACK_PHRASES = ("in other words", "more precisely", "for example", "for instance")

def cut_card(sentence, span, concept, text, max_words=35):
    """Return the card text if its words are a continuous part of the blanked sentence, else None.

    The model may shorten a sentence but not reword it, so the words must appear in
    the source in the same order. Case and punctuation may differ, and the card is
    copied from the source, so it keeps the source's own punctuation.
    """
    source = blanked(sentence, span)
    token = r"\[\.\.\.\]|\w+"
    source_tokens = list(re.finditer(token, source))
    words = [w.lower() for w in re.findall(token, text)]
    if words.count("[...]") != 1:
        return None

    lowered = [t.group().lower() for t in source_tokens]
    for i in range(len(lowered) - len(words) + 1):
        if lowered[i:i + len(words)] == words:
            break
    else:
        return None

    # A cut must not split a phrase: "with [...] laws" cut to "[...] laws" loses its preposition
    if i > 0 and lowered[i - 1] in PHRASE_WORDS:
        return None
    if lowered[i + len(words) - 1] in PHRASE_WORDS:
        return None
    if words[0] in BACK_WORDS or " ".join(words[:3]).startswith(BACK_PHRASES):
        return None

    end = source_tokens[i + len(words) - 1].end()
    # Keep a closing quote or bracket that belongs to the last word
    while end < len(source) and source[end] in "\"')”’":
        end += 1
    # A cut must end with its clause: "vectors that span V" cut to "vectors that span" loses its object
    if source[end:].lstrip()[:1].isalnum() and lowered[i + len(words)] not in CLAUSE_WORDS:
        return None
    card = source[source_tokens[i].start():end]
    card = card[0].upper() + card[1:]
    if not card.endswith((".", "!", "?")):
        card += "."

    others = len(words) - 1
    # The concept may occur again later in the source sentence; the cut must leave it out
    if others < 6 or len(card.split()) > max_words or find_concept(card, concept):
        return None
    return card

def generate_cloze_from_concepts(concepts, content, api_key):
    """Write one cloze card per concept from a source sentence, with the concept as the blank.

    The model chooses a sentence and may cut a shorter part out of a long one. The
    text is checked against the source and the blank is made here, so a card can
    never state something the source doesn't, and the answer is always the concept
    in the form the source uses.
    """
    candidates = concept_candidates(concepts, content)
    if not candidates:
        return []

    def numbered(i, concept, sentence, span):
        line = f"{i+1}. [{concept}] {blanked(sentence, span)}"
        words = len(sentence.split())
        if words > 30:
            line += f" ({words} words, too long: cut it)"
        return line

    candidate_list = "\n".join(numbered(i, *c) for i, c in enumerate(candidates))
    rules = load_prompt("cloze")

    prompt = f"""Below are candidate cloze cards: sentences from a source text with a concept
replaced by [...]. The concept is shown in brackets before each sentence.

Choose the best sentence for each concept. Choose none for a concept if every
sentence for it breaks a rule. A sentence marked "too long" can still be chosen,
but you must cut it.

{rules}

Reply with one line per chosen card, the number and the card text:
number: card text

Example:
4: Organized crime received a boost from [...] and a large source of new funds

Candidates:
{candidate_list}"""

    client = OpenAI(api_key=api_key)
    cards = []
    used = set()

    def ask(prompt):
        """Send the prompt and keep the cuts that pass. Returns the rejected ones."""
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            reasoning_effort=JUDGE_EFFORT,
            messages=[{"role": "user", "content": prompt}]
        )
        rejected = []
        for line in response.choices[0].message.content.split('\n'):
            match = re.match(r"^\s*(\d+)\s*[:.)]\s*(.+)$", line)
            if not match or not 0 < int(match.group(1)) <= len(candidates):
                continue
            number = int(match.group(1))
            concept, sentence, span = candidates[number - 1]
            if concept in used:
                continue
            card = cut_card(sentence, span, concept, match.group(2))
            if not card:
                debug("Cloze cards", f"REJECTED cut (not word for word, ends mid-phrase, starts by pointing back, too long or too short): {match.group(2)}\nsource: {sentence}")
                rejected.append((number, match.group(2)))
                continue
            used.add(concept)
            answer = sentence[span[0]:span[1]]
            debug("Cloze cards", f"chosen: {card} → {answer}")
            cards.append((card, answer))
        return rejected

    rejected = ask(prompt)

    # The model often returns a long sentence whole, so ask once more for just those
    if rejected:
        retry_list = "\n".join(f"{n}. {blanked(*candidates[n - 1][1:])}\n   your card: {text}" for n, text in rejected)
        retry = f"""These cloze cards were rejected. A card must be at most 30 words, keep
at least 6 words besides [...], and be copied word for word from its source
sentence as one continuous part.

For each, cut a continuous part around [...] out of the source sentence that
makes sense on its own. Cut at a clause boundary: end where the sentence ends or
at a comma, never before the rest of a phrase. Leave out a starting word such
as "while", "but" or "however" when the rest is a complete statement. Do not
start with "This", "Such" or another word that points back to an earlier
sentence. Copy the words exactly; do not fix spelling or change anything.

Reply with one line per card, the same number and the new card text:
number: card text

Cards:
{retry_list}"""
        ask(retry)

    for concept in concepts:
        if concept not in used and any(c == concept for c, _, _ in candidates):
            debug("Cloze cards", f"no sentence chosen for: {concept}")
    return cards

def generate_cards(content, api_key, num_cards, difficulty, card_type="Q/A cards", exclude=None, tried_concepts=None):
    """Send content to OpenAI and return a list of (question, answer) pairs.

    exclude: cards from earlier rounds that must not be repeated.
    tried_concepts: cloze concepts from earlier rounds. New concepts are added to it,
    so a concept that found no usable sentence isn't proposed again.
    """
    if card_type == "Cloze cards":
        if tried_concepts is None:
            tried_concepts = []
        used = [a for q, a in exclude] if exclude else []
        concepts = extract_concepts(content, api_key, num_cards * 3, used + tried_concepts or None)
        tried_concepts += [c for c in concepts if c not in tried_concepts]
        return generate_cloze_from_concepts(concepts, content, api_key)

    client = OpenAI(api_key=api_key)
    prompt_files = {
        "Q/A cards": "qa",
        "Both": "both",
    }
    card_instruction = load_prompt(prompt_files.get(card_type, "qa"))
    shared_rules = load_prompt("shared_rules")
    avoid = ""
    if exclude:
        avoid = "\nDo not repeat these cards or ask about the same facts:\n" + "\n".join(f"{q} → {a}" for q, a in exclude) + "\n"

    prompt = f"""Create exactly {num_cards * 2} flashcards at {difficulty} level from this content.

{card_instruction}
{shared_rules}
{avoid}

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

    return [(strip_quotes(q), strip_quotes(a)) for q, a in flashcards]

QUOTES = '"“”„'

def strip_quotes(text):
    """Remove quotation marks the model sometimes wraps around a whole card."""
    if len(text) >= 2 and text[0] in QUOTES and text[-1] in QUOTES:
        return text[1:-1].strip()
    return text

def answer_matches_concept(answer, concept):
    """True if the answer is the concept itself, allowing for inflected endings.

    Each word must share most of its beginning with the concept word,
    so "alkurikoksen" matches "Alkurikos" but "kauppaan" does not match "Huumekauppa".
    """
    answer_words = re.sub(r"[^\w\s-]", "", answer.lower()).split()
    concept_words = concept.lower().split()
    if len(answer_words) != len(concept_words):
        return False

    for a, c in zip(answer_words, concept_words):
        shared = len(os.path.commonprefix([a, c]))
        needed = len(c) if len(c) < 4 else max(4, int(len(c) * 0.7))
        if shared < needed:
            return False
    return True

FILLER_WORDS = {"the", "and", "for", "but", "not", "are", "was", "its", "has", "ja", "tai", "ei"}

def answer_leaks(sentence, answer):
    """True if the answer, or a significant word from it, appears in the sentence."""
    words_in_sentence = sentence.lower().split()
    for word in re.sub(r"[^\w\s-]", "", answer.lower()).split():
        # Three-letter words like "tax" count too; only filler words are skipped
        if len(word) < 3 or word in FILLER_WORDS:
            continue
        stem = word[:4]
        for other in words_in_sentence:
            if other.startswith(stem):
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
- The answer is a single common word that could be guessed without reading the source
- The sentence defines the concept instead of using it in context, for example a
  sentence built around "means", "refers to" or "is called", in any language
- The answer is one of several examples listed together, so another item from
  the same list would fill the blank equally well
- Other terms from the same field would fill the blank equally well, because the
  sentence only gives a statistic, date, estimate or general remark instead of
  something characteristic of this one concept
- The sentence starts with a word such as "He", "She", "It", "This", "Additionally"
  or "However" that only makes sense after the previous sentence
- The sentence is ungrammatical, for example with a word missing or in the wrong
  form. Cards are copied from the source word for word, so its errors stay in"""
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
        model=JUDGE_MODEL,
        reasoning_effort=JUDGE_EFFORT,
        messages=[{"role": "user", "content": prompt}]
    )

    keep = []
    for part in response.choices[0].message.content.split(','):
        part = part.strip().rstrip('.')
        if part.isdigit():
            index = int(part) - 1
            if 0 <= index < len(cards):
                keep.append(cards[index])

    for q, a in cards:
        if (q, a) not in keep:
            debug("Review", f"DROPPED by model: {q} → {a}")

    if card_type == "Cloze cards":
        for q, a in keep:
            if answer_leaks(q, a):
                debug("Review", f"DROPPED (answer leaks into sentence): {q} → {a}")
        keep = [(q, a) for q, a in keep if not answer_leaks(q, a)]

    return keep, len(cards) - len(keep)

def generate_reviewed_cards(content, api_key, num_cards, difficulty, card_type="Q/A cards", max_rounds=3):
    """Generate and review cards, topping up with new ones until num_cards pass review.

    Returns (cards, rejected_count).
    """
    cards = []
    tried = []
    tried_concepts = []
    rejected = 0

    for _ in range(max_rounds):
        missing = num_cards - len(cards)
        if missing <= 0:
            break

        new = generate_cards(content, api_key, missing, difficulty, card_type, exclude=tried, tried_concepts=tried_concepts)
        new = [card for card in new if card not in tried]
        # An empty round can still be followed by a better one, since cloze concepts that
        # failed are excluded next time; max_rounds keeps this from running forever
        if not new:
            continue
        tried += new

        kept, dropped = review_cards(new, api_key, card_type)
        rejected += dropped

        # Two cloze cards for the same answer test the same thing, so keep the first.
        # Q/A cards may share an answer, since they can ask different things about it.
        for q, a in kept:
            if "[...]" in q and any("[...]" in q2 and answer_matches_concept(a, a2) for q2, a2 in cards):
                debug("Review", f"DROPPED (duplicate answer): {q} → {a}")
                rejected += 1
                continue
            cards.append((q, a))

    return cards[:num_cards], rejected


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