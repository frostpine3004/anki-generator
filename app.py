import math
import time
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import flashcards

# Sets how fast the progress bar fills. A run that takes this long ends with the bar
# about two-thirds full; a typical cloze run (about 50 s) ends at about 80 %.
TYPICAL_SECONDS = {"Cloze cards": 30, "Q/A cards": 15, "Both": 45}


# Streamlit reruns the whole script on every click, so the feed and transcript
# are cached instead of being downloaded again each time.
# A failed feed raises, and Streamlit doesn't cache errors, so the next run retries.
@st.cache_data(ttl=600, show_spinner=False)
def load_episodes(feed_url):
    episodes, info = flashcards.fetch_episodes(feed_url)
    if not episodes:
        raise ValueError(info)
    return episodes


@st.cache_data(ttl=600, show_spinner=False)
def load_transcript(transcript_url, transcript_type):
    return flashcards.fetch_transcript(transcript_url, transcript_type)

# Page config
st.set_page_config(
    page_title="AI Flashcard Generator",
    layout="centered"
)

st.title("AI Flashcard Generator")
st.subheader("Turn webpages, PDFs, podcasts and notes into Anki flashcards")

st.info("Your API key is never stored. It's only used for your current session.")

api_key = st.text_input(
    "Enter your OpenAI API key",
    type="password",
    placeholder="sk-proj-..."
)

input_type = st.selectbox(
    "Choose input type",
    ["Webpage/Article URL", "YouTube Transcript", "Podcast RSS Feed", "PDF Upload", "My Own Notes"]
)

url = None
youtube_transcript = None
own_notes = None
pdf_file = None
page_range = None
podcast_content = None

if input_type == "Webpage/Article URL":
    url = st.text_input("Enter webpage URL", placeholder="https://en.wikipedia.org/wiki/...")
    st.caption("Paste any public webpage or article URL")

elif input_type == "YouTube Transcript":
    st.info("How to get transcript: Open YouTube video → click '...' → Show transcript → copy all text")
    youtube_transcript = st.text_area("Paste YouTube transcript here", height=200)

elif input_type == "Podcast RSS Feed":
    url = st.text_input(
        "Enter podcast RSS feed URL",
        placeholder="https://feeds.example.com/podcast.rss"
    )
    st.caption("Find RSS feed by searching 'podcast name + RSS feed'")

    episodes = []
    if url:
        with st.spinner("Fetching RSS feed..."):
            try:
                episodes = load_episodes(url)
            except ValueError as e:
                st.warning(str(e))

    if episodes:
        titles = [e["title"] for e in episodes]
        choice = st.selectbox("Choose an episode", titles)
        episode = episodes[titles.index(choice)]

        with st.spinner("Looking for a published transcript..."):
            published = load_transcript(episode["transcript_url"], episode["transcript_type"])

        if published:
            podcast_content = published
            st.success(f"Published transcript found — free to use ({len(published)} characters)")
        else:
            if episode["transcript_url"]:
                st.info("The feed lists a transcript for this episode, but it couldn't be read. Whisper can transcribe the audio instead.")
            else:
                st.info("No published transcript for this episode. Whisper can transcribe the audio.")

            minutes = episode["duration"] / 60

            col1, col2 = st.columns(2)
            start_min = col1.number_input("Start at minute", min_value=0, value=0, step=5)
            limit = col2.number_input("Minutes to transcribe (0 = to end)", min_value=0, value=0, step=5)

            # Without a length, Whisper transcribes from the start minute to the end
            span = limit or max(0, minutes - start_min)
            if span:
                cost = span * flashcards.WHISPER_COST_PER_MINUTE
                label = f"Transcribe {span:.0f} minutes with Whisper (about ${cost:.2f})"
            else:
                label = "Transcribe with Whisper"
                if not minutes:
                    st.caption("Duration unknown — cost cannot be estimated in advance")

            # Keep each transcript with the episode and range it came from,
            # so switching episodes never sends an old transcript to the card step.
            transcripts = st.session_state.setdefault("whisper_transcripts", {})
            key = (episode["audio_url"], start_min, limit)
            podcast_content = transcripts.get(key)

            if podcast_content:
                st.success(f"Transcribed {len(podcast_content)} characters")
            elif st.button(label):
                est = max(1, round(span / 7)) if span else 2
                bar = st.progress(0.0, text=f"Downloading audio — about {est} min total")

                def on_progress(i, total):
                    bar.progress((i + 1) / total, text=f"Transcribing part {i + 1} of {total}...")

                try:
                    transcripts[key] = flashcards.transcribe_episode(
                        episode["audio_url"], api_key,
                        progress=on_progress,
                        start_min=start_min,
                        limit_min=limit,
                    )
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                bar.empty()
                if key in transcripts:
                    # Redraw so the button is replaced by the "Transcribed" message
                    st.rerun()

elif input_type == "PDF Upload":
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
    if pdf_file:
        total_pages = flashcards.count_pdf_pages(pdf_file)
        page_range = st.text_input(
            f"Pages (1-{total_pages})",
            placeholder="e.g. 1, 5-8, 27-30"
        )
        st.caption("Counts from the file's first page, which may differ from printed page numbers. Leave empty for the whole document.")

        if st.button("Preview extracted text"):
            pages = flashcards.extract_pdf_pages(pdf_file, page_range)
            total_chars = sum(len(text) for _, text in pages)
            st.caption(f"{len(pages)} pages, {total_chars} characters")

            for page_num, text in pages:
                with st.expander(f"Page {page_num}"):
                    st.text(text)

elif input_type == "My Own Notes":
    own_notes = st.text_area("Paste your notes here", height=200)
    st.caption("Paste any text, notes or content you want to study")

num_cards = st.slider("Number of flashcards", min_value=3, max_value=20, value=5)

difficulty = st.selectbox("Difficulty level", ["Beginner", "Intermediate", "Advanced"])

card_type = st.selectbox(
    "Card type",
    ["Q/A cards", "Cloze cards", "Both"]
)

concept = st.text_input(
    "Narrow down to a topic (optional)",
    placeholder="mitochondria, cellular respiration"
)
st.caption("Separate alternatives with commas. Leave empty to use the whole source.")

context = st.slider("Paragraphs to include around each match", 0, 2, 1)

deck_name = st.text_input("Deck name", value="My AI Flashcards")

if st.button("Generate Flashcards"):
    if not api_key:
        st.error("Please enter your OpenAI API key!")
        st.stop()

    with st.spinner("Fetching content..."):
        if input_type == "Webpage/Article URL":
            if not url:
                st.error("Please enter a URL!")
                st.stop()
            content = flashcards.scrape_url(url)

        elif input_type == "YouTube Transcript":
            if not youtube_transcript:
                st.error("Please paste a YouTube transcript!")
                st.stop()
            content = youtube_transcript

        elif input_type == "Podcast RSS Feed":
            content = podcast_content
            if not content:
                st.error("No transcript yet. Pick an episode, and if it has no published transcript, transcribe it with Whisper first.")
                st.stop()

        elif input_type == "PDF Upload":
            if not pdf_file:
                st.error("Please upload a PDF file!")
                st.stop()
            content = flashcards.extract_pdf(pdf_file, page_range)

        elif input_type == "My Own Notes":
            if not own_notes:
                st.error("Please paste your notes!")
                st.stop()
            content = own_notes

    if concept:
        content, matched, total = flashcards.filter_by_concept(content, concept, context=context)
        if not matched:
            st.error("No paragraphs matched that topic. Try other words or leave the field empty.")
            st.stop()
        st.caption(f"Topic filter kept {matched} of {total} paragraphs")

    # A run takes a minute or more. It runs in a second thread so this one can keep
    # the progress bar moving and show which step is running.
    current = {"step": "starting"}

    def on_progress(step):
        # Called from the worker thread, which can't draw on the page, so only note the step
        current["step"] = step

    pool = ThreadPoolExecutor(max_workers=1)
    run = pool.submit(flashcards.generate_reviewed_cards,
                      content, api_key, num_cards, difficulty, card_type, progress=on_progress)
    # Lets the thread end when the run is done, without making this one wait for it
    pool.shutdown(wait=False)

    bar = st.progress(0.0)
    started = time.monotonic()
    while not run.done():
        # How many rounds a run needs isn't known in advance, so the bar moves by time:
        # quickly at first, then ever slower, and fills only when the run ends.
        share = 1 - math.exp(-(time.monotonic() - started) / TYPICAL_SECONDS[card_type])
        step = current["step"]
        bar.progress(share, text=f"{step[0].upper()}{step[1:]}...")
        time.sleep(0.3)
    bar.progress(1.0, text="Done")
    cards, rejected = run.result()
    bar.empty()

    if rejected:
        st.caption(f"{rejected} cards rejected in review")

    if not cards:
        st.error("No flashcards could be generated. Try a different source.")
        st.stop()

    st.success(f"Generated {len(cards)} flashcards!")

    for i, (question, answer) in enumerate(cards, 1):
        with st.expander(f"Flashcard {i}: {question[:50]}..."):
            # Escape $ so Streamlit doesn't render "$200 billion ... $6 billion" as a formula
            st.write("**Question:** " + question.replace("$", "\\$"))
            st.write("**Answer:** " + answer.replace("$", "\\$"))

    st.download_button(
        label="Download Anki Deck",
        data=flashcards.build_deck(cards, deck_name),
        file_name=f"{deck_name}.apkg",
        mime="application/octet-stream"
    )

st.divider()
st.subheader("Why flashcards instead of AI summaries?")

with st.expander("Read the case for this tool"):
    st.markdown("""
**An Anki deck generator that costs a few cents, not a subscription.**

Most card generators charge monthly or as a one-time purchase whether you use
them or not. This one asks for your own OpenAI key and adds nothing on top.
Cards from an article cost around a cent. Transcribing a ten-minute podcast
costs about six cents.

**You see the price before you press the button**

Transcription is the only step that costs real money, so the cost is shown up
front. You can also narrow the input: pick which pages of a PDF to use, or which
minutes of an episode to transcribe. Nothing runs up a bill in the background.

**Podcasts, not just documents**

Pick an episode from a podcast's RSS feed. If the show publishes a transcript,
it's used for free; if not, Whisper transcribes the audio. Most tools in this
space handle PDFs and web pages; few handle audio.

**No account, no server in the middle**

Nothing is stored here. Your key and whatever you paste in are used once and
discarded. Content does go to OpenAI's API, as it does with any tool like this.
It isn't used for training, and OpenAI keeps it 30 days for abuse monitoring.
No second company holds a copy.

**YouTube transcripts are pasted, not scraped**

Automated access would breach YouTube's terms, so this tool asks you to copy the
transcript yourself. One extra step, and the decision about what to fetch stays
with you.

**Direct Anki export**

Produces a .apkg file that imports into your existing decks. Your scheduler and
settings stay as they are; card writing is the only step this takes over.

**Why testing yourself beats rereading**

Karpicke and Roediger compared the two in Science in 2008. Students who tested
themselves remembered far more weeks later than students who reread the
material. Rereading felt more productive at the time, which is why people keep
doing it. Cepeda and colleagues found the same pattern for timing in 2006:
review spread across several days beat one long session. Anki's scheduler is
built on both findings. This tool writes the cards; Anki does the rest.

**Open source**

All the code is on [GitHub](https://github.com/frostpine3004/anki-generator) if you want to check any of that.
""")