import streamlit as st
import flashcards

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

    if url and st.button("Check for a published transcript (latest episode)"):
        with st.spinner("Looking for a transcript..."):
            text, info = flashcards.fetch_podcast_transcript(url)
            if text:
                st.session_state["podcast_content"] = text
                st.session_state["podcast_info"] = info
                st.success(f"Transcript found: {info}")
            else:
                st.warning(info)

    if url and st.button("Load episodes"):
        with st.spinner("Fetching RSS feed..."):
            episodes, info = flashcards.fetch_episodes(url)
            st.session_state["episodes"] = episodes
            if episodes:
                st.success(info)
            else:
                st.warning(info)

    episodes = st.session_state.get("episodes", [])
    if episodes:
        titles = [e["title"] for e in episodes]
        choice = st.selectbox("Choose an episode", titles)
        episode = episodes[titles.index(choice)]

        minutes = episode["duration"] / 60

        col1, col2 = st.columns(2)
        start_min = col1.number_input("Start at minute", min_value=0, value=0, step=5)
        limit = col2.number_input("Minutes to transcribe (0 = to end)", min_value=0, value=0, step=5)

        span = limit or minutes
        if span:
            cost = span * flashcards.WHISPER_COST_PER_MINUTE
            st.caption(f"{span:.0f} minutes — costs about ${cost:.2f}")
        else:
            st.caption("Duration unknown — cost cannot be estimated in advance")

        if st.button("Transcribe with Whisper"):
            est = max(1, round(span / 7)) if span else 2
            bar = st.progress(0.0, text=f"Downloading audio — about {est} min total")

            def on_progress(i, total):
                bar.progress((i + 1) / total, text=f"Transcribing part {i + 1} of {total}...")

            try:
                text = flashcards.transcribe_episode(
                    episode["audio_url"], api_key,
                    progress=on_progress,
                    start_min=start_min,
                    limit_min=limit,
                )
                bar.empty()
                st.session_state["podcast_content"] = text
                st.session_state["podcast_info"] = f"{episode['title']} (Whisper transcript)"
                st.success(f"Transcribed {len(text)} characters")
            except Exception as e:
                bar.empty()
                st.error(f"Transcription failed: {e}")

    if st.session_state.get("podcast_content"):
        info = st.session_state.get("podcast_info", "")
        st.caption(f"{info} — {len(st.session_state['podcast_content'])} characters")

    st.text_area(
        "Or paste transcript manually",
        height=150,
        key="podcast_manual",
        placeholder="Paste transcript here if you'd rather not transcribe"
    )

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

show_debug = st.checkbox("Show debug info")

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
            manual = st.session_state.get("podcast_manual", "")
            content = manual or st.session_state.get("podcast_content", "")
            if not content:
                st.error("No transcript loaded. Check for a published transcript, transcribe an episode with Whisper, or paste one manually.")
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

    flashcards.debug_log.clear()
    with st.spinner("Generating and checking flashcards with AI..."):
        cards, rejected = flashcards.generate_reviewed_cards(content, api_key, num_cards, difficulty, card_type)

    if rejected:
        st.caption(f"{rejected} cards rejected in review")

    if show_debug:
        with st.expander("Debug: what happened at each step"):
            for step, text in flashcards.debug_log:
                st.text(f"[{step}] {text}")

    if not cards:
        st.error("No flashcards could be generated. Try a different source.")
        st.stop()

    st.success(f"Generated {len(cards)} flashcards!")

    for i, (question, answer) in enumerate(cards, 1):
        with st.expander(f"Flashcard {i}: {question[:50]}..."):
            # Escape $ so Streamlit doesn't render "$200 billion ... $6 billion" as a formula
            st.write("**Question:** " + question.replace("$", "\\$"))
            st.write("**Answer:** " + answer.replace("$", "\\$"))

    filename = flashcards.build_deck(cards, deck_name)
    with open(filename, "rb") as f:
        st.download_button(
            label="Download Anki Deck",
            data=f,
            file_name=filename,
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

Checks a podcast's RSS feed for a published transcript of the latest episode,
and transcribes any episode with Whisper when the show hasn't published one. Most tools in this space handle PDFs and web
pages; few handle audio.

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

All the code is on GitHub if you want to check any of that.
""")