import streamlit as st
import flashcards

# Page config
st.set_page_config(
    page_title="AI Flashcard Generator",
    layout="centered"
)

st.title("AI Flashcard Generator")
st.subheader("Turn any webpage into Anki flashcards")

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
            manual = st.session_state.get("podcast_manual", "")
            content = manual or st.session_state.get("podcast_content", "")
            if not content:
                st.error("No transcript loaded. Click 'Load latest episode' or paste one manually.")
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

    with st.spinner("Generating flashcards with AI..."):
        cards = flashcards.generate_cards(content, api_key, num_cards, difficulty, card_type)

    with st.spinner("Checking card quality..."):
        cards, rejected = flashcards.review_cards(cards, api_key, card_type)
        cards = cards[:num_cards]
              
    if rejected:
        st.caption(f"{rejected} cards rejected in review")

    if not cards:
        st.error("No flashcards could be generated. Try a different source.")
        st.stop()

    st.success(f"Generated {len(cards)} flashcards!")

    for i, (question, answer) in enumerate(cards, 1):
        with st.expander(f"Flashcard {i}: {question[:50]}..."):
            st.write(f"**Question:** {question}")
            st.write(f"**Answer:** {answer}")

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
**A summary is something you read. A flashcard is something you answer.**

Karpicke and Roediger ran a study in Science in 2008 comparing the two. Students
who tested themselves on material remembered far more of it weeks later than
students who reread it. Rereading felt more productive at the time, which is why
people keep doing it. A 2006 review by Cepeda and colleagues found the same
pattern for spacing: review spread over several days beat one long session.
Anki's scheduler is built around that finding.

The problem with AI summaries is not that they are bad summaries. It's that
reading is not retrieval.

That's what this tool is for.

**Direct Anki export**
Produces a .apkg file that imports straight into your existing decks. NotebookLM
and similar tools won't do this. Your scheduler and settings stay as they are;
card writing is the only step this takes over.

**Re-runnable as you improve**
Regenerate the same source at greater depth once the early cards feel obvious.
Generation is cheap enough that a deck is never a one-time commitment.

**Nothing is stored**
Your API key and whatever you paste in are used once and discarded.

**Open source**
All the code is on GitHub if you want to check any of that.
    """)