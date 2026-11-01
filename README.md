# AI Flashcard Generator

Final project for the Building AI course

## Summary

AI Flashcard Generator turns webpages, PDFs, podcasts, YouTube transcripts and your own notes into Anki flashcards. It writes Q/A and cloze cards with an OpenAI model, has the model review its own cards and drop the weak ones, and exports a ready-to-import `.apkg` deck.

## Background

Testing yourself is a better way to learn than rereading. Karpicke and Roediger showed this in *Science* in 2008, and Cepeda and colleagues showed in 2006 that spreading reviews over several days beats one long session. Anki is built on both findings, but writing good cards by hand is slow, so many people never get past rereading.

Problems this project addresses:
* Writing flashcards by hand takes longer than reading the material
* Generic AI summaries are something you read, not something you answer
* Existing card generators often charge a subscription whether you use them or not

My own motivation: I use Anki to study technical material and wanted a cheap way to turn articles, papers and podcast episodes into cards.

## How is it used?

Run the Streamlit app, enter your own OpenAI API key and choose a source:

* **Webpage/Article URL**: the page is fetched and its paragraphs are extracted
* **YouTube Transcript**: paste the transcript from YouTube's own "Show transcript" view
* **Podcast RSS Feed**: check the feed for a published transcript of the latest episode, or pick an episode and transcribe it with Whisper. You can choose a time range, and the cost is shown before you start.
* **PDF Upload**: choose a page range (e.g. `1, 5-8, 27-30`) and preview the extracted text
* **My Own Notes**: paste any text

Then choose the number of cards, the difficulty, the card type (Q/A, cloze or both) and, optionally, a topic to narrow the source down to. The app shows the generated cards and offers the deck as a download. Import the `.apkg` file into Anki.

The tool is meant for students and self-learners who already use Anki. It costs only what the OpenAI API charges: around a cent for cards from an article, and about $0.006 per minute of transcribed audio.

### YouTube design decision

YouTube transcripts are pasted by hand instead of scraped. YouTube's Terms of Service prohibit automated access to transcripts without permission, so users copy the transcript from YouTube's own transcript feature. To get one, open the video, click '...', choose Show transcript and copy all the text.

### Setup

```bash
git clone https://github.com/frostpine3004/anki-generator.git
cd anki-generator
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
streamlit run app.py
```

Requirements: Python 3.10+, an OpenAI API key, Anki, and [ffmpeg](https://ffmpeg.org/) for trimming and splitting podcast audio. On Streamlit Cloud, ffmpeg is installed from `packages.txt`.

## Data sources and AI methods

Data comes from whatever the user provides: public webpages (via `requests` and BeautifulSoup), PDFs (PyPDF2), podcast RSS feeds (feedparser) and pasted text. Nothing is stored.

AI methods:
* **Card generation** with `gpt-4o-mini`. For long sources, evenly spaced chunks are sampled across the whole text instead of only its beginning. Choosing cloze sentences and reviewing cards use `gpt-5-mini` at low reasoning effort, since `gpt-4o-mini` often ignored the rules in these judgment steps.
* **Concept extraction for cloze cards**: the model first lists the key concepts, and the main topic of the text is dropped since it appears in nearly every sentence. The app then finds the source sentences that contain each concept, skipping reference entries, glossary lines, statistics ("US$14 billion"), definitions ("known as …") and sentences that point back to an earlier one ("these sites") or use a symbol defined earlier ("span V"). Words that PDF extraction split in two ("anonymit y") are joined back first. The model picks the sentence that points to that one concept best, and may cut a continuous part out of a long one. The app checks the cut word for word against the source, rejects cuts that split a phrase or stop mid-clause, and blanks the concept out. Every cloze card is a real sentence from the source, and the answer is the concept in the exact form the source uses.
* **Top-up rounds**: if too few cards pass, up to three rounds ask for new concepts, leaving out every concept already tried.
* **Self-review**: a second model call rejects vague, guessable or bibliographic cards, and cloze cards whose blank other terms from the same field would fill equally well. A rule-based check also drops cloze cards whose answer leaks into the sentence.
* **Debug view**: the "Show debug info" checkbox lists what happened to each concept and card at every step.
* **Speech-to-text** with OpenAI Whisper (`whisper-1`). Audio over 25 MB is split into chunks with ffmpeg.
* **Topic filter**: a simple stem-matching filter keeps only the paragraphs that mention your topic, plus optional surrounding context.

## Challenges

* Card quality: models easily produce cards that can be answered from sentence structure alone, or cloze cards whose answer appears elsewhere in the sentence. The review step reduces this but doesn't eliminate it.
* Grounding: when the model wrote cloze sentences itself, it reworded the source and sometimes stitched two sentences into a claim the source never made. Cloze cards are now copied from the source, so the model can only choose and cut, not write.
* PDF text extraction splits words ("cryptocurrencie s"), drops spaces after full stops and leaves citation markers, which all have to be cleaned before sentences can be matched.
* Strict filters mean a source with few concrete sentences can yield fewer cards than requested.
* Only around 30–40% of podcasts publish transcripts, and Whisper transcription costs money, so the time range and cost estimate matter.
* PDF page numbers in the file don't always match the printed page numbers, and scanned PDFs have no extractable text.
* Content is sent to OpenAI's API, so the tool isn't suited for confidential material.
* The topic filter matches word stems, not meaning, so it can miss synonyms.

## What next?

* Interview prep mode with "why" and "compare" questions answered in short key points
* Cards from terminal man pages
* Automatic topic tags and duplicate detection against existing decks
* AnkiConnect integration to write cards straight into Anki

## Acknowledgments

* Based on [marcbln/ai-flashcard-generator](https://github.com/marcbln/ai-flashcard-generator), with significant upgrades
* Building AI course by Reaktor and the University of Helsinki
* Karpicke, J. D. & Roediger, H. L. (2008). The critical importance of retrieval for learning. *Science*, 319(5865)
* Cepeda, N. J. et al. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin*, 132(3)
* Libraries: Streamlit, OpenAI Python SDK, genanki, BeautifulSoup, PyPDF2 and feedparser

## License

MIT
