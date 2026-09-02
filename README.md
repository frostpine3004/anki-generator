# AI Flashcard Generator

An AI-powered tool that automatically generates Anki flashcards from multiple content sources.

## Features
- Webpage/Article URL — scrape any public webpage
- YouTube Transcript — paste transcript to generate cards
- Podcast RSS Feed — fetch published transcripts, or transcribe audio with Whisper
- PDF Upload — upload any PDF document
- My Own Notes — paste any text

## How it works
1. Enter your OpenAI API key
2. Choose your input source
3. Set difficulty level and number of cards
4. Generate and download your Anki deck (.apkg)

## YouTube Transcript — Design Decision
YouTube transcript scraping was intentionally left as a manual paste instead of automated scraping. YouTube's Terms of Service prohibit automated access to transcripts without explicit permission. To respect these terms, users copy the transcript directly from YouTube's own transcript feature and paste it into the app. This keeps the tool legally safe and puts content responsibility with the user.

To get a YouTube transcript: Open video → click '...' → Show transcript → copy all text.

## Setup
```bash
git clone https://github.com/frostpine3004/anki-generator.git
cd anki-generator
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
streamlit run app.py
```

## Requirements
- Python 3.7+
- OpenAI API key
- Anki (to import generated decks)

## Built on
Based on marcbln/ai-flashcard-generator with significant upgrades.

## License
MIT

Built as a course project, 2026. 
