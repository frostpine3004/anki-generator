# Design decisions

Why the tool is built the way it is. For what it does and how to run it, see
the [README](README.md).

This project is a course work for the Building AI course / University of Helsinki.

---

## Contents

1. [Architecture](#1-architecture)
2. [Two-stage cloze generation](#2-two-stage-cloze-generation)
3. [Card-type-aware review](#3-card-type-aware-review)
4. [Sampling instead of truncation](#4-sampling-instead-of-truncation)
5. [Prompts as files](#5-prompts-as-files)
6. [Audio transcription and cost control](#6-audio-transcription-and-cost-control)
7. [Fetching content](#7-fetching-content)
8. [Grammar correction, and why it was removed](#8-grammar-correction-and-why-it-was-removed)
9. [Known limitations](#9-known-limitations)

---

## 1. Architecture

Two modules. `app.py` holds the Streamlit interface, `flashcards.py` holds
everything else. Only `app.py` imports Streamlit.

The pipeline runs in four stages:

| Stage | Function | Output |
|---|---|---|
| Extraction | `scrape_url`, `extract_pdf`, `fetch_podcast_transcript`, `transcribe_episode` | Plain text |
| Sampling | `sample_content` | Text within the context budget |
| Generation | `generate_cards`, or `extract_concepts` + `generate_cloze_from_concepts` | List of (front, back) tuples |
| Review and export | `review_cards`, `build_deck` | `.apkg` file |

Every extraction function returns plain text, so the generation stage does not
know where the content came from. Adding a source type means one extraction
function and one branch in the interface.

The split between the two modules was not planned. It happened because the
transcription function needed a progress bar, and putting Streamlit calls
inside it felt wrong. The callback approach came out of that:

```python
def transcribe_episode(audio_url, api_key, progress=None, start_min=0, limit_min=0):
    for i, part in enumerate(parts):
        if progress:
            progress(i, len(parts))
```

The interface decides what the callback does. The same function would work from
a command line with a callback that prints.

---

## 2. Two-stage cloze generation

Cloze quality did not improve across five prompt versions. Each fix introduced a
new failure mode, which should have been the signal to stop editing the prompt
sooner than it was.

The model wrote the sentence first and picked the word to hide second. The blank
landed on whatever the finished sentence offered: function words, fragments of
multi-word terms, words inferable from grammar. Cards looked correct and tested
nothing.

Prohibitions could not fix this. A rule against hiding common words has no
effect when the finished sentence contains no better candidate. The problem
happened before the selection step.

The fix was two API calls instead of one. `extract_concepts` returns a list of
concepts from the source. `generate_cloze_from_concepts` takes that list plus
the source text and finds a sentence for each, hiding the concept. The blank is
decided before the sentence exists, and the rule against common words now
applies during extraction where it can be enforced.

Cost roughly doubled for that card type. That seemed worth it.

---

## 3. Card-type-aware review

Every card was rejected. Ten out of ten, every run, with no error anywhere.

I added a counter showing how many cards the review pass rejected. That
separated two possibilities which look identical from the outside: cards being
generated and discarded, or never generated at all. The number said ten, so
generation was fine.

`answer_leaks` rejects a card when a word from the answer appears in the
question. For cloze cards that check is required — a hidden term visible in the
sentence makes the card useless. For question-and-answer cards the same word
appears in both by nature. "Inflation means a decline in purchasing power" is a
correct answer to "What does inflation mean?" and the check threw it out.

One criterion, two opposite requirements. `review_cards` now takes a
`card_type` argument and applies different criteria per type. `answer_leaks`
runs only for cloze cards.

Rejection settled between 40 and 60 percent depending on card type and source,
which is why generation requests double the number of cards the user asked for.

### Stem matching

Exact word comparison missed leaks across inflection. A sentence containing
"diet" with "dietary" as the answer passed the check.

The current version compares the first four characters:

```python
stem = word[:4]
for other in words_in_sentence:
    if other.startswith(stem):
        return True
```

Four is a guess. Three produced false positives on short English words, and I
did not test five or six. It works well enough on Finnish, where the stem
survives inflection and the ending changes, which was the case I cared about.

---

## 4. Sampling instead of truncation

Long sources produced cards drawn only from the opening. A podcast episode
yielded cards about the host's job title and how many episodes the show had run,
because the model never saw past the introduction.

`content[:20000]` was the cause. An hour of audio transcribes to roughly 54 000
characters, so two thirds were discarded.

`sample_content` now takes six evenly spaced chunks across the whole text:

```python
def sample_content(content, limit=20000):
    if len(content) <= limit:
        return content
    chunks = 6
    size = limit // chunks
    step = len(content) // chunks
    parts = [content[i * step:i * step + size] for i in range(chunks)]
    return "\n[...]\n".join(parts)
```

Six was arbitrary. Text between chunks is lost and boundaries cut mid-sentence,
which has not caused a visible problem but probably should be measured.

For PDFs the page range selector is the better tool anyway, since the user knows
which sections matter.

---

## 5. Prompts as files

Prompts started out embedded in the code. Editing a rule meant editing a string
inside a function, and the change disappeared into the commit diff among
unrelated lines.

They now live in five text files under `prompts/`, loaded at call time:

```python
def load_prompt(name):
    path = os.path.join(os.path.dirname(__file__), "prompts", f"{name}.txt")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()
```

A prompt is configuration, not logic.

The migration was left half done and I did not notice for some time. The cloze
branch returns early, before `load_prompt` is reached, so `cloze.txt` was never
read in cloze mode. I edited that file repeatedly and wondered why nothing
changed. Nothing errored, because the code worked correctly using a different
set of rules.

---

## 6. Audio transcription and cost control

Roughly 60 to 70 percent of podcasts publish no transcript, which made the
RSS-only version useless on most feeds. Whisper fills the gap.

`fetch_episodes` reads the feed and returns episodes with audio URLs and
durations. `transcribe_episode` downloads the file, trims it if a range was
selected, splits it if needed, and sends each part to the API.

### The 25 MB limit

The endpoint accepts files up to 25 MB, roughly 25 minutes of audio. Longer
episodes get split with ffmpeg using stream copy, which cuts without
re-encoding:

```python
subprocess.run(
    ["ffmpeg", "-i", path, "-f", "segment",
     "-segment_time", str(minutes * 60), "-c", "copy", pattern],
    check=True, capture_output=True,
)
```

Parts are sorted by filename before transcription, so the zero-padded numbering
matters: `part000.mp3` sorts before `part010.mp3`, `part0.mp3` would not.

Installing ffmpeg pulled in fourteen dependencies and a few hundred megabytes,
almost all of it video codecs this project never touches. There is no lighter
path that I found.

### Cost

Card generation costs fractions of a cent. Transcription costs 0.006 USD per
minute, about a hundred times more. A week of active development and testing
came to under ten cents in total, but that was mostly text.

The interface shows an estimate before the call and lets the user pick a start
point and duration:

```python
span = limit or max(0, minutes - start_min)
cost = span * flashcards.WHISPER_COST_PER_MINUTE
st.caption(f"{span:.0f} minutes — costs about ${cost:.2f}")
```

Trimming happens before splitting, so transcribing ten minutes of an hour-long
episode costs ten minutes.

ffmpeg is a system dependency, declared in `packages.txt` for deployment. When
it is missing, the error names the tool rather than surfacing a file-not-found.

---

## 7. Fetching content

### Identifying the client

Wikipedia and a podcast host both returned 403. The User-Agent header was the
cause.

The first workaround used a browser string, which is the standard way around
this and also the practice the policy exists to discourage. Replaced with one
that says what the client actually is:

```python
HEADERS = {
    'User-Agent': 'AnkiFlashcardGenerator/1.0 (https://github.com/frostpine3004/anki-generator)'
}
```

Wikimedia asks for a descriptive User-Agent with contact information. The issue
is server load, not copyright — Wikipedia content is openly licensed.

### Certificates

Feed parsing failed with `SSL: CERTIFICATE_VERIFY_FAILED` on macOS.
`feedparser` opens its own connection through `urllib`, which uses system
certificates. `requests` bundles its own, which is why scraping worked and feed
parsing did not.

Running the certificate installer fixed it locally. The code fix routes the
fetch through `requests` and hands the bytes to the parser:

```python
resp = requests.get(rss_url, headers=HEADERS, timeout=10)
feed = feedparser.parse(resp.content)
```

That one works on other people's machines too.

### YouTube

Automated retrieval would breach YouTube's terms. The user pastes the transcript
manually. One extra step, and the decision about what to fetch stays with them.

---

## 8. Grammar correction, and why it was removed

In Finnish a cloze answer has to appear in the case the sentence requires.
Nominative "hyväntahtoisuus" does not fit a slot that wants partitive
"hyväntahtoisuutta". I built a correction pass to handle this automatically.

First version returned the corrected sentence. The model removed the blank
marker and gave back a filled sentence.

Second version kept the instruction to preserve the marker. The model preserved
it and put the answer next to it.

Third version returned only the corrected word, leaving the sentence untouched
in code so the model could not break it. Structurally this was right, and the
debug output showed the model does know Finnish morphology. It changed correct
inflections to incorrect ones in two cases out of three.

Removed from use. The function is still in the file, uncalled.

Two wrong corrections out of three is worse than none, and the user cannot see
the error until the wrong form has been memorised.

Only `gpt-4o-mini` was tested. A larger model might parse structure well enough,
but that is a guess and would cost roughly fifteen times more per run.
Separating a model-size problem from a task-structure problem would take its own
test, which I have not run.

Card editing before export is the more likely fix. The model missed these
errors; a reader catches them at a glance.

---

## 9. Known limitations

**Non-determinism.** The same prompt gives different output each run. A rule
that works on three cards may fail on the fourth. Judging a change needs a
metric set in advance and enough repetitions to separate effect from variance.

**Source quality dominates.** The same code produced usable cards from a
Wikipedia article and unusable ones from a literature review. When output is
bad, check the source before the prompt.

**Conflicting prompt rules.** One rule requires taking the sentence from the
source; another prohibits dictionary-style definitions. In most sources both
hold. In a theory section where concepts are defined, every source sentence is a
definition and nothing satisfies both. Unresolved.

**"Both" mode is inconsistent.** It uses single-call generation while pure cloze
mode uses the two-stage pipeline, so its cloze cards are weaker. `review_cards`
also receives "Both" as the card type, which matches neither branch cleanly, so
`answer_leaks` never runs for it. Untested end to end. Either it should run both
pipelines and merge, or it should be removed.

**PDF page numbering.** Page selection counts physical pages from the start of
the file, which differs from printed numbers when a document has front matter.
The interface says so rather than trying to detect it.

**No validation after review.** Cards that pass review are exported as-is.
Nothing checks that the `.apkg` imports correctly or that the deck contains what
was asked for.
