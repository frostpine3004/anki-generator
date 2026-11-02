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
| Extraction | `scrape_url`, `extract_pdf`, `fetch_transcript`, `transcribe_episode` | Plain text |
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

### State shared between users

Streamlit runs every user's session in the same process, so anything stored at
module level or on disk is shared. Two things were:

- The debug log was one module-level list. People generating at the same time
  saw each other's steps.
- The deck was saved as `<deck name>.apkg` in the app folder. Two users who kept
  the default name could download each other's cards.

The log is now a `ContextVar`, which holds a separate value for each session's
thread, and `new_debug_log()` starts a fresh one for every run. The deck is built
in memory and handed straight to the download button.

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
concepts from the source. `generate_cloze_from_concepts` then builds one card
per concept, with the concept as the blank. The blank is decided before the
sentence is chosen, and the rule against common words now applies during
extraction where it can be enforced.

Cost roughly doubled for that card type. That seemed worth it.

### Copying sentences instead of writing them

Two stages fixed where the blank went, but the model still wrote the sentence.
It reworded the source and stitched parts of different sentences together, so a
card could state something the source never said.

Now the model doesn't write sentences at all. For each concept the code finds
the source sentences that contain it, and `usable_sentence` drops the ones that
can't work as a card: reference entries, headings, statistics ("US$14 billion in
[...]" fits almost any financial crime), definitions ("known as …", "called
the …"), references to an earlier part of a podcast ("the [...] that Daryl Peel
was talking about earlier", "after the break") and sentences that use a symbol
defined earlier ("span V"). The model only chooses
the best candidate and may cut a shorter part out of a long one.

`cut_card` checks the cut word for word against the source and rejects cuts
that split a phrase or stop mid-clause. The blank is made in code, so the answer
is always the concept in the form the source uses.

### Sentences that point back

A copied sentence can depend on the one before it: "That fermentation generally
happens right on the farms", or "More European countries planted [...]". These
were rejected at first, which threw away good material. Now `points_back`
detects them, and the card shows the sentence before it too, copied whole:

> Fresh from the pod, the seeds are light in color and don't have a lot of
> flavor. To get the chocolate we know today, they have to be [...].

That sentence must make sense on its own and must not contain the answer, or the
card is skipped. A leading "But", "However" or "Then" is simply left out of a
card, since the rest stands on its own.

### A stronger model

Choosing a sentence and reviewing cards take judgment that `gpt-4o-mini` lacked.
It chose a sentence the prompt gives as a bad example, and the review kept it.
These two steps moved to `gpt-5-mini` at low reasoning effort first. At the
default effort a run took minutes and the app looked hung.

Concept extraction and Q/A cards stayed on `gpt-4o-mini` for a while, and its
Q/A answers were often clumsy fragments ("Exponent needed for base"). Now every
text step uses `gpt-5-mini`, set in one place in `flashcards.py`. It costs a
little more per run.

At first every step thought at low effort, and a run on ten PDF pages took four
minutes. Listing concepts and writing Q/A cards read the whole source text, so
thinking there is slow, and those steps need knowledge more than judgment. They
now think as little as possible ("minimal"), and only choosing and reviewing
think at low effort. The debug log shows how long each step took.

A run still takes a minute or more, and a spinner alone made it look stuck. The
app now shows a progress bar with the name of the running step, which
`generate_reviewed_cards` reports through a `progress` callback like the one
`transcribe_episode` uses. The run happens in a second thread so the page can
keep the bar moving. How many rounds a run needs isn't known in advance, so the
bar moves by time: fast at first, then slower, and it fills when the run ends.
A seconds counter was tried first and wasn't useful.

### Both card types

"Both" used to ask for Q/A and cloze cards in one call, so its cloze cards were
written by the model and skipped every check above. Now it runs the two
pipelines one after the other. Half the cards go through the cloze pipeline,
and Q/A cards fill the rest, so a source with few good sentences still gives
the number of cards asked for. The Q/A step gets the cloze cards as cards not
to repeat, so the two don't test the same fact.

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

The first fix compared the first four characters. Four was a guess, and it
turned out too loose: "Monero" matched "money", and a good card was thrown out.

`shares_stem` now requires most of the shorter word to match, at least five
characters or 80 percent of it:

```python
def shares_stem(a, b):
    shortest = min(len(a), len(b))
    if shortest < 4:
        return False
    needed = min(shortest, max(5, int(shortest * 0.8)))
    return len(os.path.commonprefix([a, b])) >= needed
```

It still works on Finnish, where the stem survives inflection and the ending
changes, which was the case I cared about: "rahanpesudirektiivi" shares a stem
with "rahanpesun". A three-letter answer such as "tax" must match the start of a
word. An abbreviation gives the answer away too, as in "[...] (NFTs)" for
"non-fungible tokens", so that is checked as well.

### Two cards from one sentence

Two cloze cards cut from the same sentence give each other away, since each
shows the other's answer: "Al Capone on [...]" and "[...] on tax evasion". The
second one is dropped.

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

They now live in text files under `prompts/`, loaded at call time:

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

`fetch_episodes` reads the feed and returns episodes with audio URLs, durations
and transcript links. `transcribe_episode` downloads the file, trims it if a
range was selected, splits it if needed, and sends each part to the API.

### Published transcripts first

Picking an episode checks for a published transcript before anything else,
since it is free. If there is one, the app uses it and shows no Whisper button.
If not, one button transcribes the episode, with the cost on it. An earlier
version only checked the latest episode, behind a button of its own, and had a
box for pasting a transcript by hand.

Feeds announce transcripts with a `podcast:transcript` tag, one per format.
`feedparser` keeps only one of them per episode, so `feed_transcripts` reads the
feed's XML directly, and `pick_transcript` takes the format that is easiest to
clean: HTML, plain text, VTT, SRT, then JSON.

Subtitle formats put a timestamp before every line, and the first test with a
Podnews transcript had one in every sentence. `transcript_to_text` removes
timestamps, line numbers and speaker tags such as `<v Sam>`, and joins the text
segments of a JSON transcript.

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

The button shows an estimate before the call, and the user can pick a start
point and duration:

```python
span = limit or max(0, minutes - start_min)
cost = span * flashcards.WHISPER_COST_PER_MINUTE
label = f"Transcribe {span:.0f} minutes with Whisper (about ${cost:.2f})"
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

### Character encoding

Plain-text transcripts often come without a charset. `requests` then decodes
them as Latin-1, and "mélangeur" in a chocolate episode came out garbled. VTT and
JSON must be UTF-8 by their specifications, and SRT and plain-text transcripts
are in practice, so `fetch_transcript` decodes them as UTF-8.

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

Removed from use, and later from the code. Cloze cards are now copied from the
source, so the word in the blank already has the form the sentence needs.

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
Wikipedia article and unusable ones from a literature review. Podcasts show the
same: a daily news show gave cards about who did what this week, while a
scripted explainer (BrainStuff's episode on chocolate) gave five good cards out
of five. When output is bad, check the source before the prompt.

**Speaker labels in transcripts.** Some shows publish plain-text transcripts
with a label before every turn, as in "00:24:37 Speaker 2:". These are not
removed yet.

**List items in PDFs.** A list item without a verb ("Online job marketplaces
such as Fiverr, which accept funds...") can pass as a sentence. The PDF text has
no list markers, and the model ignores the rule against it.

**Conflicting prompt rules.** One rule requires taking the sentence from the
source; another prohibits dictionary-style definitions. In most sources both
hold. In a theory section where concepts are defined, every source sentence is a
definition and nothing satisfies both. Unresolved.

**PDF page numbering.** Page selection counts physical pages from the start of
the file, which differs from printed numbers when a document has front matter.
The interface says so rather than trying to detect it.

**No validation after review.** Cards that pass review are exported as-is.
Nothing checks that the `.apkg` imports correctly or that the deck contains what
was asked for.

**Temporary files.** genanki leaves its temporary database in the server's temp
folder for every deck it builds. Each has a unique name and is never served to
anyone, but they pile up.
