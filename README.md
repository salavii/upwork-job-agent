# Upwork Job Matcher & Proposal Drafter

A semi-automatic pipeline that scores Upwork job postings against your
skills using a **local LLM**, ranks them, flags red flags (like a $10
budget for a multi-month build), and drafts honest first-draft proposals
for the jobs worth applying to — all running on your own machine, with
nothing auto-submitted and no scraping of Upwork.

You still click "save" on jobs you're looking at, and you still click
"submit" on Upwork yourself. This tool just does the reading, scoring,
and first-draft writing in between.

## What it does

1. A small Chrome extension grabs the title, metadata, and description of
   whatever Upwork job page you're currently looking at.
2. A local Python server saves it to a JSON file, skipping duplicates.
3. A local LLM (via [Ollama](https://ollama.com), running entirely on
   your machine) scores each job against your profile, breaks the job
   into its main skill components, and estimates what fraction of it you
   could realistically deliver alone.
4. For jobs that score well, it drafts a short, honest proposal — grounded
   only in skills/projects that actually exist in your profile.
5. Everything is rendered into a ranked, color-coded HTML report you can
   open in a browser, with a one-click delete button per job.

Nothing here auto-applies to jobs, auto-browses Upwork, or scrapes pages
you haven't opened yourself — see [ToS-safe by design](#tos-safe-by-design)
below for why that mattered.

## Architecture

```
┌─────────────────────┐   click "Save this job"   ┌─────────────────────┐
│  Chrome Extension    │ ─────────────────────────▶│      server.py       │
│  (content.js scrapes │   POST /save_job           │  localhost:8765      │
│   the current page,  │   { job_text }              │  - dedup check       │
│   popup.js sends it) │                            │  - writes jobs.json  │
└─────────────────────┘                            └──────────┬───────────┘
                                                                │
                                                                ▼
                                                        ┌───────────────┐
                                                        │   jobs.json    │  (your saved jobs)
                                                        └───────┬────────┘
                                                                │ read
                                                                ▼
                                                     ┌────────────────────┐        ┌───────────────────────┐
                                                     │    match_llm.py     │◀──────▶│        Ollama          │
                                                     │  score_job()         │        │  llama3.1:8b, running  │
                                                     │  draft_proposal()    │        │  locally on :11434     │
                                                     └──────────┬──────────┘        └───────────────────────┘
                                                                │ reads/writes
                                                                ▼
                                                        ┌───────────────┐
                                                        │ results.json   │  (score/verdict/proposal cache,
                                                        └───────┬────────┘   keyed by a hash of the job text)
                                                                │ writes
                                                                ▼
                                                        ┌────────────────┐
                                                        │  report.html    │  ranked, color-coded, with a
                                                        └───────┬────────┘   🗑 delete button per job
                                                                │
                                                    click delete → POST /delete_job → server.py removes
                                                    the job from jobs.json + results.json, then calls
                                                    match_llm.build_report() to rebuild report.html
                                                    (no re-scoring, no Ollama call)
```

There's also a standalone baseline, `match.py`, which scores jobs with
plain keyword matching and no LLM at all — kept around deliberately (see
below).

## Engineering decisions — and why

This project went through several real iterations, not a single design
pass. The decisions below are the ones that actually mattered.

**Started with keyword matching, hit its ceiling fast.**
`match.py` is a from-scratch keyword matcher: it looks for your skills as
whole words/phrases in the job text and computes a percentage match. It's
simple, fast, and fully deterministic — a good baseline. But it can't
tell "the job needs FastAPI and I don't have it" from "the job needs
FastAPI and I do" without a hardcoded skill list, and it has no way to
judge *how much* of a large job you can actually deliver. It's kept in
the repo as a deliberate point of comparison against the LLM-based
version, not as dead code.

**Moved scoring to a local LLM once keyword matching couldn't reason
about fit.** `match_llm.py` sends your profile and the job text to a
local model (via Ollama) and asks it to act as a strict, skeptical
judge — not an encouraging recruiter. Running locally means job
descriptions (which can contain client-identifying detail) never leave
your machine.

**`temperature=0` for scoring, `temperature=0.7` for proposals — on
purpose, not a leftover default.** Scoring needs to be reproducible: the
same job should get the same score every run, or the ranking becomes
untrustworthy. Proposal drafting needs the opposite — some natural
variation so proposals don't all read like the same template.

**The scoring rubric went through four real revisions**, each fixing a
concrete failure observed in testing, not a hypothetical one:
- v1 (no rubric) scored a job needing mostly backend/infra skills at
  82/100, because it just checked "does *any* skill match" rather than
  weighting how central the missing skills were.
- v2 added strict weighting of mandatory requirements, but over-corrected —
  it started penalizing gaps for skills the job never even asked for
  (e.g. docking a computer-vision job for "no backend experience").
- v3 fixed that by forcing the model to first extract what the job
  *actually* requires, then only score against that extracted list.
- v4 fixed a subtler bug: v3 still scored large, multi-skill jobs too
  generously in the middle (a job needing 3D avatars, voice, a database,
  and an installer scored 60, when the candidate could only deliver one
  small slice of it). v4 forces the model to break the job into
  components, judge each one, and score based on the *fraction* of
  components it could deliver — a strong match on 1 of 5 unrelated
  components now scores low, not medium.

**The score is computed in Python from the model's labels, not trusted
from the model's own arithmetic.** An 8B local model can classify
"can I do this component: yes/partial/no" reasonably reliably, but is
not reliable at then doing the division to turn that into a percentage —
in testing it wrote a fraction of "1 YES + 2 PARTIAL out of 12 = 0.583"
when the correct answer is 0.167, and its self-reported score inherited
that wrong number. The fix: ask the model only for the classification,
then compute `score = round((yes + 0.5*partial) / total * 100)` in code,
where the arithmetic is always correct. A regex-based filter also strips
out components the model sometimes invents for administrative busywork
("write documentation", "installation guide") before counting, since
those aren't real engineering skill areas.

**Caching turned re-runs from ~70s/job to instant.** Every scored job is
cached in `results.json`, keyed by a SHA-256 hash of its text. A normal
run only calls Ollama for jobs that changed or are new — a re-run of 5
already-scored jobs went from ~72 seconds to ~0.6 seconds (about **125x**
faster). `--fresh` bypasses the cache entirely when the prompt itself
changes and everything needs re-scoring.

**Deleting a job rebuilds the report without re-scoring anything.**
`match_llm.build_report()` is a pure function of the current
`jobs.json` + `results.json` — it never calls Ollama. `server.py`'s
`/delete_job` endpoint calls it directly after removing a job, so
`report.html` reflects the deletion immediately (and the delete button
in the browser just reloads the page to see it), without waiting for a
full rescoring pass.

### ToS-safe by design

Upwork's Terms of Service prohibit automated scraping and automated
proposal submission. This tool is built around that constraint, not
around ignoring it:

- The extension only reads the page you *already have open* — it never
  navigates Upwork on its own, crawls search results, or fetches pages
  in the background.
- Saving a job is a manual click, every time.
- Proposals are drafted, never submitted. You read, edit, and paste them
  in yourself.

If you fork this, keep it that way.

## Setup

**1. Install [Ollama](https://ollama.com) and pull the model:**
```
ollama pull llama3.1:8b
```

**2. Install the one Python dependency** (everything else is standard
library):
```
pip install requests
```

**3. Edit your profile.** Open `match_llm.py` and replace `MY_PROFILE`
and `MY_PROJECTS_BY_DOMAIN` with your own background — these are
hardcoded on purpose, not read from a config file, so there's exactly
one obvious place to look. Do the same for `MY_SKILLS` in `match.py` if
you want to use the keyword-matching baseline too.

**4. Start the local server:**
```
python server.py
```
This runs at `http://localhost:8765` and must stay running for both the
extension (to save jobs) and the report (to delete jobs) to work.

**5. Load the browser extension:**
- Open `chrome://extensions`, enable **Developer mode**
- **Load unpacked** → select the `extension/` folder

**6. Capture jobs.** Open a real Upwork job posting, click the extension
icon, click **Save this job**. Repeat for as many jobs as you want scored.

**7. Score and generate the report:**
```
python match_llm.py
```
This writes `report.html` — open it in a browser. With `server.py`
running, each job card's 🗑 delete button works and refreshes the report
automatically.

### CLI reference

```
python match_llm.py            # score any new jobs, reuse cached results, write report.html
python match_llm.py --fresh    # ignore the cache, re-score every job from scratch
python match_llm.py --list     # list all saved jobs with their index number
python match_llm.py --delete N # delete job number N (from --list) and its cached result
```

### Example data format

`jobs.json` (and your real `results.json`) are gitignored since they
contain your saved job postings. See `jobs.example.json` for the exact
shape `server.py` writes and `match_llm.py` expects — a JSON list of
`{"text": ..., "date_added": "YYYY-MM-DD", "saved_at": "<ISO timestamp>"}`
objects.

## Limitations

Read this before trusting the output blindly:

- **The local 8B model isn't perfect.** It occasionally skips the
  structured component labels the scorer needs, in which case scoring
  falls back to the model's own (less reliable) number. This is rare but
  does happen — if a score looks obviously wrong, it might be one of
  these cases.
- **Scores are guidance, not gospel.** They're a fast first filter to
  help you triage a long job list, not a replacement for reading the
  posting yourself before applying.
- **Proposals need human review.** They're grounded in your real profile
  and won't invent skills you don't have, but they're a first draft —
  read them, adjust the tone, and personalize before sending.
- **Job extraction depends on Upwork's current page structure.** The
  extension uses a chain of fallback CSS selectors and ultimately falls
  back to grabbing the page's visible text if nothing matches, so it
  degrades gracefully rather than failing outright — but a real Upwork
  layout change could still affect field extraction (title vs.
  description vs. metadata) until the selectors are updated.
- **No automated test suite.** Everything here has been verified by
  running it against real Ollama calls and real saved jobs, not by a CI
  pipeline.

## Tech stack

- **Python 3** standard library only for `server.py` (`http.server`,
  `json`, `hashlib`, `difflib`) — no framework, nothing to install to run
  the server.
- **[requests](https://pypi.org/project/requests/)** — the one external
  dependency, used by `match_llm.py` to call Ollama's local HTTP API.
- **[Ollama](https://ollama.com)** running **llama3.1:8b** locally for
  both scoring and proposal drafting.
- **Chrome Extension, Manifest V3** — vanilla JavaScript, no build step,
  no frameworks.
- **Generated HTML report** — inline CSS and a small vanilla-JS script
  block for the delete button, no external libraries or CDN dependencies.

## Project structure

```
match.py             Baseline keyword-matching scorer (no LLM)
match_llm.py          LLM-based scoring, proposal drafting, caching, report generation, CLI
server.py             Local HTTP server: saves/deletes jobs, dedups, rebuilds the report on delete
extension/            Chrome extension (Manifest V3) that captures the current Upwork job page
jobs.example.json     Example shape of jobs.json, for anyone cloning this repo
KNOWN_ISSUES.md        Known, low-priority rough edges
```
