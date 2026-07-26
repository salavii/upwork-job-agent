# ML/AI Job Agent

A personal job-hunting agent for ML/AI/data-science roles: it **finds**
jobs automatically from seven public job boards, **scores** every one
against your real resume using a **local LLM** (nothing leaves your
machine), applies a set of hard-coded fit gates tuned from real testing
(role type, seniority, location, freshness), and hands you a clean,
ranked HTML report with direct links — every morning, on a schedule, with
zero manual browsing.

It also supports a second, fully manual path: a Chrome extension that
saves individual Upwork postings you're already looking at, scored the
same way, with an honest first-draft proposal for the ones worth
applying to.

**Nothing here ever submits an application.** Every flag, cap, and
report stops at "here's what I found, ranked, with a link" — you read,
decide, and click apply yourself. See [Applying stays
manual](#applying-stays-manual) for why that's permanent, not a
missing feature.

## What it does

1. **Finds jobs automatically.** `src/sources.py` pulls current listings
   from 7 public, official job-board APIs/feeds — no scraping, no login
   bypass. See [Automatic sources](#automatic-sources) below.
2. **Filters for relevance and freshness.** Keyword-matches for ML/AI/
   data-science work, keeps only jobs posted in the last couple of days,
   and drops postings that are actually dead ("no open roles").
3. **Scores every job against your real resume**, using a local LLM (via
   [Ollama](https://ollama.com)) that breaks each posting into its main
   skill components and judges what fraction of it you could realistically
   deliver — not just "does a keyword match."
4. **Applies hard-coded fit gates** on top of the model's own judgment,
   because testing showed the model alone isn't reliable enough — see
   [Scoring gates](#scoring-gates) below.
5. **Writes a ranked HTML report** (`daily_report.html`) — every fresh,
   relevant job, highest score first, each with its score, verdict,
   flags, source, and a direct link to the original posting.
6. **Runs unattended on a schedule** (Windows Task Scheduler, 3x/day out
   of the box) so the report is just waiting for you.
7. **(Optional, manual)** A Chrome extension + local server let you save
   individual Upwork postings you're already viewing, scored the same
   way, with a drafted first-pass proposal for strong matches.

## Automatic sources

`src/sources.py` pulls jobs from seven public job boards — only
documented, official, public endpoints/feeds, nothing that requires
scraping, an API key, or bypassing a login:

| Source | Endpoint |
|---|---|
| RemoteOK | `https://remoteok.com/api` (public JSON) |
| Remotive | `https://remotive.com/api/remote-jobs` (public JSON) |
| Arbeitnow | `https://www.arbeitnow.com/api/job-board-api` (public JSON) |
| Himalayas | `https://himalayas.app/jobs/api` (public JSON) |
| Jobicy | `https://jobicy.com/api/v2/remote-jobs` (public JSON) |
| The Muse | `https://www.themuse.com/api/public/jobs` (public JSON) |
| WeWorkRemotely | `https://weworkremotely.com/categories/remote-programming-jobs.rss` (public RSS) |

Boards considered and **skipped** for not having a genuinely public/free
API: Adzuna (requires a registered API key), GitHub Jobs (discontinued),
LinkedIn/Indeed (no public job-search API for third parties).

To add another source: write one `fetch_<name>()` function returning
normalized job dicts and add it to `SOURCE_FETCHERS` at the bottom of
`sources.py` — nothing else needs to change.

**Relevance filtering.** `KEYWORD_PATTERNS` (a config constant at the top
of `sources.py`) lists the ML/AI terms to look for: machine learning, ML,
deep learning, AI, LLM, large language model, NLP, RAG, computer vision,
PyTorch, TensorFlow, fine-tuning, prompt engineering, Hugging Face, data
scientist, model training, embeddings, vector database, and more. A
single keyword hit anywhere in a long description let through a lot of
unrelated roles that mentioned "AI" once in passing, so `is_relevant()`
requires EITHER a match in the job's TITLE, OR at least two DISTINCT
keywords matched in the full text.

**Freshness filtering.** Jobs older than `MAX_JOB_AGE_DAYS` (2 by
default) are dropped — the whole point is fresh, low-competition
postings, not a backlog already buried under other applicants. A job
whose source gives no parseable date is kept but tagged `"unknown"`
rather than silently lost.

**Dedup.** Every fetched job is deduped against everything already
saved using TWO keys: an exact text hash, AND a normalized
title+company key (catches the same real posting re-served with a
slightly different description snapshot, which would otherwise slip
through as a "new" job with a different exact hash).

**Dead-posting filter.** Postings that say there's no open role right
now ("we don't currently have any open roles") are dropped entirely at
fetch time.

**Resilience.** Each source retries once on a transient network failure,
then gives up gracefully — one source's outage never crashes the run or
blocks any other source.

## Scoring gates

Testing repeatedly showed the local 8B model's own self-reported
judgment isn't reliable enough to trust alone — a Gartner "AI Strategy"
analyst role scored 100/100, a "Technical Recruiter" scored 75, because
the model conflates "this text talks a lot about AI" with "this role IS
AI engineering." So on top of the model's component-fraction scoring,
these deterministic checks run in code:

| Gate | Type | What it catches |
|---|---|---|
| **Role type** | Hard cap (15) | Titles/content indicating management, sales, recruiting, accounting, marketing, "subject matter expert," solutions/sales/value engineer, consulting, customer success, etc. — genuinely not hands-on ML/AI engineering, regardless of how much AI vocabulary surrounds it. |
| **Seniority** | Soft penalty (−12) | "Senior"/"Lead" titles, or a stated requirement of 5+ years — downweighted, not hidden, since this is a free board with no per-application cost and a strong-fitting senior role is still worth seeing. |
| **Location/eligibility** | Flag only, no score effect | "US-based," "work authorization," "on-site," "security clearance," etc. — surfaced as an informational flag in the report so you can judge eligibility yourself. |
| **Dead posting** | Dropped entirely | "We don't currently have any open roles" — not a real job. |

The role-type gate also has a **content-based backstop**: some
non-engineering roles don't signal it in the title at all (e.g. a job
that reads like business-process/consulting/customer-facing work), so
the job's full text is checked too — but only when the title doesn't
already contain a clear engineering word, so a real ML Engineer posting
that mentions "customer-facing" once in passing isn't penalized for it.

All of these constants (`ROLE_TYPE_MISMATCH_PATTERN`,
`SENIORITY_PENALTY_POINTS`, `LOCATION_RESTRICTION_PATTERN`,
`MAX_JOB_AGE_DAYS`, `KEYWORD_PATTERNS`) are near the top of
`src/match_llm.py`/`src/sources.py` and are meant to be tuned — they're
the result of iterating against real, messy job-board data, not a fixed
design.

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

**3. Edit your profile.** Open `src/match_llm.py` and replace
`MY_PROFILE` and `MY_PROJECTS_BY_DOMAIN` with your own background,
including honest weaknesses (the scorer is deliberately strict about
gaps) — these are hardcoded on purpose, not read from a config file, so
there's exactly one obvious place to look.

**4. Run it:**
```
python src/match_llm.py --daily
```
This fetches from all 7 sources, scores anything new, and writes
`daily_report.html` — open it in a browser.

### (Optional) Manual Upwork extension

If you also want to save individual Upwork postings you're browsing:

1. **Start the local server:** `python src/server.py` (runs at
   `http://localhost:8765`, must stay running for the extension to save
   jobs and for the report's delete button to work).
2. **Load the extension:** open `chrome://extensions`, enable **Developer
   mode**, **Load unpacked** → select the `extension/` folder.
3. **Capture jobs:** open a real Upwork job posting, click the extension
   icon, click **Save this job**.
4. **Score and generate the full report:** `python src/match_llm.py` —
   writes `report.html` (every job ever saved/fetched, scored, with a
   delete button per card).

## CLI reference

```
python src/match_llm.py            # score any new jobs, reuse cached results, write report.html
python src/match_llm.py --fresh    # ignore the cache, re-score every job from scratch
python src/match_llm.py --list     # list all saved jobs with their index number
python src/match_llm.py --delete N # delete job number N (from --list) and its cached result
python src/match_llm.py --fetch    # pull new jobs from all 7 boards, append to jobs.json
python src/match_llm.py --daily    # fetch + score new jobs + write daily_report.html (no proposals, fast)
```

`--daily` is the main workflow: fetch → score anything unscored (reusing
`results.json`, so a normal day is a handful of new postings, not a full
re-score) → write `daily_report.html` covering every automatically-sourced
job saved since the last successful report (a wall-clock cutoff in
`last_daily_run.json`, not just "whatever this exact run's fetch added" —
this survives a run that gets interrupted mid-way, e.g. Ollama not
running yet, without silently dropping jobs from every future report).

## Running on a schedule (Windows Task Scheduler)

`run_daily.bat` wraps `python src/match_llm.py --daily` for unattended runs:
cds into the project directory, calls the real Python interpreter
directly (not a PATH alias), runs with `-X utf8` so non-ASCII job titles
never crash the run over a console codepage mismatch, checks whether
Ollama is responding and starts it if not, and appends everything with
timestamps to `daily_run_log.txt`.

To schedule it 3x/day (9am, 2pm, 7pm):
```
schtasks /create /tn "MLJobAgent_Daily_9AM" /tr "D:\path\to\repo\run_daily.bat" /sc daily /st 09:00 /f
schtasks /create /tn "MLJobAgent_Daily_2PM" /tr "D:\path\to\repo\run_daily.bat" /sc daily /st 14:00 /f
schtasks /create /tn "MLJobAgent_Daily_7PM" /tr "D:\path\to\repo\run_daily.bat" /sc daily /st 19:00 /f
```
Useful follow-ups: `schtasks /query /tn "..." /fo LIST /v` (check status),
`schtasks /run /tn "..."` (trigger immediately, for testing),
`schtasks /delete /tn "..." /f` (remove).

## Applying stays manual

This is deliberate, not a missing feature. `--daily` stops at "here are
today's fresh, scored jobs, ranked, with direct links." It never opens a
browser, never fills in a form, and never submits anything, anywhere.

Automated application submission would violate most job boards' (and
Upwork's) Terms of Service and risks accounts being banned — a risk not
worth the marginal time saved. Reading the report, clicking through, and
applying yourself is a permanent part of the workflow.

The extension follows the same principle for Upwork specifically: it
only reads the page you already have open (never navigates or crawls on
its own), saving is a manual click every time, and drafted proposals are
never submitted automatically — you read, edit, and paste them in
yourself.

## Limitations

- **The local 8B model isn't perfect.** It occasionally skips the
  structured labels the scorer needs, falling back to a less reliable
  self-reported number. The hard-coded gates exist specifically because
  the model's own judgment on role fit wasn't trustworthy enough alone.
- **Scores are guidance, not gospel.** A fast first filter to triage a
  long job list, not a replacement for reading the posting yourself.
- **The location/eligibility gate is a heuristic text search, not true
  NLP** — it can't detect negation (e.g. "no work authorization
  required" would still match "work authorization"), so treat the flag
  as a prompt to go check, not a verdict.
- **Job extraction (Upwork extension) depends on the current page
  structure.** Falls back to grabbing visible text if selectors don't
  match, so it degrades gracefully rather than failing outright.
- **No automated test suite.** Verified by running against real Ollama
  calls and real job-board data, not a CI pipeline.

## Tech stack

- **Python 3**, standard library only for `src/server.py` (`http.server`,
  `json`, `hashlib`, `difflib`).
- **[requests](https://pypi.org/project/requests/)** — the one external
  dependency, used to call Ollama's local API and the job-board APIs.
- **[Ollama](https://ollama.com)** running **llama3.1:8b** locally for
  scoring and proposal drafting — nothing sent to any external LLM API.
- **Chrome Extension, Manifest V3** — vanilla JavaScript, no build step.
- **Generated HTML report** — inline CSS and vanilla JS, no external
  libraries or CDN dependencies.

## Project structure

```
src/
  match_llm.py        Core scoring engine, gates, proposal drafting, caching, report/daily-digest, CLI
  sources.py          Automatic job discovery from 7 public job-board APIs/feeds
  server.py           Local HTTP server for the manual Upwork extension flow (save/delete jobs)
  match.py            Deterministic keyword-matching baseline (no LLM) - kept as a comparison point
extension/            Chrome extension (Manifest V3) for the manual Upwork flow
run_daily.bat          Task Scheduler wrapper for unattended --daily runs
jobs.example.json      Example shape of jobs.json, for anyone cloning this repo
KNOWN_ISSUES.md         Known, low-priority rough edges
```

`jobs.json`, `results.json`, `report.html`, `daily_report.html`,
`last_daily_run.json`, and `daily_run_log.txt` are all gitignored
(personal job data and generated output) — see `jobs.example.json` for
the shape `src/server.py`/`src/sources.py` write.

`match.py` is a from-scratch keyword matcher (looks for skills as
whole words/phrases, computes a percentage match) — simple, fast, fully
deterministic, but can't reason about *how much* of a large job you
could actually deliver. Kept deliberately as a baseline to compare
against the LLM-based scorer, not as dead code.
