# ML/AI Job Agent

A personal job-hunting agent for ML/AI/data-science roles: it **finds**
jobs automatically from seven public job boards, **scores** every one
against your real resume using an **LLM** (a free local model via Ollama,
or a cloud model like Claude Haiku for better accuracy), applies a set of
hard-coded fit gates tuned from real testing (role type, seniority, job
type, location/eligibility), and hands you a clean, ranked HTML report
with direct links and a skill-gap analysis — every day, on a schedule,
with zero manual browsing.

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
2. **Filters for relevance and active listings.** Keyword-matches for
   ML/AI/data-science work, keeps any job still actively listed
   (regardless of how long ago it was posted — this isn't Upwork, there's
   no "first to apply wins"), and drops postings that are actually dead
   ("no open roles").
3. **Scores every job against your real resume**, using a local LLM (via
   [Ollama](https://ollama.com)) that breaks each posting into its main
   skill components and judges what fraction of it you could realistically
   deliver — not just "does a keyword match."
4. **Applies hard-coded fit gates** on top of the model's own judgment,
   because testing showed the model alone isn't reliable enough — see
   [Scoring gates](#scoring-gates) below.
5. **Maintains a persistent job list** (`daily_report.html`) — every job
   that clears the fit bar gets a PERMANENT spot, accumulating across
   every run, with a checkbox-style Remove button per job and a
   Clear-all button, so it behaves like a real to-do list rather than a
   report that resets every time — see [The persistent job
   list](#the-persistent-job-list) below.
6. **Runs unattended on a schedule** (Windows Task Scheduler, once daily
   at 1pm out of the box) so the report is just waiting for you.
7. **Flags skill gaps on near-miss jobs.** For every job scoring above 60,
   the report shows exactly what's costing you a clean match, split into
   quick-to-learn gaps (a specific tool/library) vs. large gaps (years of
   experience, a different domain), plus a "Focus on these skills" summary
   ranking the most-requested gaps across your whole list — see [Skill-gap
   analysis](#skill-gap-analysis) below.
8. **(Optional, manual)** A Chrome extension + local server let you save
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

**Active-listing filtering, not age filtering.** These are normal job
boards, not Upwork — there's no "first to apply wins" dynamic, so what
matters is whether a posting is still open, not how recently it was
posted. There is deliberately no age cutoff: a good job posted a week
ago that's still listed is exactly as valid as one posted today. Every
source's API only returns currently-listed postings in the first place,
so a job is kept unless the source gives concrete evidence it has
expired — currently only Himalayas exposes a real expiry timestamp
(`expiryDate`); every other source has no such signal, so nothing there
is ever dropped for staleness.

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
| **Full-time employee** | Excluded entirely, config-driven | Postings that read as permanent full-time EMPLOYEE roles (as opposed to contract/freelance/part-time) are dropped from the job list completely — not flagged, not scored lower — only active if `config.json`'s `work_eligibility.full_time_employee_ok` is `false`; a no-op for anyone who hasn't set that. See [Work eligibility](#work-eligibility) below. |
| **Location/eligibility** | Flag only, no score effect | On-site/citizenship/security-clearance requirements always flag. "US-based"/"work authorization" language only flags for postings that read as full-time EMPLOYEE roles — a contract/freelance/part-time posting is never flagged for this, since a contractor can legally work for a client anywhere while based elsewhere. |
| **Dead posting** | Dropped entirely | "We don't currently have any open roles" — not a real job. |

The role-type gate also has a **content-based backstop**: some
non-engineering roles don't signal it in the title at all (e.g. a job
that reads like business-process/consulting/customer-facing work), so
the job's full text is checked too — but only when the title doesn't
already contain a clear engineering word, so a real ML Engineer posting
that mentions "customer-facing" once in passing isn't penalized for it.

### Work eligibility

`config.json`'s optional `work_eligibility` section describes any real
work-authorization constraints you have, which the full-time-employee
and location gates above read from:
```json
"work_eligibility": {
  "based_in": "Italy",
  "full_time_employee_ok": false,
  "notes": "Free text describing your real situation, e.g. a student visa that only permits freelance/contract work up to a certain number of hours/year."
}
```
- `full_time_employee_ok: false` means permanent full-time EMPLOYEE
  postings are excluded entirely from the job list; freelance/contract/
  part-time postings are never affected by this.
- Missing this section entirely (or leaving `full_time_employee_ok:
  true`) makes both of these gates a complete no-op — the default is "no
  restriction," so this never affects anyone who hasn't filled it in.

All of these constants (`ROLE_TYPE_MISMATCH_PATTERN`,
`SENIORITY_PENALTY_POINTS`, `EMPLOYEE_ONLY_RESTRICTION_PATTERN`,
`KEYWORD_PATTERNS`) are near the top of `src/match_llm.py`/`src/sources.py`
and are meant to be tuned — they're the result of iterating against real,
messy job-board data, not a fixed design.

## LLM provider: free local, or paid cloud for more accuracy

Scoring can run on your own machine via **[Ollama](https://ollama.com)**
— free, private (nothing leaves your computer) — or on a cloud LLM
(**Anthropic Claude** or **OpenAI GPT**), which reasons about job fit
noticeably better than a local 8B model out of the box, at the cost of a
small per-run API charge and your job descriptions/profile being sent to
that provider.

Set it in `config.json`:
```json
{
  "llm_provider": "anthropic",
  "llm_model": "claude-haiku-4-5-20251001"
}
```
(The public `config.example.json` template defaults to
`"ollama"`/`"llama3.1:8b"` instead, so anyone cloning the repo starts on
the free, zero-setup path.)

To use a cloud provider, set the matching API key as an **environment
variable** — never in `config.json` or anywhere else in the repo:
```
# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
# macOS/Linux
export ANTHROPIC_API_KEY="sk-ant-..."
```
(`OPENAI_API_KEY` for `"openai"`.) If a cloud API call fails for any
reason (rate limit, network blip, outage), that single request falls
back to a local Ollama model (`llama3.2`) instead of crashing the whole
run — every cached result records exactly which model actually answered
it (`"model": "anthropic:claude-haiku-4-5-20251001"` vs. e.g.
`"ollama:llama3.2 (fallback)"`), so a run is never silently scored by a
mix of models without you knowing.

Scoring logic, prompts, gates, and output are byte-for-byte identical
regardless of provider — swapping providers only changes which API
answers the same prompt (see `ask_llm()` in `src/match_llm.py`). Any
cache entry from before this "model" field existed is automatically
treated as legacy and re-scored on the next run.

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

**3. Set up your profile.** Copy `config.example.json` to `config.json`
and fill in your own background, including honest weaknesses (the scorer
is deliberately strict about gaps):
```
cp config.example.json config.json
```
`config.json` is gitignored — it's your personal data and never gets
committed. If it's missing, `match_llm.py` exits immediately with a clear
message telling you to create it, instead of a raw traceback.

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
re-score) → add any newly-qualifying jobs to the persistent list → rebuild
`daily_report.html` from that list.

## The persistent job list

`daily_report.html` is not regenerated from scratch each run — it's
rendered from `job_list.json`, a small JSON file that accumulates
qualifying jobs FOREVER, across every `--daily` run, until you remove
them yourself:

- **Accumulates, doesn't wipe.** Each run adds newly-found relevant jobs
  to the existing list. The same job is never added twice (deduped by
  the same job-text hash used everywhere else).
- **Auto-drops bad jobs at the door.** A job only earns a spot if its
  score is above `FIT_SCORE_THRESHOLD` (40 by default) AND it doesn't
  trip the role-type hard-cap gate — low-scoring or off-field jobs never
  clutter the list in the first place.
- **Date stamps, sorted newest-found-first.** Each card shows both when
  the posting says it was made ("Posted") and when this tool first
  found it ("Found") — the list is sorted by the latter, since it's
  always a precise timestamp (some boards don't expose a reliable
  posted date).
- **Per-job Remove button.** Each card has a small "×" button. Clicking
  it calls `src/server.py`'s `/remove_daily_job` endpoint, which deletes
  the job from `job_list.json` PERMANENTLY — its hash also goes into a
  `"removed_hashes"` tombstone list, so a later run re-discovering the
  same posting can never silently bring it back.
- **Clear-all button.** Wipes the entire list (and the removed-hashes
  tombstone list) for a genuine fresh start.
- **Score, verdict, source badge, flags, and the "Open job" link** are
  still shown on every card — no proposal/cover-letter section, same as
  before.

`src/server.py` must be running for Remove/Clear-all to work (same
requirement as `report.html`'s delete button) — see
[Setup](#optional-manual-upwork-extension).

## Skill-gap analysis

Every job scoring above `SKILL_GAP_SCORE_THRESHOLD` (60 by default) is a
realistic near-miss — a role you could plausibly have gotten, if not for
some specific gap. The same scoring prompt/response already extracts a
`GAPS:` list (requirements your profile lacks); this feature just adds one
more thing to that list for free, no extra LLM calls: whether each gap is
`(quick to learn)` — a specific tool/library/framework learnable in
days/weeks (e.g. Docker, a specific API, a JS framework) — or a
`(large gap)` — years of experience or a genuinely different domain.

- **Per-job.** Each above-60 card shows a line like "Strong match, but
  missing: **quick to learn:** Docker · **large gap:** 5+ years production
  experience" — only real gaps, nothing you already have.
- **Aggregate summary.** The top of `daily_report.html` has a "🎯 Focus on
  these skills" section, ranking every gap across your whole above-60 list
  by how often it shows up — quick wins first, since a gap that's both
  frequent and fast to learn unlocks the most near-miss jobs for the least
  effort (see `compute_skill_gap_summary()` in `src/match_llm.py`).
- Only computed for roles you're actually eligible for — full-time
  postings excluded by the [work eligibility](#work-eligibility) gate never
  get analyzed, since there's no point spending analysis on a job you
  can't take anyway.

## Running on a schedule (Windows Task Scheduler)

`run_daily.bat` wraps `python src/match_llm.py --daily` for unattended runs:
cds into the project directory, calls the real Python interpreter
directly (not a PATH alias), runs with `-X utf8` so non-ASCII job titles
never crash the run over a console codepage mismatch, checks whether
Ollama is responding and starts it if not, and appends everything with
timestamps to `daily_run_log.txt`.

To schedule it once daily at 1pm:
```
schtasks /create /tn "MLJobAgent_Daily_1PM" /tr "D:\path\to\repo\run_daily.bat" /sc daily /st 13:00 /f
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
  scoring and proposal drafting by default — nothing sent to any
  external LLM API unless you opt into a cloud provider (see [LLM
  provider](#llm-provider-free-local-or-paid-cloud-for-more-accuracy)).
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
config.example.json    Template for config.json - copy and fill in your own profile
jobs.example.json      Example shape of jobs.json, for anyone cloning this repo
KNOWN_ISSUES.md         Known, low-priority rough edges
```

`config.json` (your personal profile), `jobs.json`, `results.json`,
`report.html`, `daily_report.html`, `job_list.json`, and
`daily_run_log.txt` are all gitignored (personal data and generated
output) — see `config.example.json` for the profile shape and
`jobs.example.json` for the shape `src/server.py`/`src/sources.py` write.

`match.py` is a from-scratch keyword matcher (looks for skills as
whole words/phrases, computes a percentage match) — simple, fast, fully
deterministic, but can't reason about *how much* of a large job you
could actually deliver. Kept deliberately as a baseline to compare
against the LLM-based scorer, not as dead code.
