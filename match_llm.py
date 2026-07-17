"""
match_llm.py

Step 2 of the Upwork job-scoring agent: LLM-based scoring.

What this script does:
- Connects to a locally running Ollama server (see ask_ollama()).
- Has my profile hardcoded as a short text block (MY_PROFILE).
- Defines score_job(job_text), which sends BOTH my profile and a job
  posting to the model, asking it to act like a technical recruiter and
  return a score, strengths, gaps, and a verdict, in a fixed format.
- Prints the result nicely for one example job.

This is a different, LLM-based approach from match.py (which uses plain
keyword matching with no AI model involved). match.py is left untouched
as our simple, deterministic baseline to compare against.

Requirements before running this file:
1. Ollama must be installed and running on this machine.
2. The llama3.1:8b model must be pulled already (run: ollama pull llama3.1:8b).
   Ollama's default install runs a local web server at http://localhost:11434
   as soon as it starts, which is what this script talks to.
"""

# `requests` is a popular third-party library for making HTTP calls. We
# use it here to send a normal HTTP request to Ollama's local API,
# because Ollama exposes itself as a small web server on your machine.
import requests

# `re` (regular expressions) is used to pull just the SCORE and VERDICT
# lines out of the model's reply, so we can rank jobs without printing
# the model's full raw text for each one.
import re

# `html` (standard library) escapes text like "<" and "&" into their
# safe HTML equivalents, so job text or model output can never
# accidentally break our generated HTML page's structure.
import html

# `datetime` is used to stamp today's date onto the HTML report title,
# and to turn each job's "date_added" string into a nicely formatted
# section heading (e.g. "July 17, 2026").
import datetime

# `json` is used to read jobs.json, which is now written by server.py as
# a JSON list of job objects (see server.py's save_jobs()).
import json

# `sys` gives us access to command-line arguments (sys.argv), so this
# file can be run as `python match_llm.py --list`, `--delete N`, or
# `--fresh` instead of only the default scoring run.
import sys

# `hashlib` is used to turn a job's text into a short, stable ID (a
# hash), so we can cache that job's result in results.json and recognize
# "this is the same job I already scored" even without needing a
# database - two identical job texts always hash to the same value.
import hashlib

# The file server.py saves scraped jobs into: a JSON list of objects
# shaped like {"text": "...", "date_added": "YYYY-MM-DD", "saved_at": "..."}.
JOBS_JSON_FILE = "jobs.json"

# Where already-scored jobs are cached, keyed by a hash of their text, so
# re-running this script doesn't re-call Ollama for jobs we've already
# scored. See load_results()/save_results()/hash_job_text() below.
RESULTS_JSON_FILE = "results.json"

# Where the HTML report gets written.
REPORT_FILE = "report.html"

# Only jobs scoring at or above this get a drafted proposal - low-scoring
# jobs aren't worth spending the client's (or the model's) time on.
PROPOSAL_SCORE_THRESHOLD = 70

# This is the standard local address Ollama listens on. "11434" is just
# the fixed port number Ollama always uses by default.
OLLAMA_URL = "http://localhost:11434/api/generate"

# The name of the model we want Ollama to use. This must match a model
# you've already pulled locally (check with `ollama list` in a terminal).
MODEL_NAME = "llama3.1:8b"

# My profile, as plain text. This gets pasted straight into the prompt
# we send the model, so it has context on my background when judging
# whether a job is a good fit. Deliberately does NOT pre-list weaknesses
# (e.g. "backend experience is limited") - the model is expected to infer
# any gaps itself by comparing what a job asks for against what's
# actually present here.
MY_PROFILE = """[REDACTED - moved to gitignored config.json]"""

# My concrete projects, grouped by domain. draft_proposal() uses this to
# make sure it only pulls in projects that actually match the job's
# domain (e.g. never mention an LLM project in a computer-vision
# proposal, or vice versa).
MY_PROJECTS_BY_DOMAIN = """[REDACTED - moved to gitignored config.json]"""

# Scoring rules for the model to follow. This has gone through two
# rounds of fixes:
# - v1 just asked for "a match score" with no guidance on how strict to
#   be, and the model came back too generous (82/100 on a job that was
#   mostly heavy backend work I can't do alone).
# - v2 added strict weighting of mandatory/core requirements, but
#   OVERCORRECTED: the model started penalizing gaps (e.g. "no
#   backend/infra experience") even on jobs that never asked for
#   backend/infra at all - it was judging me against my whole profile's
#   weaknesses instead of against what THIS job actually needs.
# - v3 (this version) explicitly forces the model to first extract what
#   the job actually requires, and only count a gap against me if it
#   maps to one of those requirements. A skill I lack that the job never
#   asked for must NOT lower the score.
STRICT_SCORING_GUIDELINES = """
Score as a STRICT freelance-fit judge, not an encouraging recruiter. Be
skeptical: most real jobs need several different skills, and partially
matching is not the same as being able to deliver the job. At the same
time, be FAIR: only judge the candidate against what THIS job actually
asks for, never against unrelated skills the candidate happens to lack.

Follow these steps before assigning a score:
1. Extract a short list of what THIS job actually requires, based only
   on the job posting text. Split that list into MANDATORY/CORE
   requirements (explicitly required/mandatory, or clearly central to
   what the job is about) versus NICE-TO-HAVE requirements (secondary
   features, bonus skills, or minor details).
2. Compare the candidate profile only against that extracted list.
   - A "gap" only counts if it maps to something on this list. If the
     job never mentioned or implied a skill (e.g. backend frameworks,
     deployment, a specific language), the candidate's lack of that
     skill is IRRELEVANT and must NOT be listed as a gap or lower the
     score.
   - If the candidate is missing one or more MANDATORY/CORE items from
     the extracted list, the score MUST drop sharply - a missing
     essential, job-relevant skill caps the score low even if many
     nice-to-haves match.
3. Only after core, job-relevant requirements are covered should
   nice-to-have matches push the score higher.

Use these score anchors as a guide:
- 80-100: the candidate covers essentially all of what THIS job actually
  requires (mandatory/core requirements included), so they could
  realistically deliver most of it alone.
- 40-60: the candidate covers only part of what THIS job requires -
  some job-relevant mandatory/core requirements are weak or missing.
- Below 40: the candidate is missing multiple job-relevant
  mandatory/core requirements and would need significant help to
  deliver this job.
"""

# The exact format we want the model to reply in. Spelling this out in
# the prompt (instead of just asking a free-form question) makes the
# model's reply predictable and easy to read - and easier to parse with
# code later, if we want to extract just the score, for example.
RESPONSE_FORMAT_INSTRUCTIONS = """
Respond in EXACTLY this format, with nothing before or after it, and no
markdown formatting (no asterisks, no headers). Fill in JOB REQUIRES
FIRST, and only list a gap in GAPS if it also appears in JOB REQUIRES -
if a candidate weakness is not in JOB REQUIRES, it must be left out of
GAPS entirely, since the job never asked for it:

JOB REQUIRES: <comma-separated list of what this job actually needs, mandatory items marked with (core)>
SCORE: <a single integer from 0 to 100>
STRENGTHS:
- <strength 1>
- <strength 2>
- <strength 3 (optional)>
GAPS:
- <gap - must be an item from JOB REQUIRES that the candidate lacks>
- <gap 2 (optional, same rule)>
VERDICT: <one short sentence>
"""


def ask_ollama(prompt, temperature=0):
    """
    Send `prompt` to the local Ollama server and return the model's
    text reply as a string.

    How it works:
    - Ollama's /api/generate endpoint expects a JSON body with at least
      "model" (which model to use) and "prompt" (what to ask it).
    - By default, Ollama STREAMS its reply back piece by piece. To keep
      this first test as simple as possible, we set "stream": False so
      Ollama waits and sends back the whole answer in one single
      response instead of many small chunks.
    - "temperature" controls how random the model's word choices are.
      Higher temperature (e.g. 0.7-1.0) makes replies more varied and
      "creative" - good for brainstorming, bad for scoring, because the
      same job could get a different score every time we ask. Setting
      temperature to 0 makes the model always pick its single most
      likely next word, which is what we want for a scoring task: the
      same (profile, job) pair should produce the same score every run,
      so the ranking is trustworthy and comparable across jobs and over
      time. We pass it inside an "options" object, which is where
      Ollama expects generation settings like this to live.
    - requests.post(...) sends the HTTP request; .json() parses the
      JSON reply into a normal Python dictionary.
    - The actual generated text lives under the "response" key of that
      dictionary, so that's what we return.
    """
    request_body = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    response = requests.post(OLLAMA_URL, json=request_body)

    # Raise an error early if Ollama responded with a failure status
    # code (e.g. model not found, server not running), instead of
    # silently continuing with a broken response.
    response.raise_for_status()

    response_data = response.json()
    return response_data["response"]


def score_job(job_text):
    """
    Ask the model to evaluate how well MY_PROFILE fits the given job
    posting (`job_text`), acting as a technical recruiter.

    We build one big prompt containing:
      1. Instructions telling the model what role to play (a strict
         freelance-fit judge, not an encouraging recruiter) and what to
         evaluate.
      2. The strict scoring rubric (see STRICT_SCORING_GUIDELINES
         above), so mandatory/core requirements are weighted far more
         heavily than nice-to-haves.
      3. My profile, so it knows my background.
      4. The job posting text, so it knows what's being asked for.
      5. The strict response format we want back (see
         RESPONSE_FORMAT_INSTRUCTIONS above), so the reply is
         consistent and easy to read every time.

    Returns the model's raw reply as a string (already in the format
    described above), which we can print directly.
    """
    prompt = f"""You are a strict freelance-fit judge deciding whether a candidate
should apply to a job posting. Your job is to protect the candidate's
time - be skeptical, not encouraging.

{STRICT_SCORING_GUIDELINES}

CANDIDATE PROFILE:
{MY_PROFILE}

JOB POSTING:
{job_text}

{RESPONSE_FORMAT_INSTRUCTIONS}
"""

    # temperature=0 so the same job scores the same way every time we
    # run this - see the comment on ask_ollama() for why that matters.
    return ask_ollama(prompt, temperature=0)


def draft_proposal(job_text):
    """
    Ask the model to write a short Upwork proposal for `job_text`, based
    on MY_PROFILE.

    Unlike score_job(), this call uses a HIGHER temperature (0.7) on
    purpose. Scoring needs to be consistent every time (temperature=0),
    but writing needs some natural variation in wording and phrasing to
    avoid sounding stiff and robotic - a bit of randomness here is a
    feature, not a bug.

    The earlier version of this prompt just said "pick whichever skills
    are most relevant" without forcing the model to commit to a domain
    first - so on a computer-vision job, it sometimes still mentioned
    LLM fine-tuning, because that's an impressive-sounding item
    elsewhere in my profile, even though it has nothing to do with the
    job. This version fixes that by making domain classification an
    explicit first step, and restricting the proposal to only that
    domain's skills/projects (from MY_PROJECTS_BY_DOMAIN).

    The prompt asks for a short (120-160 word) proposal that:
    - first identifies the job's domain (e.g. computer vision vs.
      LLM/NLP vs. backend/infra), silently, before writing anything
    - opens with something specific to this job, not a generic greeting
    - names 2-3 of my most relevant skills/projects for THIS job's
      domain only - never an impressive but unrelated one
    - stays honest - never claims a skill that isn't in MY_PROFILE
    - ends with a simple call to action
    - reads like a warm, human message, not an overly formal cover letter

    Returns the drafted proposal as a plain string.
    """
    prompt = f"""You are me, a freelancer, writing a short Upwork proposal to apply
to the job posting below. Write in first person, as if I'm writing it
myself.

CANDIDATE PROFILE (this is who I am - only mention skills/experience
that actually appear here; never invent or exaggerate anything):
{MY_PROFILE}

MY PROJECTS, GROUPED BY DOMAIN:
{MY_PROJECTS_BY_DOMAIN}

JOB POSTING:
{job_text}

Before writing anything, decide which ONE domain this job is mainly
about (for example: computer vision, LLM/NLP, or backend/infra). Then
write the proposal using ONLY skills and projects from that matching
domain above. Do not mention skills or projects from a different,
unrelated domain, even if they sound impressive - e.g. never mention
LLM/NLP work (GPT from scratch, LoRA, RAG, Hugging Face) in a
computer-vision proposal, and never mention computer-vision work
(ResNet, CNNs, EuroSAT, plant classification) in an LLM/NLP proposal.

Write a proposal that:
- Is 120-160 words long.
- Opens by addressing the client's specific need described in the job
  posting - not a generic "I am excited to apply" opener.
- Names 2-3 of my most relevant skills or projects FOR THIS SPECIFIC
  JOB'S DOMAIN ONLY (see the domain rule above).
- Is completely honest - do NOT claim any skill, tool, or experience
  that isn't in my profile or projects above.
- Ends with a simple, low-pressure call to action (e.g. inviting a
  quick chat or asking a clarifying question).
- Sounds human and warm, like a real person wrote it - not robotic,
  not overly formal, no corporate buzzwords.

Respond with ONLY the proposal text itself - no preamble, no headers,
no notes about word count or which domain you picked.
"""

    # temperature=0.7 (higher than scoring) so the writing sounds natural
    # instead of terse/robotic - see the docstring above for why.
    raw_reply = ask_ollama(prompt, temperature=0.7)

    # The prompt above already asks for no preamble, but an 8B model
    # doesn't always obey that - strip_proposal_preamble() is the
    # reliable guarantee, on top of (not instead of) the prompt instruction.
    return strip_proposal_preamble(raw_reply)


# A small set of phrases that signal the model is narrating ABOUT the
# proposal ("Here is the proposal:", "Sure, I'd be happy to help...")
# rather than writing the proposal itself. Only used to recognize
# leading paragraphs to remove - see strip_proposal_preamble() below.
PREAMBLE_PATTERNS = [
    r"^here'?s?\s+(is\s+)?(the|your|my)?\s*proposal",
    r"^here'?s my attempt",
    r"^i'?d be happy to help",
    r"^sure[,!]",
    r"^okay,?\s",
    r"^based on the job posting",
    r"^let me (write|draft)",
]


def strip_proposal_preamble(raw_text):
    """
    Remove any leading narration the model added before the actual
    proposal (e.g. "Here is the proposal:" or "I'd be happy to help...
    Here's my attempt..."), as a reliable backstop for when the model
    ignores the "no preamble" instruction in the prompt.

    How it works:
    - Split the reply into paragraphs, wherever there's a blank line.
    - Look at paragraphs from the top. If a paragraph's text matches one
      of the PREAMBLE_PATTERNS above (the model talking ABOUT the
      proposal, not writing it), drop that paragraph and check the next
      one the same way.
    - Stop as soon as we reach a paragraph that does NOT match a
      preamble pattern - that's where the real proposal starts, and we
      leave everything from there onward untouched.

    This only ever removes paragraphs from the very top of the reply,
    and only ones that match a known lead-in phrase, so real proposal
    content (which won't match these patterns) is never at risk of
    being deleted.
    """
    paragraphs = raw_text.strip().split("\n\n")

    while len(paragraphs) > 1:
        first_paragraph = paragraphs[0].strip()
        looks_like_preamble = any(
            re.match(pattern, first_paragraph, re.IGNORECASE) for pattern in PREAMBLE_PATTERNS
        )
        if looks_like_preamble:
            paragraphs.pop(0)
        else:
            break

    return "\n\n".join(paragraphs).strip()


def load_jobs(file_path):
    """
    Read `file_path` (jobs.json, written by server.py) and return its
    contents as a list of job dicts, each with at least "text" and
    "date_added" keys.

    If the file doesn't exist yet (no jobs saved so far) or somehow
    isn't valid JSON, we return an empty list instead of crashing.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            jobs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    return jobs


def extract_title_and_metadata(job_text):
    """
    Pull a short title and (if present) a metadata line out of a job's
    raw text, for display purposes only (this doesn't affect scoring).

    - Title: always just the job's first line, so both old plain-text
      jobs (a single unbroken paragraph) and jobs captured by the
      browser extension (title on its own first line) work the same way.
    - Metadata: the browser extension writes job text as
      "title\\n\\nExperience level: ... | Pricing type: ... \\n\\ndescription...".
      If the SECOND paragraph (split on a blank line) looks like that
      metadata line - mentioning things like "Experience level" or
      "Hours per week" - we treat it as metadata. Older jobs that were
      typed/pasted by hand won't have this paragraph, so metadata simply
      comes back as None for them, which is fine.
    """
    stripped_text = job_text.strip()
    title = stripped_text.splitlines()[0].strip() if stripped_text else ""

    metadata = None
    paragraphs = stripped_text.split("\n\n")
    if len(paragraphs) > 1:
        second_paragraph = paragraphs[1].strip()
        if re.search(r"Experience level|Pricing type|Hours per week|Duration", second_paragraph):
            metadata = second_paragraph

    return title, metadata


def save_jobs(jobs):
    """
    Write `jobs` (a list of job dicts) back to JOBS_JSON_FILE. Used by
    --delete to persist a job's removal. We always rewrite the whole
    file, since JSON is one big list rather than something we can
    append/remove single lines from.
    """
    with open(JOBS_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)


def hash_job_text(job_text):
    """
    Turn a job's text into a short, stable identifier (a SHA-256 hash),
    used as its key in results.json. The same job text always produces
    the same hash, which is exactly what we want for caching: "have I
    already scored this exact job before?"
    """
    return hashlib.sha256(job_text.strip().encode("utf-8")).hexdigest()


def load_results():
    """
    Read RESULTS_JSON_FILE and return its contents as a dict mapping
    job-text hash -> cached result (title, score, verdict, proposal).
    Returns an empty dict if the file doesn't exist yet or isn't valid
    JSON, so a missing/corrupt cache never crashes a run - it just means
    everything gets (re-)scored.
    """
    try:
        with open(RESULTS_JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_results(results):
    """
    Write the `results` cache dict back to RESULTS_JSON_FILE.
    """
    with open(RESULTS_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def parse_score_and_verdict(raw_reply):
    """
    Pull just the SCORE and VERDICT values out of a score_job() reply,
    so we can rank jobs without re-printing the model's full response
    for each one.

    We use simple regexes to find "SCORE: <number>" and
    "VERDICT: <text>" anywhere in the reply. If the model didn't follow
    the format for some reason and a piece can't be found, we fall back
    to a 0 score / "(no verdict found)" placeholder instead of crashing,
    so one malformed reply doesn't stop the whole ranking.
    """
    score_match = re.search(r"SCORE:\s*(\d+)", raw_reply)
    verdict_match = re.search(r"VERDICT:\s*(.+)", raw_reply)

    score = int(score_match.group(1)) if score_match else 0
    verdict = verdict_match.group(1).strip() if verdict_match else "(no verdict found)"

    return score, verdict


def score_to_color(score):
    """
    Map a 0-100 score to a hex color for the HTML report:
    green (>= 70), orange (50-69), red (< 50). A score of None (a job
    that hasn't been scored yet - see build_report()) gets a neutral gray.
    """
    if score is None:
        return "#888888"  # gray - not scored yet
    elif score >= 70:
        return "#1b8a3d"  # green
    elif score >= 50:
        return "#c77700"  # orange
    else:
        return "#c53030"  # red


def format_date_heading(date_added):
    """
    Turn a "YYYY-MM-DD" string into a friendly heading like
    "July 17, 2026". Falls back to the raw string unchanged if it's
    missing or not in the expected format, rather than crashing the
    whole report over one malformed date.
    """
    try:
        return datetime.datetime.strptime(date_added, "%Y-%m-%d").strftime("%B %d, %Y")
    except (ValueError, TypeError):
        return date_added or "Unknown date"


def build_job_card_html(entry):
    """
    Build the HTML for a single job "card" (rank, color-coded score,
    title, metadata line if present, verdict, and drafted proposal if
    there is one).

    We use html.escape() on every piece of text that came from the job
    posting or the model's reply, since that text is unpredictable and
    could otherwise contain characters (like "<" or "&") that would
    break the HTML page's structure.

    entry["score"] may be None for a job that hasn't been scored yet
    (see build_report()) - shown as "Not scored yet" instead of a number.
    """
    color = score_to_color(entry["score"])
    score_label = "Not scored yet" if entry["score"] is None else f"{entry['score']}/100"

    metadata_html = ""
    if entry["metadata"]:
        metadata_html = f'<p class="metadata">{html.escape(entry["metadata"])}</p>'

    proposal_html = ""
    if entry["proposal"]:
        # Convert newlines in the proposal text to <br> so paragraph
        # breaks show up correctly in the browser.
        proposal_text_html = html.escape(entry["proposal"]).replace("\n", "<br>")
        proposal_html = f"""
        <div class="proposal-box">
            <div class="proposal-label">Drafted proposal</div>
            <p>{proposal_text_html}</p>
        </div>
        """

    # data-job-id carries this job's text hash (see hash_job_text()), so
    # the delete button's JavaScript (in build_html_report()) can tell
    # server.py's /delete_job endpoint exactly which job to remove,
    # without needing to send the whole job text back.
    return f"""
    <div class="card" data-job-id="{entry['job_id']}">
        <div class="card-header">
            <span class="rank">#{entry['rank']}</span>
            <span class="score" style="color: {color};">{score_label}</span>
        </div>
        <h3 class="job-title">{html.escape(entry['title'])}</h3>
        {metadata_html}
        <p class="verdict">{html.escape(entry['verdict'])}</p>
        {proposal_html}
        <button class="delete-button" data-job-id="{entry['job_id']}">🗑 Delete</button>
    </div>
    """


def build_html_report(results_by_date, sorted_dates):
    """
    Build a full HTML page (as a string), grouped into one section per
    date (newest first), with each date's jobs shown as cards ranked
    high-to-low by score within that section.

    `results_by_date` is a dict: {"YYYY-MM-DD": [entry, entry, ...]},
    where each entry is a dict with keys "rank", "score", "title",
    "metadata", "verdict", "proposal" (metadata/proposal may be None).
    `sorted_dates` is the list of those date keys, already ordered
    newest-first - see the main block below for how both are built.
    """
    generated_at_str = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")

    sections_html = ""
    for date_added in sorted_dates:
        date_heading = format_date_heading(date_added)
        cards_html = "".join(build_job_card_html(entry) for entry in results_by_date[date_added])
        sections_html += f"""
        <section class="date-section">
            <h2 class="date-heading">{html.escape(date_heading)}</h2>
            {cards_html}
        </section>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Upwork Job Matches</title>
<style>
    * {{
        box-sizing: border-box;
    }}
    body {{
        font-family: "Segoe UI", Arial, Helvetica, sans-serif;
        background-color: #f4f5f7;
        color: #22262b;
        max-width: 820px;
        margin: 40px auto;
        padding: 0 20px 60px 20px;
        line-height: 1.5;
    }}
    .page-header {{
        margin-bottom: 32px;
    }}
    h1 {{
        font-size: 26px;
        margin: 0 0 6px 0;
    }}
    .generated-at {{
        font-size: 13px;
        color: #777;
        margin: 0;
    }}
    .date-section {{
        margin-bottom: 36px;
    }}
    .date-heading {{
        font-size: 18px;
        font-weight: 600;
        color: #333;
        border-bottom: 2px solid #ddd;
        padding-bottom: 8px;
        margin-bottom: 18px;
    }}
    .card {{
        background-color: #ffffff;
        border: 1px solid #e2e2e2;
        border-radius: 10px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
    }}
    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 15px;
        font-weight: 600;
    }}
    .rank {{
        color: #888;
    }}
    .score {{
        font-size: 21px;
        font-weight: 700;
    }}
    .job-title {{
        font-size: 17px;
        font-weight: 600;
        margin: 10px 0 4px 0;
    }}
    .metadata {{
        font-size: 13px;
        color: #666;
        margin: 0 0 10px 0;
    }}
    .verdict {{
        margin: 0 0 10px 0;
        color: #444;
    }}
    .proposal-box {{
        background-color: #f7f8fa;
        border-left: 4px solid #888;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 10px;
        font-size: 14px;
    }}
    .proposal-label {{
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        color: #666;
        margin-bottom: 6px;
    }}
    .delete-button {{
        margin-top: 14px;
        padding: 6px 12px;
        font-size: 13px;
        border: 1px solid #e0b4b4;
        border-radius: 6px;
        background-color: #fdf2f2;
        color: #a33;
        cursor: pointer;
    }}
    .delete-button:hover {{
        background-color: #fbe4e4;
    }}
</style>
</head>
<body>
    <div class="page-header">
        <h1>Upwork Job Matches</h1>
        <p class="generated-at">Generated {generated_at_str}</p>
    </div>
    {sections_html}

    <script>
        // This runs in the browser when report.html is opened. It wires
        // up every "Delete" button to call server.py's /delete_job
        // endpoint (server.py must be running for this to work - see
        // the comment on DELETE_URL below).
        const DELETE_URL = "http://localhost:8765/delete_job";

        document.querySelectorAll(".delete-button").forEach(function (button) {{
            button.addEventListener("click", function () {{
                if (!window.confirm("Delete this job?")) {{
                    return;
                }}

                const jobId = button.getAttribute("data-job-id");

                fetch(DELETE_URL, {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ job_id: jobId }}),
                }})
                    .then(function (response) {{
                        if (!response.ok) {{
                            throw new Error("Server responded with an error.");
                        }}
                        // server.py has already rebuilt report.html (with
                        // this job gone) by the time it replies, so just
                        // reload the page to show the fresh version -
                        // no manual DOM surgery needed.
                        location.reload();
                    }})
                    .catch(function () {{
                        // Most likely cause: server.py isn't running, so
                        // the fetch couldn't connect at all.
                        alert("Start server.py to enable delete.");
                    }});
            }});
        }});
    </script>
</body>
</html>
"""


# The `if __name__ == "__main__":` line below means: "only run this code
# when this file is executed directly (e.g. `python match_llm.py`), not
# when it's imported by another file."
def cli_list_jobs():
    """
    Handle `python match_llm.py --list`: print every job in jobs.json
    with a 1-based index number and its title, so the user knows which
    number to pass to --delete.
    """
    jobs = load_jobs(JOBS_JSON_FILE)

    if not jobs:
        print(f"No jobs found in {JOBS_JSON_FILE}.")
        return

    print(f"{len(jobs)} job(s) in {JOBS_JSON_FILE}:\n")
    for index, job in enumerate(jobs, start=1):
        title, _ = extract_title_and_metadata(job.get("text", ""))
        date_added = job.get("date_added", "?")
        print(f"{index}. [{date_added}] {title}")


def cli_delete_job(job_number):
    """
    Handle `python match_llm.py --delete N`: remove job number N (as
    shown by --list) from jobs.json, and also remove its cached result
    from results.json if one exists, so a deleted job doesn't linger in
    either file.
    """
    jobs = load_jobs(JOBS_JSON_FILE)

    if job_number < 1 or job_number > len(jobs):
        print(f"No job #{job_number} - there are only {len(jobs)} job(s). Use --list to see them.")
        return

    # Lists are 0-indexed but we show/accept 1-based numbers to the user.
    removed_job = jobs.pop(job_number - 1)
    save_jobs(jobs)

    title, _ = extract_title_and_metadata(removed_job.get("text", ""))

    results = load_results()
    removed_hash = hash_job_text(removed_job.get("text", ""))
    had_cached_result = removed_hash in results
    if had_cached_result:
        del results[removed_hash]
        save_results(results)

    print(f"Deleted job #{job_number}: {title}")
    if had_cached_result:
        print("Also removed its cached result from results.json.")


def build_report():
    """
    Rebuild report.html directly from the CURRENT jobs.json and cached
    results.json, WITHOUT calling Ollama. Jobs that don't have a cached
    result yet are shown as "Not scored yet" cards instead of being
    scored on the spot.

    This is the single source of truth for turning jobs.json + the
    results cache into a grouped report, used both by the normal scoring
    run below (after it finishes scoring/caching) and by server.py right
    after a job is deleted - so report.html reflects jobs.json
    immediately, without needing a full rescoring run.

    Returns (results_by_date, sorted_dates) so callers that also want to
    print a terminal summary (like run_scoring_and_report()) don't have
    to rebuild the same grouping a second time.
    """
    jobs = load_jobs(JOBS_JSON_FILE)
    results_cache = load_results()

    # Group jobs by date_added, so today's jobs stay visually separate
    # from older ones in the HTML report.
    results_by_date = {}

    for job in jobs:
        job_text = job.get("text", "")
        date_added = job.get("date_added") or "Unknown date"
        title, metadata = extract_title_and_metadata(job_text)
        job_hash = hash_job_text(job_text)

        cached = results_cache.get(job_hash)
        if cached:
            score = cached["score"]
            verdict = cached["verdict"]
            proposal = cached.get("proposal")
        else:
            # No cached result for this job - show it as unscored rather
            # than calling Ollama here (that only happens in
            # run_scoring_and_report(), never as a side effect of
            # rebuilding the report).
            score = None
            verdict = 'Not scored yet - run "python match_llm.py" to score this job.'
            proposal = None

        entry = {
            "score": score,
            "verdict": verdict,
            "title": title,
            "metadata": metadata,
            "proposal": proposal,
            "job_id": job_hash,  # lets the HTML report's delete button identify this job
        }
        results_by_date.setdefault(date_added, []).append(entry)

    # Within each date, rank scored jobs highest-first; unscored jobs
    # (score is None) sort to the bottom of their date section rather
    # than crashing a plain numeric sort.
    for date_added, entries in results_by_date.items():
        entries.sort(key=lambda entry: (entry["score"] is None, -(entry["score"] or 0)))
        for rank, entry in enumerate(entries, start=1):
            entry["rank"] = rank

    # Newest date first. "YYYY-MM-DD" strings sort correctly as plain
    # text, so a simple reverse-sorted() is enough here.
    sorted_dates = sorted(results_by_date.keys(), reverse=True)

    report_html = build_html_report(results_by_date, sorted_dates)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_html)

    return results_by_date, sorted_dates


def run_scoring_and_report(use_cache):
    """
    The main pipeline: score every job in jobs.json (reusing cached
    results from results.json unless `use_cache` is False), then hand
    off to build_report() to group everything by date, write
    report.html, and print a terminal summary.
    """
    jobs = load_jobs(JOBS_JSON_FILE)
    results_cache = load_results()

    print(f"Loaded {len(jobs)} jobs from {JOBS_JSON_FILE}. Scoring each with {MODEL_NAME}...\n")

    for job in jobs:
        job_text = job.get("text", "")
        title, _ = extract_title_and_metadata(job_text)
        job_hash = hash_job_text(job_text)

        # Reuse a cached result if we have one for this exact job text
        # and the caller didn't ask for a fresh re-score (--fresh).
        if use_cache and job_hash in results_cache:
            print(f"Reused cached: {title}")
            continue

        print(f"Scoring new: {title}")
        raw_reply = score_job(job_text)
        score, verdict = parse_score_and_verdict(raw_reply)

        # Only draft a proposal for high-scoring jobs - it's not worth
        # spending the model's (or the client's) time on a poor fit.
        proposal = None
        if score >= PROPOSAL_SCORE_THRESHOLD:
            proposal = draft_proposal(job_text)

        results_cache[job_hash] = {
            "title": title,
            "score": score,
            "verdict": verdict,
            "proposal": proposal,
        }

    # Persist the (possibly updated) cache so future runs - and
    # build_report() below - can see today's results too.
    save_results(results_cache)

    results_by_date, sorted_dates = build_report()

    print("\n===== Ranked Job Matches (grouped by date) =====\n")
    for date_added in sorted_dates:
        print(f"--- {format_date_heading(date_added)} ---")
        for entry in results_by_date[date_added]:
            score_display = "Not scored yet" if entry["score"] is None else entry["score"]
            print(f"#{entry['rank']} - SCORE: {score_display} - {entry['title']}")
            if entry["metadata"]:
                print(f"   {entry['metadata']}")
            print(f"   Verdict: {entry['verdict']}")
            if entry["proposal"]:
                print("\n   Drafted proposal:")
                print(f"   {entry['proposal']}")
            print()
        print()

    print(f"HTML report saved to {REPORT_FILE}")


if __name__ == "__main__":
    cli_args = sys.argv[1:]

    if "--list" in cli_args:
        cli_list_jobs()
    elif "--delete" in cli_args:
        delete_flag_index = cli_args.index("--delete")
        try:
            job_number_to_delete = int(cli_args[delete_flag_index + 1])
        except (IndexError, ValueError):
            print("Usage: python match_llm.py --delete <job number> (see --list for numbers)")
        else:
            cli_delete_job(job_number_to_delete)
    else:
        # --fresh ignores the results.json cache and re-scores every
        # job with Ollama, regardless of whether it's been scored before.
        use_cache = "--fresh" not in cli_args
        run_scoring_and_report(use_cache)
