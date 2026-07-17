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

# `datetime` is used to stamp today's date onto the HTML report title.
import datetime

# The file containing multiple job postings, separated by a line with
# just "---". See jobs.txt in this same folder for the example jobs.
JOBS_FILE = "jobs.txt"

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
    Read `file_path` and split its contents into a list of separate job
    posting strings, wherever a line contains just "---".

    Each job's leading/trailing blank lines are stripped with .strip(),
    so the job text is clean before we hand it to score_job().
    """
    with open(file_path, "r", encoding="utf-8") as f:
        file_contents = f.read()

    raw_jobs = file_contents.split("---")
    return [job.strip() for job in raw_jobs if job.strip()]


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
    green (>= 70), orange (50-69), red (< 50).
    """
    if score >= 70:
        return "#1b8a3d"  # green
    elif score >= 50:
        return "#c77700"  # orange
    else:
        return "#c53030"  # red


def build_html_report(ranked_results):
    """
    Build a full HTML page (as a string) showing one "card" per job,
    ranked high to low, with the score color-coded and the drafted
    proposal shown for high-scoring jobs.

    `ranked_results` is a list of dicts, one per job, each with keys:
    "rank", "score", "title", "verdict", "proposal" (proposal is None
    for jobs that didn't get one drafted).

    We use html.escape() on every piece of text that came from the job
    posting or the model's reply, since that text is unpredictable and
    could otherwise contain characters (like "<" or "&") that would
    break the HTML page's structure.
    """
    today_str = datetime.date.today().strftime("%B %d, %Y")

    cards_html = ""
    for entry in ranked_results:
        color = score_to_color(entry["score"])

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

        cards_html += f"""
        <div class="card">
            <div class="card-header">
                <span class="rank">#{entry['rank']}</span>
                <span class="score" style="color: {color};">{entry['score']}/100</span>
            </div>
            <h2 class="job-title">{html.escape(entry['title'])}</h2>
            <p class="verdict">{html.escape(entry['verdict'])}</p>
            {proposal_html}
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Upwork Job Matches</title>
<style>
    body {{
        font-family: Arial, Helvetica, sans-serif;
        background-color: #f4f5f7;
        color: #222;
        max-width: 800px;
        margin: 40px auto;
        padding: 0 20px;
    }}
    h1 {{
        font-size: 24px;
        margin-bottom: 24px;
    }}
    .card {{
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }}
    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 18px;
        font-weight: bold;
    }}
    .rank {{
        color: #555;
    }}
    .score {{
        font-size: 20px;
    }}
    .job-title {{
        font-size: 17px;
        margin: 10px 0 6px 0;
    }}
    .verdict {{
        margin: 0 0 10px 0;
        color: #444;
    }}
    .proposal-box {{
        background-color: #f9f9f9;
        border-left: 4px solid #888;
        border-radius: 4px;
        padding: 12px 16px;
        margin-top: 10px;
    }}
    .proposal-label {{
        font-size: 13px;
        font-weight: bold;
        text-transform: uppercase;
        color: #666;
        margin-bottom: 6px;
    }}
</style>
</head>
<body>
    <h1>Upwork Job Matches &mdash; {today_str}</h1>
    {cards_html}
</body>
</html>
"""


# The `if __name__ == "__main__":` line below means: "only run this code
# when this file is executed directly (e.g. `python match_llm.py`), not
# when it's imported by another file."
if __name__ == "__main__":
    jobs = load_jobs(JOBS_FILE)

    print(f"Loaded {len(jobs)} jobs from {JOBS_FILE}. Scoring each with {MODEL_NAME}...\n")

    # Score every job, and keep (score, verdict, job_text) together so we
    # can sort and print them afterwards.
    scored_jobs = []
    for job_text in jobs:
        raw_reply = score_job(job_text)
        score, verdict = parse_score_and_verdict(raw_reply)
        scored_jobs.append((score, verdict, job_text))

    # Sort highest score first. key=lambda item: item[0] tells sort() to
    # compare by the score (the first element of each tuple); reverse=True
    # makes it descending instead of ascending.
    scored_jobs.sort(key=lambda item: item[0], reverse=True)

    print("===== Ranked Job Matches =====\n")

    # Collect one dict per job as we go, so we can hand the same data to
    # build_html_report() after the terminal printing is done.
    ranked_results = []

    for rank, (score, verdict, job_text) in enumerate(scored_jobs, start=1):
        # Use the job's first line as a short title in the output.
        job_title = job_text.splitlines()[0].strip()
        print(f"#{rank} - SCORE: {score} - {job_title}")
        print(f"   Verdict: {verdict}")

        # Only draft (and print) a proposal for high-scoring jobs - it's
        # not worth writing one for a job I'm a poor fit for.
        proposal = None
        if score >= PROPOSAL_SCORE_THRESHOLD:
            proposal = draft_proposal(job_text)
            print("\n   Drafted proposal:")
            print(f"   {proposal}")

        print()

        ranked_results.append(
            {
                "rank": rank,
                "score": score,
                "title": job_title,
                "verdict": verdict,
                "proposal": proposal,
            }
        )

    # Also save the same ranking as a readable HTML report.
    report_html = build_html_report(ranked_results)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_html)

    print(f"HTML report saved to {REPORT_FILE}")
